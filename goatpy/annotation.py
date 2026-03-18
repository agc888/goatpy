"""
annotations.py
==============
Functions for adding QuPath (or any GeoJSON-format) annotations to a
SpatialData object produced by load_and_align().

How the coordinate transform works
-----------------------------------
load_and_align() stores the following in sdata['maldi_adata'].uns['he_transform']:

    he_reg_size      : [rows, cols]  H&E size at reg_mpp BEFORE rotation
    canvas_placement : [pr, pc]      exact integer offsets where the rotated
                                     H&E top-left was placed in the canvas
    rotation_deg     : float         CCW rotation applied (PIL convention)
    he_pixel_um      : float         native H&E pixel size (µm/px)
    reg_mpp          : float         registration resolution (µm/px)
    img_upscaling    : int           upscaling factor

The pipeline applied to each annotation geometry is:

    native H&E px
        → scale to reg_mpp          (factor = he_pixel_um / reg_mpp)
        → rotate CCW around centre of pre-rotation H&E image
        → translate by canvas_placement  (pr, pc)
        → upscale by img_upscaling

canvas_placement values are the exact integers computed by _build_full_output
during load_and_align() — no floating-point reconstruction of PIL's arithmetic
is needed, which was the source of previous alignment errors.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Union

import geopandas as gpd
from shapely.geometry import shape
from shapely.affinity import rotate, scale as shapely_scale, translate

from spatialdata import SpatialData
from spatialdata.models import ShapesModel
from spatialdata.transformations import Identity


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _require_transform(sdata: SpatialData) -> dict:
    """
    Retrieve the he_transform dict stored by load_and_align().
    Raises a clear error if it is missing or incomplete.
    """
    tf = sdata["maldi_adata"].uns.get("he_transform")
    if tf is None:
        raise KeyError(
            "sdata['maldi_adata'].uns['he_transform'] not found.\n\n"
            "This is written automatically by load_and_align().  "
            "If you built the sdata object a different way, populate "
            "sdata['maldi_adata'].uns['he_transform'] with the keys:\n"
            "  rotation_deg, he_pixel_um, reg_mpp, img_upscaling,\n"
            "  he_reg_size ([rows, cols] before rotation),\n"
            "  canvas_placement ([pr, pc] offset in canvas)."
        )

    required = ["rotation_deg", "he_pixel_um", "reg_mpp",
                "img_upscaling", "he_reg_size", "canvas_placement"]
    missing = [k for k in required if k not in tf]
    if missing:
        raise KeyError(
            f"he_transform is missing keys: {missing}\n\n"
            "Re-run load_and_align() to regenerate the transform with all "
            "required fields.  Older sdata objects may lack he_reg_size and "
            "canvas_placement which were added to fix annotation alignment."
        )
    return tf


def _transform_geometry(geom,
                         he_pixel_um: float,
                         reg_mpp: float,
                         rotation_deg: float,
                         he_reg_h: int,
                         he_reg_w: int,
                         canvas_pr: int,
                         canvas_pc: int,
                         img_upscaling: int):
    """
    Apply the same coordinate pipeline used by load_and_align() to a single
    shapely geometry whose coordinates are in native H&E pixel space.

    Pipeline
    --------
    1. Scale  : native H&E px → registration resolution
                factor = he_pixel_um / reg_mpp

    2. Rotate : CCW by rotation_deg around the centre of the H&E image at
                reg_mpp resolution — (he_reg_w/2, he_reg_h/2).
                This matches PIL's rotate() which rotates around the image
                centre before expand=True enlarges the canvas.

    3. Translate : shift by (canvas_pc, canvas_pr) — the exact integer offsets
                stored in he_transform['canvas_placement'].  These are the
                top-left position of the rotated H&E image inside the canvas,
                computed with integer // 2 arithmetic by _build_full_output.
                Using the stored values directly avoids any floating-point
                discrepancy from trying to reconstruct PIL's arithmetic.

    4. Upscale : multiply all coords by img_upscaling to match the stored
                 H&E image in sdata.images['he_image'].

    Parameters
    ----------
    geom         : shapely geometry in native H&E pixel coordinates
    he_pixel_um  : H&E native pixel size (µm/px)
    reg_mpp      : registration resolution (µm/px)
    rotation_deg : CCW rotation in degrees (PIL convention)
    he_reg_h     : H&E image height at reg_mpp, BEFORE rotation (rows)
    he_reg_w     : H&E image width  at reg_mpp, BEFORE rotation (cols)
    canvas_pr    : row offset where rotated H&E top-left sits in canvas
    canvas_pc    : col offset where rotated H&E top-left sits in canvas
    img_upscaling: integer upscaling factor

    Returns
    -------
    Transformed shapely geometry in the upscaled canvas coordinate system.
    """
    # 1. Scale to registration resolution
    s = he_pixel_um / reg_mpp
    geom = shapely_scale(geom, xfact=s, yfact=s, origin=(0, 0))

    # 2. Rotate around the centre of the pre-rotation H&E image.
    #    PIL rotates around (width/2, height/2) — shapely uses (x, y) = (col, row).
    cx = he_reg_w / 2.0
    cy = he_reg_h / 2.0
    geom = rotate(geom, angle=rotation_deg, origin=(cx, cy), use_radians=False)

    # 3. Translate by the canvas placement offset.
    #    After PIL rotates with expand=True, the rotated image is placed at
    #    (pc, pr) in the canvas.  Our shapely rotation left coords relative
    #    to the same origin, so we just shift by (pc, pr).
    geom = translate(geom, xoff=canvas_pc, yoff=canvas_pr)

    # 4. Upscale to match the stored H&E image resolution
    geom = shapely_scale(geom, xfact=img_upscaling, yfact=img_upscaling,
                          origin=(0, 0))

    return geom


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def add_annotations(
    sdata: SpatialData,
    geojson_path: Union[str, Path],
    shapes_key: str = "annotations",
    classification_key: str = "classification",
    he_pixel_um: Optional[float] = None,
) -> SpatialData:
    """
    Read a QuPath GeoJSON annotation file and add the annotations to
    sdata as a ShapesModel in the same coordinate system as the registered
    H&E image.

    Requires sdata to have been produced by load_and_align() so that
    sdata['maldi_adata'].uns['he_transform'] contains he_reg_size and
    canvas_placement.  If you have an older sdata object, re-run
    load_and_align() to regenerate it.

    Parameters
    ----------
    sdata : SpatialData
        Object produced by load_and_align().
    geojson_path : str or Path
        Path to the QuPath GeoJSON export.  Coordinates must be in native
        H&E pixel space (as QuPath exports them by default).
    shapes_key : str, default "annotations"
        Key under which annotations are stored in sdata.shapes.
        Use different keys to add multiple annotation files:
            add_qupath_annotations(sdata, "tumour.geojson",  shapes_key="tumour")
            add_qupath_annotations(sdata, "stroma.geojson",  shapes_key="stroma")
    classification_key : str, default "classification"
        Column name in the resulting GeoDataFrame for the QuPath class label.
    he_pixel_um : float or None
        Override the H&E native pixel size.  If None, uses the value stored
        in sdata['maldi_adata'].uns['he_transform']['he_pixel_um'].

    Returns
    -------
    sdata : SpatialData
        The same object with sdata.shapes[shapes_key] added in-place.
    """
    tf = _require_transform(sdata)

    # Read transform parameters
    rotation_deg  = float(tf["rotation_deg"])
    reg_mpp       = float(tf["reg_mpp"])
    img_upscaling = int(tf["img_upscaling"])
    native_mpp    = float(he_pixel_um or tf["he_pixel_um"])

    # H&E size at reg_mpp BEFORE rotation — defines the rotation centre
    he_reg_h, he_reg_w = int(tf["he_reg_size"][0]), int(tf["he_reg_size"][1])

    # Exact canvas placement offset stored by _build_full_output
    canvas_pr, canvas_pc = int(tf["canvas_placement"][0]), int(tf["canvas_placement"][1])

    # Read GeoJSON
    geojson_path = Path(geojson_path)
    with open(geojson_path, "r") as f:
        geojson = json.load(f)

    features = geojson if isinstance(geojson, list) else geojson.get("features", [])
    if not features:
        raise ValueError(f"No features found in {geojson_path}")

    geoms  = []
    labels = []
    names  = []

    for feat in features:
        geom_raw = feat.get("geometry")
        if geom_raw is None:
            continue

        geom = shape(geom_raw)

        geom_tf = _transform_geometry(
            geom,
            he_pixel_um  = native_mpp,
            reg_mpp      = reg_mpp,
            rotation_deg = rotation_deg,
            he_reg_h     = he_reg_h,
            he_reg_w     = he_reg_w,
            canvas_pr    = canvas_pr,
            canvas_pc    = canvas_pc,
            img_upscaling = img_upscaling,
        )

        geoms.append(geom_tf)

        props = feat.get("properties") or {}
        clf   = props.get("classification") or {}
        label = clf.get("name", "unknown") if isinstance(clf, dict) else str(clf)
        name  = props.get("name", "")
        labels.append(label)
        names.append(name)

    gdf = gpd.GeoDataFrame(
        {
            classification_key: labels,
            "name":             names,
        },
        geometry=geoms,
    )

    shapes_model = ShapesModel.parse(gdf, transformations={"global": Identity()})
    sdata.shapes[shapes_key] = shapes_model

    print(f"  Added {len(gdf)} annotations → sdata.shapes['{shapes_key}']")
    unique_labels = gdf[classification_key].unique().tolist()
    print(f"  Classes: {unique_labels}")

    return sdata