"""
annotation.py
=============
Add QuPath GeoJSON annotations to a SpatialData object produced by
load_and_align().

The preferred workflow is to pass geojson_path directly to load_and_align()
so annotations are transformed at registration time.  Use
add_qupath_annotations() when you want to add annotations to an already-built
sdata object (e.g. loaded from disk).

The affine matrix stored in sdata['maldi_adata'].uns['he_transform']['affine_matrix']
is a 3x3 matrix mapping reg-resolution H&E coords -> buffer canvas coords.
It was derived empirically by _fit_affine_from_pil() in auto_align.py, which
fits a least-squares affine from a dense grid of point correspondences computed
through PIL's exact rotation geometry.  This is robust to PIL's internal
conventions about expand=True, rotation centre, and canvas placement.

annotation.py then folds in the scale_to_reg and img_upscaling factors to
produce the final native-H&E-pixel -> upscaled-canvas transform.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Union

import numpy as np
import geopandas as gpd
from shapely.geometry import shape
from shapely import transform as shapely_transform

from spatialdata import SpatialData
from spatialdata.models import ShapesModel
from spatialdata.transformations import Identity


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _require_transform(sdata: SpatialData) -> dict:
    """
    Retrieve he_transform from sdata['maldi_adata'].uns.
    Raises a clear, actionable error if it is missing or incomplete.
    """
    tf = sdata["maldi_adata"].uns.get("he_transform")
    if tf is None:
        raise KeyError(
            "sdata['maldi_adata'].uns['he_transform'] not found.\n"
            "Re-run load_and_align() to generate it, or pass geojson_path= "
            "directly to load_and_align() to avoid this step entirely."
        )
    if "affine_matrix" not in tf or tf["affine_matrix"] is None:
        raise KeyError(
            "he_transform is missing 'affine_matrix'.\n"
            "Re-run load_and_align() with the updated auto_align.py to "
            "regenerate the transform with the empirically fitted affine "
            "matrix included."
        )
    return tf


def _build_full_matrix(
    he_pixel_um: float,
    reg_mpp: float,
    affine_matrix: np.ndarray,
    img_upscaling: int,
) -> np.ndarray:
    """
    Combine the three transform steps into one 3x3 matrix:

        scale_to_reg  ->  affine_matrix (reg-res -> canvas)  ->  upscale

    The stored affine_matrix maps reg-resolution coords -> canvas coords.
    Here we prepend the scale_to_reg step and append the upscaling step so
    the result maps native H&E pixel coords -> upscaled canvas coords.

    Parameters
    ----------
    he_pixel_um   : native H&E pixel size (um/px)
    reg_mpp       : registration resolution (um/px)
    affine_matrix : 3x3 float64 from he_transform['affine_matrix']
    img_upscaling : upscaling factor

    Returns
    -------
    M : np.ndarray shape (3, 3), dtype float64
    """
    scale_to_reg = he_pixel_um / reg_mpp

    M_scale = np.array([
        [scale_to_reg, 0,            0],
        [0,            scale_to_reg, 0],
        [0,            0,            1],
    ], dtype=np.float64)

    us = float(img_upscaling)
    M_up = np.array([
        [us, 0,  0],
        [0,  us, 0],
        [0,  0,  1],
    ], dtype=np.float64)

    return M_up @ affine_matrix @ M_scale


def _apply_matrix_to_geojson(
    geojson_path: Union[str, Path],
    M: np.ndarray,
    classification_key: str = "classification",
) -> gpd.GeoDataFrame:
    """
    Load a QuPath GeoJSON file and apply a 3x3 affine matrix to all geometries.

    Parameters
    ----------
    geojson_path      : path to QuPath GeoJSON (native H&E pixel coords)
    M                 : 3x3 affine, native H&E px -> upscaled canvas coords
    classification_key: column name for QuPath class labels
    """
    with open(geojson_path, "r") as f:
        geojson = json.load(f)

    features = geojson if isinstance(geojson, list) else geojson.get("features", [])
    if not features:
        raise ValueError(f"No features found in {geojson_path}")

    a, b, tx = M[0, 0], M[0, 1], M[0, 2]
    d, e, ty = M[1, 0], M[1, 1], M[1, 2]

    def _transform_coords(coords: np.ndarray) -> np.ndarray:
        x, y = coords[:, 0], coords[:, 1]
        return np.column_stack([
            a * x + b * y + tx,
            d * x + e * y + ty,
        ])

    geoms  = []
    labels = []
    names  = []

    for feat in features:
        geom_raw = feat.get("geometry")
        if geom_raw is None:
            continue

        geoms.append(shapely_transform(shape(geom_raw), _transform_coords))

        props = feat.get("properties") or {}
        clf   = props.get("classification") or {}
        label = clf.get("name", "unknown") if isinstance(clf, dict) else str(clf)
        labels.append(label)
        names.append(props.get("name", ""))

    return gpd.GeoDataFrame(
        {classification_key: labels, "name": names},
        geometry=geoms,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def add_qupath_annotations(sdata, geojson_path, shapes_key="annotations",
                            classification_key="classification", he_pixel_um=None):
    tf            = _require_transform(sdata)
    native_mpp    = float(he_pixel_um or tf["he_pixel_um"])
    reg_mpp       = float(tf["reg_mpp"])
    img_upscaling = int(tf["img_upscaling"])
    M_stored      = np.array(tf["affine_matrix"], dtype=np.float64)

    # Re-use the same function as load_and_align
    from .auto_align import _transform_geojson
    gdf = _transform_geojson(
        geojson_path       = geojson_path,
        he_pixel_um        = native_mpp,
        reg_mpp            = reg_mpp,
        M_stored           = M_stored,
        img_upscaling      = img_upscaling,
        classification_key = classification_key,
    )

    sdata.shapes[shapes_key] = ShapesModel.parse(
        gdf, transformations={"global": Identity()}
    )
    print(f"  Added {len(gdf)} annotations -> sdata.shapes['{shapes_key}']")
    print(f"  Classes: {gdf[classification_key].unique().tolist()}")
    return sdata