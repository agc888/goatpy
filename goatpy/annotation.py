"""
annotations.py
==============
Add QuPath GeoJSON annotations to a SpatialData object produced by
load_and_align().

The preferred workflow is to pass geojson_path directly to load_and_align()
so annotations are transformed at registration time.  Use
add_qupath_annotations() when you want to add annotations to an already-built
sdata object (e.g. loaded from disk).

The affine matrix stored in sdata['maldi_adata'].uns['he_transform']['affine_matrix']
encodes the complete transform:
    native H&E px -> scale to reg_mpp -> PIL rotation -> canvas placement -> upscale
It was derived from PIL's actual rotate() output during load_and_align(), so
it exactly matches the H&E image transform with no reconstruction error.
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
    Raises a clear error if it is missing or lacks required keys.
    """
    tf = sdata["maldi_adata"].uns.get("he_transform")
    if tf is None:
        raise KeyError(
            "sdata['maldi_adata'].uns['he_transform'] not found.\n"
            "Re-run load_and_align() to generate it, or pass geojson_path= "
            "directly to load_and_align() to avoid this step entirely."
        )
    if "affine_matrix" not in tf:
        raise KeyError(
            "he_transform is missing 'affine_matrix'.\n"
            "Re-run load_and_align() with the updated auto_align.py to "
            "regenerate the transform with the affine matrix included."
        )
    return tf


def _apply_affine_to_geojson(geojson_path: Union[str, Path],
                              scale_to_reg: float,
                              affine_matrix: np.ndarray,
                              classification_key: str = "classification",
                              ) -> gpd.GeoDataFrame:
    """
    Load a QuPath GeoJSON file and transform all geometries into the
    final upscaled canvas coordinate system using the stored affine matrix.

    The full transform is:
        1. Scale by scale_to_reg  (he_pixel_um / reg_mpp)
        2. Apply affine_matrix    (rotation + canvas placement + upscaling)

    Both steps are combined into a single matrix multiply so each vertex
    is transformed exactly once with no intermediate rounding.

    Parameters
    ----------
    geojson_path      : path to QuPath GeoJSON (native H&E pixel coords)
    scale_to_reg      : he_pixel_um / reg_mpp
    affine_matrix     : 3x3 float64 matrix from he_transform['affine_matrix']
    classification_key: column name for QuPath class labels
    """
    with open(geojson_path, "r") as f:
        geojson = json.load(f)

    features = geojson if isinstance(geojson, list) else geojson.get("features", [])
    if not features:
        raise ValueError(f"No features found in {geojson_path}")

    # Fold scale_to_reg into affine_matrix for a single-pass transform
    M_scale_reg = np.array([
        [scale_to_reg, 0,            0],
        [0,            scale_to_reg, 0],
        [0,            0,            1],
    ], dtype=np.float64)
    M = affine_matrix @ M_scale_reg

    a, b, tx = M[0, 0], M[0, 1], M[0, 2]
    d, e, ty = M[1, 0], M[1, 1], M[1, 2]

    def _transform_coords(coords: np.ndarray) -> np.ndarray:
        x = coords[:, 0]
        y = coords[:, 1]
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

        geom    = shape(geom_raw)
        geom_tf = shapely_transform(geom, _transform_coords)
        geoms.append(geom_tf)

        props = feat.get("properties") or {}
        clf   = props.get("classification") or {}
        label = clf.get("name", "unknown") if isinstance(clf, dict) else str(clf)
        name  = props.get("name", "")
        labels.append(label)
        names.append(name)

    return gpd.GeoDataFrame(
        {classification_key: labels, "name": names},
        geometry=geoms,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def add_qupath_annotations(
    sdata: SpatialData,
    geojson_path: Union[str, Path],
    shapes_key: str = "annotations",
    classification_key: str = "classification",
    he_pixel_um: Optional[float] = None,
) -> SpatialData:
    """
    Read a QuPath GeoJSON annotation file and add it to an existing sdata
    object in the same coordinate system as the registered H&E image.

    Uses the affine matrix stored in sdata['maldi_adata'].uns['he_transform']
    which was derived from PIL's actual rotate() output during load_and_align().
    This guarantees exact alignment with the H&E image.

    Requires sdata to have been produced by a recent version of load_and_align()
    that stores 'affine_matrix' in he_transform.  If that key is missing,
    re-run load_and_align() with the updated auto_align.py.

    Parameters
    ----------
    sdata : SpatialData
        Object produced by load_and_align().
    geojson_path : str or Path
        Path to the QuPath GeoJSON export.  Coordinates must be in native
        H&E pixel space (QuPath default).
    shapes_key : str, default "annotations"
        Key under which annotations are stored in sdata.shapes.
    classification_key : str, default "classification"
        Column name for the QuPath class label in the GeoDataFrame.
    he_pixel_um : float or None
        Override the H&E native pixel size.  If None, uses the value stored
        in he_transform.

    Returns
    -------
    sdata with sdata.shapes[shapes_key] added in-place.
    """
    tf = _require_transform(sdata)

    native_mpp    = float(he_pixel_um or tf["he_pixel_um"])
    reg_mpp       = float(tf["reg_mpp"])
    scale_to_reg  = native_mpp / reg_mpp
    affine_matrix = np.array(tf["affine_matrix"], dtype=np.float64)

    gdf = _apply_affine_to_geojson(
        geojson_path       = geojson_path,
        scale_to_reg       = scale_to_reg,
        affine_matrix      = affine_matrix,
        classification_key = classification_key,
    )

    shapes_model = ShapesModel.parse(gdf, transformations={"global": Identity()})
    sdata.shapes[shapes_key] = shapes_model

    print(f"  Added {len(gdf)} annotations -> sdata.shapes['{shapes_key}']")
    print(f"  Classes: {gdf[classification_key].unique().tolist()}")

    return sdata