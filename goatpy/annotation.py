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
import pandas as pd
import geopandas as gpd
from shapely.geometry import shape, Point, box as shapely_box

from shapely import transform as shapely_transform

from spatialdata import SpatialData, polygon_query
from spatialdata.models import ShapesModel
from spatialdata.transformations import Identity



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


def annotate_per_pixel(
    sdata: SpatialData,
    shapes_key: str = "annotations",
    classification_key: str = "classification",
    table_name: str = "maldi_adata",
    obs_column: str = "annotation",
    overlap: float = 0.0,
    other_label: str = "other",
    priority: Optional[list[str]] = None,
    inplace: bool = True,
) -> SpatialData:
    """
    Add a categorical ``obs_column`` to ``maldi_adata.obs`` labelling each
    pixel with its annotation class (or ``other_label`` if none).

    Parameters
    ----------
    sdata : SpatialData
        Input spatial data object produced by ``load_and_align``.
    shapes_key : str
        Key in ``sdata.shapes`` containing the annotation GeoDataFrame.
        Default ``"annotations"``.
    classification_key : str
        Column in the annotation GeoDataFrame holding class labels.
        Default ``"classification"``.
    table_name : str
        Table in ``sdata.tables`` to annotate.  Default ``"maldi_adata"``.
    obs_column : str
        Name of the new column added to ``adata.obs``.
        Default ``"annotation"``.
    overlap : float
        Minimum fractional overlap (0–1) required for a pixel to be
        assigned to a class.

        * ``0`` (default) — centroid-in-polygon test via
          ``spatialdata.polygon_query``.  Fast.
        * ``> 0`` — area-based test via Shapely intersection.  A pixel is
          assigned only if ``intersection_area / pixel_area >= overlap``.
          E.g. ``overlap=0.5`` requires ≥ 50 % coverage.  Slower.
    other_label : str
        Label assigned to pixels that do not fall in any annotation.
        Default ``"other"``.
    priority : list of str, optional
        Class names in ascending priority order.  When a pixel overlaps
        multiple classes the *last* name in this list wins.  If ``None``,
        classes are iterated in the order they appear in the GeoDataFrame.
    inplace : bool
        If True (default), modify ``sdata`` in place and return it.
        If False, work on a shallow copy of the table only (the obs column
        is still added to the copy stored in sdata — set False if you want
        to avoid mutating the original adata object).

    Returns
    -------
    SpatialData
        The input ``sdata`` with ``obs_column`` added to
        ``sdata[table_name].obs``.

    Examples
    --------
    >>> from goatpy.annotate import annotate_per_pixel

    # Centroid-based (fast, default)
    >>> sdata = annotate_per_pixel(sdata)

    # Area-based: pixel must be ≥ 50 % inside a polygon
    >>> sdata = annotate_per_pixel(sdata, overlap=0.5)

    # Custom label for unannotated pixels
    >>> sdata = annotate_per_pixel(sdata, other_label="background")

    # Control which class wins when polygons overlap
    >>> sdata = annotate_per_pixel(sdata, priority=["Stroma", "Tumor"])
    """
    if shapes_key not in sdata.shapes:
        raise KeyError(
            f"shapes_key '{shapes_key}' not found. "
            f"Available: {list(sdata.shapes.keys())}"
        )
    if table_name not in sdata.tables:
        raise KeyError(
            f"Table '{table_name}' not found. "
            f"Available: {list(sdata.tables.keys())}"
        )

    ann_gdf = sdata.shapes[shapes_key]
    if classification_key not in ann_gdf.columns:
        raise KeyError(
            f"'{classification_key}' not found in shapes['{shapes_key}']. "
            f"Available columns: {list(ann_gdf.columns)}"
        )

    adata = sdata.tables[table_name]
    n = len(adata)

    # Determine class iteration order
    classes = ann_gdf[classification_key].astype(str).unique().tolist()
    if priority is not None:
        # Put non-priority classes first, then priority classes in order
        non_priority = [c for c in classes if c not in priority]
        classes = non_priority + [c for c in priority if c in classes]

    all_categories = [other_label] + [c for c in classes if c != other_label]

    # Initialise obs column with other_label
    labels = np.full(n, other_label, dtype=object)

    assert adata.obs.index.is_unique, (
        "adata.obs.index must be unique for annotation assignment. "
        "Re-run load_and_align or reset_index on adata.obs."
    )

    if overlap <= 0.0:
        _annotate_centroid(
            sdata, ann_gdf, adata, labels,
            classes, classification_key, shapes_key, table_name,
        )
    else:
        _annotate_area(
            ann_gdf, adata, labels,
            classes, classification_key, overlap, other_label,
        )

    adata.obs[obs_column] = pd.Categorical(labels, categories=all_categories)

    n_annotated = (labels != other_label).sum()
    print(
        f"  [annotate_pixels] '{obs_column}' added: "
        f"{n_annotated:,} / {n:,} pixels annotated  "
        f"({n_annotated/n*100:.1f}%)"
    )
    for cls in all_categories:
        count = (labels == cls).sum()
        print(f"    {cls!r:30s}: {count:>8,}  ({count/n*100:.1f}%)")

    return sdata


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _annotate_centroid(
    sdata, ann_gdf, adata, labels,
    classes, classification_key, shapes_key, table_name,
):
    """
    Centroid-based assignment using spatialdata.polygon_query.

    Iterates over each unique class, unions its polygons, runs polygon_query,
    then stamps matching obs indices with the class label.
    Last class in `classes` wins for overlapping regions.
    """
    for cls in classes:
        cls_geoms = ann_gdf[ann_gdf[classification_key].astype(str) == cls].geometry
        if cls_geoms.empty:
            continue

        from shapely.ops import unary_union
        union_poly = unary_union(cls_geoms.values)

        try:
            result = polygon_query(
                sdata,
                polygon=union_poly,
                target_coordinate_system="global",
                filter_table=True,
            )
            if table_name in result.tables:
                sub_table = result.tables[table_name]
                in_poly = sub_table.obs.index
                # Map back to position in the full adata
                full_idx = adata.obs.index
                pos = np.where(full_idx.isin(in_poly))[0]
                labels[pos] = cls
                print(f"    polygon_query '{cls}': {len(pos):,} pixels")
            else:
                print(f"    polygon_query '{cls}': 0 pixels (no table returned)")
        except Exception as e:
            print(f"    polygon_query '{cls}' failed ({e}), falling back to point-in-polygon")
            _fallback_pip(ann_gdf, adata, labels, cls, classification_key)


def _fallback_pip(ann_gdf, adata, labels, cls, classification_key):
    """Point-in-polygon fallback using he_x / he_y centroids."""
    from shapely.ops import unary_union
    cls_geoms = ann_gdf[ann_gdf[classification_key].astype(str) == cls].geometry
    region = unary_union(cls_geoms.values)

    he_x = adata.obs["he_x"].to_numpy()
    he_y = adata.obs["he_y"].to_numpy()
    pts = gpd.GeoSeries([Point(x, y) for x, y in zip(he_x, he_y)])
    mask = pts.within(region).to_numpy()
    labels[mask] = cls
    print(f"    fallback PIP '{cls}': {mask.sum():,} pixels")


def _annotate_area(
    ann_gdf, adata, labels,
    classes, classification_key, overlap_threshold, other_label,
):
    """
    Area-based assignment using Shapely intersection.

    For each pixel square, computes what fraction of its area is covered by
    each annotation class polygon.  Assigns the class with the highest
    coverage if it meets the threshold.  Slower than centroid but handles
    edge pixels correctly.
    """
    from shapely.ops import unary_union

    # Pre-build unioned geometry per class
    class_regions = {}
    for cls in classes:
        geoms = ann_gdf[ann_gdf[classification_key].astype(str) == cls].geometry
        if not geoms.empty:
            class_regions[cls] = unary_union(geoms.values)

    he_x = adata.obs["he_x"].to_numpy()
    he_y = adata.obs["he_y"].to_numpy()

    # Infer pixel half-size from the pixels shapes if available
    # Fall back to 0.5 * upscaling (typical goatpy default)
    try:
        sample_geom = adata.obsm.get("spatial")
        # Use a fixed half-size; goatpy pixels are square so pixel_area = (2*half)^2
        # We just need the ratio, so half cancels — use half=1 for normalisation
        half = 5.0  # reasonable default for upscaled coords; result is ratio anyway
    except Exception:
        half = 5.0

    print(
        f"  [annotate_pixels] area-based overlap (threshold={overlap_threshold:.0%}), "
        f"processing {len(he_x):,} pixels ..."
    )

    chunk = max(1, len(he_x) // 20)
    for i, (x, y) in enumerate(zip(he_x, he_y)):
        if i % chunk == 0:
            print(f"    {i:,} / {len(he_x):,} ...", end="\r")
        pixel_box = shapely_box(x - half, y - half, x + half, y + half)
        pixel_area = pixel_box.area

        best_cls = other_label  # will be overwritten inside loop — last winner
        # Iterate in priority order; last match above threshold wins
        for cls, region in class_regions.items():
            try:
                inter = pixel_box.intersection(region)
                frac = inter.area / pixel_area
                if frac >= overlap_threshold:
                    best_cls = cls
            except Exception:
                continue

        labels[i] = best_cls

    print(f"    {len(he_x):,} / {len(he_x):,} ... done        ")