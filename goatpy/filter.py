"""
filter.py
=========
A wrapper around spatialdata's filter_by_table_query that preserves
images, points, and non-pixel shapes rather than dropping them.

Usage
-----
>>> from goatpy.filter import filter_spatialdata

# Filter by any obs column
>>> sub = filter_spatialdata(sdata, "annotation == 'Tumor'")
>>> sub = filter_spatialdata(sdata, "GPCA_clusters == '3'")
>>> sub = filter_spatialdata(sdata, "MPI > 1000")

# Filter by ion intensity (var_names)
>>> sub = filter_spatialdata(sdata, "1581.6 > 500", on="expression")

# Keep points unsubsetted (just retain all centroids)
>>> sub = filter_spatialdata(sdata, "annotation == 'Tumor'", subset_points=False)
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

from spatialdata import SpatialData
from spatialdata.models import PointsModel, ShapesModel, TableModel
from spatialdata.transformations import get_transformation, set_transformation


def filter_spatialdata(
    sdata: SpatialData,
    query: str,
    on: Literal["obs", "expression"] = "obs",
    table_name: str = "maldi_adata",
    subset_points: bool = True,
    subset_annotations: bool = False,
) -> SpatialData:
    """
    Subset a SpatialData object by a query string, preserving images, points,
    and annotation shapes.

    Internally builds a boolean mask from the query, filters the table and
    pixel shapes consistently, then re-attaches images, centroids, and
    annotation polygons so nothing is silently dropped.

    Parameters
    ----------
    sdata : SpatialData
    query : str
        A pandas ``.query()`` string.
    on : "obs" | "expression"
        ``"obs"``        — query runs against ``maldi_adata.obs`` columns
                           (annotation, GPCA_clusters, MPI, he_x, he_y …).
        ``"expression"`` — query runs against ion intensity columns
                           (m/z var_names, dots auto-sanitised to underscores).
    table_name : str
        Default ``"maldi_adata"``.
    subset_points : bool
        If True (default), centroids are subsetted to match the filtered pixels.
        If False, all original centroids are kept as-is.
    subset_annotations : bool
        If True, annotation polygon shapes are filtered to only keep classes
        that are present in the filtered pixels' ``annotation`` obs column.
        If False (default), all annotation polygons are kept as-is.

    Returns
    -------
    SpatialData with:
        - ``images``      — always kept unchanged
        - ``shapes["pixels"]``  — subsetted to matching pixels
        - ``shapes["annotations"]`` — kept or filtered depending on subset_annotations
        - ``points["centroids"]``   — subsetted or kept depending on subset_points
        - ``tables[table_name]``    — subsetted to matching rows

    Examples
    --------
    >>> filter_spatialdata(sdata, "annotation == 'Tumor'")
    >>> filter_spatialdata(sdata, "GPCA_clusters == '3'")
    >>> filter_spatialdata(sdata, "MPI > 1000")
    >>> filter_spatialdata(sdata, "annotation == 'Tumor' and MPI > 500")
    >>> filter_spatialdata(sdata, "1581.6 > 500", on="expression")
    """
    adata = sdata.tables[table_name]
    n = len(adata)

    if on == "obs":
        obs = adata.obs.copy()
        for col in obs.select_dtypes("category").columns:
            obs[col] = obs[col].astype(str)
        try:
            matched = obs.query(query).index
        except Exception as e:
            raise ValueError(
                f"query '{query}' failed on obs: {e}\n"
                f"Available columns: {list(obs.columns)}"
            ) from e
        isin = adata.obs.index.isin(matched)
        mask = isin.to_numpy() if hasattr(isin, "to_numpy") else np.asarray(isin)

    elif on == "expression":
        X = np.asarray(adata.X, dtype=np.float32)
        var_names = list(adata.var_names)
        # Prefix with 'mz_' and sanitise punctuation so pandas query accepts the
        # column names (column names starting with digits are not valid identifiers)
        safe = {
            v: "mz_" + v.replace(".", "_").replace("-", "_").replace(" ", "_")
            for v in var_names
        }
        df = pd.DataFrame(X, columns=[safe[v] for v in var_names])
        safe_query = query
        for orig, s in sorted(safe.items(), key=lambda x: -len(x[0])):
            safe_query = safe_query.replace(orig, s)
        try:
            matched = df.query(safe_query).index
        except Exception as e:
            raise ValueError(
                f"query '{query}' failed on expression: {e}\n"
                f"Available m/z: {var_names}"
            ) from e
        mask = np.zeros(n, dtype=bool)
        mask[matched] = True

    else:
        raise ValueError(f"on= must be 'obs' or 'expression', got '{on}'")

    n_kept = int(mask.sum())
    print(f"  {n_kept:,} / {n:,} pixels selected ({n_kept / n * 100:.1f}%)")

    if n_kept == 0:
        raise ValueError("Query matches 0 pixels.")

    pos_idx = np.where(mask)[0]


    adata_sub = adata[mask].copy()
    adata_sub.uns.pop("spatialdata_attrs", None)
    adata_sub.obs["region"]      = "pixels"
    adata_sub.obs["region"]      = adata_sub.obs["region"].astype("category")
    adata_sub.obs["instance_id"] = np.arange(n_kept).astype(str)


    def _strip(gdf):
        gdf = gdf.copy()
        gdf.attrs = {}
        return gdf

    pix = _strip(sdata.shapes["pixels"].iloc[pos_idx].reset_index(drop=True))
    pix.index = adata_sub.obs["instance_id"].values

    shapes_out = {
        "pixels": ShapesModel.parse(pix, transformations={"global": _get_identity_transform(sdata, "pixels")})
    }


    for key, gdf in sdata.shapes.items():
        if key == "pixels":
            continue

        gdf_out = _strip(gdf)

        if subset_annotations and "annotation" in adata_sub.obs.columns:
            present = set(adata_sub.obs["annotation"].astype(str).unique())
            for col in ("classification", "annotation"):
                if col in gdf_out.columns:
                    gdf_out = gdf_out[gdf_out[col].astype(str).isin(present)].copy()
                    break

        if len(gdf_out) == 0:
            continue

        shapes_out[key] = ShapesModel.parse(
            gdf_out, transformations={"global": _get_identity_transform(sdata, key)}
        )


    pts = sdata.points["centroids"]
    pts_df = pts.compute() if hasattr(pts, "compute") else pts.copy()
    pts_df = pts_df.reset_index(drop=True)

    if subset_points:
        pts_out = _strip(pts_df.iloc[pos_idx].reset_index(drop=True))
    else:
        pts_out = _strip(pts_df)

    points_out = {
        "centroids": PointsModel.parse(
            pts_out, transformations={"global": _get_identity_transform(sdata, "centroids")}
        )
    }

    images_out = dict(sdata.images)

    from spatialdata import SpatialData as SD

    sdata_sub = SD(
        images=images_out,
        points=points_out,
        shapes=shapes_out,
    )

    sdata_sub[table_name] = TableModel.parse(
        adata_sub,
        region="pixels",
        region_key="region",
        instance_key="instance_id",
    )

    print(f"  Done. Result: {sdata_sub}")
    return sdata_sub


def _get_identity_transform(sdata: SpatialData, element_name: str):
    """
    Re-use the existing global transform for an element if available,
    otherwise return a fresh Identity.
    """
    from spatialdata.transformations import Identity
    try:
        element = sdata[element_name]
        t = get_transformation(element, to_coordinate_system="global")
        return t
    except Exception:
        return Identity()