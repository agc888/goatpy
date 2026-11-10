"""
cell_maldi_deconvolution.py

Standalone sparse, non-negative deconvolution of MALDI pixels into single-cell
m/z intensities for a SpatialData object. It does NOT need `pixels_to_cells`
or any pre-built cell table.


One-line usage
--------------
    from cell_maldi_deconvolution import deconvolve_maldi_to_cells

    deconvolve_maldi_to_cells(sdata, model_type="morphology_gaussian", key_added="cell_decon")
    sdata.tables["cell_decon"]          # cells x m/z, with .obsm["spatial"]

Inputs (all read from sdata)
----------------------------
    sdata.shapes[pixel_layer]       MALDI pixel footprints (polygons)
    sdata.tables[table_name]        pixel x m/z intensities annotating pixel_layer
    sdata.shapes[cell_layer]        single-cell polygons (also the ONLY source of morphology)

Steps
-----
    1. Overlay cell polygons with pixel polygons (same method as pixels_to_cells:
       geopandas intersection) -> W (cells x pixels) of overlap areas.
    2. Remove cells with no pixel overlap (and cells below `min_coverage`).
    3. A = W^T, row-normalised so each pixel's weights over cells sum to 1.
    4. Fit  Y (pixels x m/z) ~= A @ X (cells x m/z)  with the chosen model_type.

model_type
----------
    "area_weighted"        Weighted mean of overlapping pixels (closed form baseline).
    "morphology_prior"     Cells take their morphology-group profile (closed form baseline).
    "nnls"                 Non-negative least squares (ridge only), initialised from the
                           area-weighted estimate. Ill-posed when cells outnumber pixels.
    "area_gaussian"        Gaussian NNLS regularised toward the area-weighted estimate.
    "morphology_gaussian"  Gaussian NNLS regularised toward the morphology-group profile
                           (default).

This is a benchmark-grade estimator, not ground-truth intracellular quantification.

Dependencies: numpy, scipy, pandas, scikit-learn, geopandas, anndata, spatialdata

---------------------------------------------------------------------------------------------------

Visualisation for the output of `deconvolve_maldi_to_cells` (the "cell_decon" table).

    compare_morphology_clusters(sdata, ...)   morphology groups: example cells + feature heatmap
    plot_pixel_pies(sdata, color, ...)        per-pixel pie charts of any categorical .obs column
    evaluate_deconvolution(sdata, ...)        pixel-level data vs deconvolved single-cell data
    plot_model_comparison(metrics)            bar chart from compare_models() metrics

All functions take the SpatialData object and the table name (default "cell_decon").

Note: "morphology_group" is the intermediate prior used by the "morphology_*" models; it
is not a cell-type label. To get cell types, cluster the deconvolved .X (e.g. scanpy leiden)
and pass that column to plot_pixel_pies().



"""

from __future__ import annotations

import warnings
from typing import Dict, Iterable, Optional, Sequence

import anndata as ad
import geopandas as gpd
import numpy as np
import pandas as pd
import scipy.sparse as sp
import matplotlib.pyplot as plt

from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


from matplotlib import colors as mcolors
from matplotlib.collections import PatchCollection, PolyCollection
from matplotlib.patches import Patch, Wedge
from scipy.cluster.hierarchy import dendrogram, linkage



MODEL_TYPES = (
    "area_weighted",
    "morphology_prior",
    "nnls",
    "area_gaussian",
    "morphology_gaussian",
)
_MORPHOLOGY_MODELS = {"morphology_prior", "morphology_gaussian"}
_OVERLAP_ARGS = {"pixel_layer", "cell_layer", "table_name", "coordinate_system", "layer", "min_coverage"}


def _check_model(model_type):
    if model_type not in MODEL_TYPES:
        raise ValueError(f"Unknown model_type {model_type!r}. Choose from {MODEL_TYPES}.")


# --------------------------------------------------------------------------- #
# Geometry: pixels x cells overlap
# --------------------------------------------------------------------------- #
def _dense_float32(X):
    if sp.issparse(X):
        return X.toarray().astype(np.float32, copy=False)
    return np.asarray(X, dtype=np.float32)


def _get_polygons(sdata, layer, coordinate_system):
    """Shapes layer as a GeoDataFrame of polygons in `coordinate_system` (index preserved)."""
    gdf = sdata.shapes[layer]
    if coordinate_system is not None:
        from spatialdata import transform

        gdf = transform(gdf, to_coordinate_system=coordinate_system)
    geom = gdf.geometry
    if (geom.geom_type == "Point").all() and "radius" in gdf.columns:  # circles -> polygons
        gdf = gdf.set_geometry(geom.buffer(gdf["radius"]))
    return gdf


def compute_pixel_overlap(
    sdata,
    *,
    pixel_layer: str = "pixels",
    cell_layer: str = "cell_segmentation",
    table_name: str = "maldi_adata",
    coordinate_system: Optional[str] = "aligned",
    layer: Optional[str] = None,
    min_coverage: float = 0.0,
    verbose: bool = True,
) -> dict:
    """
    Overlay cell polygons with MALDI pixel polygons and return everything the
    deconvolution needs. Cells with no overlap (or less than `min_coverage` of
    their area covered) are removed here.

    Pixels with non-finite intensities are excluded before the overlay, so a cell
    only counts as covered by pixels that actually have data.
    """
    # ---- intensities (pixels x m/z), from the MALDI table only -------------
    adata = sdata.tables[table_name]
    attrs = adata.uns["spatialdata_attrs"]
    region_key, instance_key = attrs["region_key"], attrs["instance_key"]

    adata_pix = adata[adata.obs[region_key] == pixel_layer]
    if adata_pix.n_obs == 0:
        raise ValueError(f"Table '{table_name}' has no rows annotating '{pixel_layer}'.")

    Y = _dense_float32(adata_pix.layers[layer] if layer is not None else adata_pix.X)
    ids = adata_pix.obs[instance_key].to_numpy()
    valid = np.isfinite(Y).all(axis=1)
    if not valid.all():
        warnings.warn(
            f"{int((~valid).sum())} pixels have NaN/inf intensities and were excluded."
        )
        Y, ids = Y[valid], ids[valid]
    var = adata_pix.var.copy()

    # ---- polygons in a common coordinate system ----------------------------
    pix = _get_polygons(sdata, pixel_layer, coordinate_system).reindex(ids)
    if pix.geometry.isna().any():
        raise ValueError("Some table rows have no matching shape in the pixel layer.")
    pix = pix.reset_index(drop=True)
    pix["pix_idx"] = np.arange(len(pix))

    cel = _get_polygons(sdata, cell_layer, coordinate_system)
    cell_ids = cel.index.to_numpy()
    if not cel.index.is_unique:
        raise ValueError(f"Index of shapes layer '{cell_layer}' is not unique.")
    cel = cel.reset_index(drop=True)
    cel["cell_idx"] = np.arange(len(cel))
    cell_area = cel.geometry.area.to_numpy()

    # ---- overlap areas -> sparse W (cells x pixels) -------------------------
    ov = gpd.overlay(
        cel[["cell_idx", "geometry"]],
        pix[["pix_idx", "geometry"]],
        how="intersection",
        keep_geom_type=True,
    )
    ov["area"] = ov.geometry.area
    ov = ov[ov["area"] > 0]

    W = sp.csr_matrix(
        (ov["area"].to_numpy(), (ov["cell_idx"].to_numpy(), ov["pix_idx"].to_numpy())),
        shape=(len(cel), len(pix)),
    )
    W.sum_duplicates()
    covered = np.asarray(W.sum(axis=1)).ravel()
    frac = np.clip(np.divide(covered, cell_area, out=np.zeros_like(covered), where=cell_area > 0), 0, 1)

    # ---- remove cells without pixel overlap ---------------------------------
    keep = (covered > 0) & (frac >= min_coverage)
    n_dropped = int((~keep).sum())
    if verbose and n_dropped:
        print(f"Removed {n_dropped}/{len(keep)} cells with no / insufficient MALDI pixel overlap.")
    if not keep.any():
        raise ValueError("No cells overlap any MALDI pixel. Check coordinate_system and layer names.")

    W = W[keep].tocsr()
    covered, frac, cell_area = covered[keep], frac[keep], cell_area[keep]
    kept_ids = cell_ids[keep]
    cells = gpd.GeoDataFrame(geometry=cel.geometry.to_numpy()[keep], index=pd.Index(kept_ids))

    # per-cell normalised weights (QC + compatibility with pixels_to_cells)
    W_norm = sp.csr_matrix(sp.diags(1.0 / covered) @ W)
    W_norm.eliminate_zeros()
    row_counts = np.diff(W_norm.indptr)
    max_w = np.maximum.reduceat(W_norm.data, W_norm.indptr[:-1])

    obs = pd.DataFrame(
        {
            "n_pixels": row_counts,
            "frac_cell_covered": frac,
            "max_pixel_weight": max_w,
            "cell_area": cell_area,
        }
    )
    obs["cell_id"] = kept_ids
    obs["region"] = cell_layer
    obs.index = obs["cell_id"].astype(str).to_numpy()

    centroids = cells.geometry.centroid
    xy = np.column_stack([centroids.x.to_numpy(), centroids.y.to_numpy()])

    return {
        "W_area": W,
        "W_norm": W_norm,
        "Y": Y,
        "pixel_ids": np.asarray(ids).astype(str),
        "cells": cells,  # kept cell polygons, same order as W rows
        "obs": obs,
        "var": var,
        "xy": xy,
        "cell_layer": cell_layer,
        "coordinate_system": coordinate_system,
        "pixel_layer": pixel_layer,
        "table_name": table_name,
        "source_layer": layer if layer is not None else "X",
    }


def _build_A(W_area, normalise_per_pixel=True, eps=1e-12):
    """A (pixels x cells) from overlap areas; optionally each pixel's weights sum to 1."""
    A = W_area.T.tocsr().astype(np.float32)
    if normalise_per_pixel:
        rowsum = np.asarray(A.sum(axis=1)).ravel()
        inv = np.zeros_like(rowsum, dtype=np.float32)
        ok = rowsum > eps
        inv[ok] = 1.0 / rowsum[ok]
        A = (sp.diags(inv) @ A).tocsr()
    return A


# --------------------------------------------------------------------------- #
# Morphology (from the cell polygons only)
# --------------------------------------------------------------------------- #
def extract_cell_morphology(cells) -> pd.DataFrame:
    """Shape features from cell polygons (GeoDataFrame/GeoSeries); one row per cell."""
    geom = cells.geometry if hasattr(cells, "geometry") else cells
    feats = pd.DataFrame(index=geom.index)
    feats["area"] = geom.area.astype(float)
    feats["perimeter"] = geom.length.astype(float)
    feats["circularity"] = 4.0 * np.pi * feats["area"] / (feats["perimeter"].pow(2) + 1e-8)

    aspect, solidity = [], []
    for g in geom:
        if g is None or g.is_empty:
            aspect.append(np.nan)
            solidity.append(np.nan)
            continue
        minx, miny, maxx, maxy = g.bounds
        dx, dy = max(maxx - minx, 1e-8), max(maxy - miny, 1e-8)
        aspect.append(max(dx, dy) / min(dx, dy))
        hull_area = g.convex_hull.area
        solidity.append(g.area / hull_area if hull_area > 0 else np.nan)
    feats["aspect_ratio"] = aspect
    feats["solidity"] = solidity

    feats = feats.replace([np.inf, -np.inf], np.nan)
    feats = feats.fillna(feats.median(numeric_only=True)).fillna(0.0)
    return feats.astype(np.float32)


def make_morphology_groups(morphology, n_groups=20, n_pcs=8, random_state=42):
    """Scale descriptors -> PCA -> MiniBatchKMeans. Returns (embeddings, groups)."""
    Xs = StandardScaler().fit_transform(morphology.to_numpy(dtype=np.float32))
    n_pcs = max(1, min(n_pcs, Xs.shape[1], Xs.shape[0] - 1))
    embeddings = PCA(n_components=n_pcs, random_state=random_state).fit_transform(Xs).astype(np.float32)
    n_groups = max(2, min(n_groups, embeddings.shape[0]))
    groups = MiniBatchKMeans(
        n_clusters=n_groups, random_state=random_state, batch_size=4096, n_init=3
    ).fit_predict(embeddings)
    return embeddings, groups


# --------------------------------------------------------------------------- #
# Estimators
# --------------------------------------------------------------------------- #
def area_weighted_baseline(A, Y):
    """Weighted mean of overlapping pixel intensities per cell (no reconstruction)."""
    numerator = A.T @ Y
    denom = np.asarray(A.sum(axis=0)).ravel().astype(np.float32)
    return np.asarray(numerator) / (denom[:, None] + 1e-8)


def group_morphology_prior(A, Y, groups):
    """Per morphology group: pixel-mass-weighted mean of Y, assigned to every member cell."""
    prior = np.zeros((A.shape[1], Y.shape[1]), dtype=np.float32)
    for group in np.unique(groups):
        idx = np.flatnonzero(groups == group)
        mass = np.asarray(A[:, idx].sum(axis=1)).ravel()
        pix = mass > 0
        if not np.any(pix):
            continue
        w = mass[pix]
        prior[idx] = ((w[:, None] * Y[pix]).sum(axis=0) / (w.sum() + 1e-8)).astype(np.float32)
    return prior


def _estimate_lipschitz(A, lambda_prior, lambda_ridge, n_iter=20, safety=1.05):
    """Power iteration for the largest eigenvalue of A.T @ A, plus penalties."""
    v = np.ones(A.shape[1], dtype=np.float32)
    v /= np.linalg.norm(v) + 1e-12
    for _ in range(n_iter):
        v_new = np.asarray(A.T @ (A @ v)).ravel()
        norm = np.linalg.norm(v_new)
        if norm < 1e-12:
            return lambda_prior + lambda_ridge + 1e-6
        v = v_new / norm
    Av = A @ v
    return safety * float(np.dot(Av, Av)) + lambda_prior + lambda_ridge + 1e-6


def gaussian_nonnegative_deconvolution(
    A,
    Y,
    X_prior=None,
    X_init=None,
    lambda_prior=0.1,
    lambda_ridge=1e-5,
    max_iter=100,
    tol=1e-4,
    feature_batch_size=64,
):
    r"""
    Projected gradient descent, all features solved in column batches:

        min_{x >= 0}  0.5 ||A x - y||^2 + 0.5 lambda_prior ||x - x_prior||^2
                      + 0.5 lambda_ridge ||x||^2

    X_prior=None disables the prior term. X_init defaults to X_prior (or zeros).
    Features that converge are frozen while the rest of the batch continues.
    """
    Y = np.asarray(Y, dtype=np.float32)
    n_cells, n_features = A.shape[1], Y.shape[1]

    if X_prior is None:
        lambda_prior = 0.0
    else:
        X_prior = np.asarray(X_prior, dtype=np.float32)
        if X_prior.shape != (n_cells, n_features):
            raise ValueError("X_prior must have shape (n_cells, n_features).")
    if X_init is None:
        X_init = X_prior if X_prior is not None else np.zeros((n_cells, n_features), np.float32)

    At = A.T.tocsr()
    step = 1.0 / _estimate_lipschitz(A, lambda_prior, lambda_ridge)
    X = np.zeros((n_cells, n_features), dtype=np.float32)

    for start in range(0, n_features, feature_batch_size):
        sl = slice(start, min(start + feature_batch_size, n_features))
        Yb = Y[:, sl]
        Pb = X_prior[:, sl] if X_prior is not None else None
        x = np.maximum(np.asarray(X_init[:, sl], dtype=np.float32), 0.0)
        active = np.arange(x.shape[1])

        for _ in range(max_iter):
            xa = x[:, active]
            grad = At @ (A @ xa - Yb[:, active]) + lambda_ridge * xa
            if Pb is not None:
                grad = grad + lambda_prior * (xa - Pb[:, active])
            x_new = np.maximum(xa - step * grad, 0.0)
            change = np.linalg.norm(x_new - xa, axis=0) / (np.linalg.norm(xa, axis=0) + 1e-8)
            x[:, active] = x_new
            active = active[change >= tol]
            if active.size == 0:
                break
        X[:, sl] = x
    return X


def reconstruction_metrics(A, Y, X):
    """Reconstruction diagnostics for Y ~= A @ X."""
    Yhat = np.asarray(A @ X, dtype=np.float32)
    residual = Y - Yhat
    return {
        "reconstructed": Yhat,
        "residuals": residual,
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "relative_error": float(np.linalg.norm(residual) / (np.linalg.norm(Y) + 1e-8)),
        "feature_rmse": np.sqrt(np.mean(residual**2, axis=0)),
    }


# --------------------------------------------------------------------------- #
# Fit, package, register
# --------------------------------------------------------------------------- #
def _fit_and_package(
    prep: dict,
    model_type: str,
    *,
    n_groups=20,
    n_pcs=8,
    lambda_prior=0.1,
    lambda_ridge=1e-5,
    max_iter=100,
    tol=1e-4,
    feature_batch_size=64,
    normalise_per_pixel=True,
    compute_metrics=True,
    random_state=42,
    verbose=True,
) -> ad.AnnData:
    _check_model(model_type)
    A, Y = _build_A(prep["W_area"], normalise_per_pixel), prep["Y"]
    if verbose:
        print(f"[{model_type}] A {A.shape} (nnz={A.nnz}), Y {Y.shape}")

    groups = None
    if model_type in _MORPHOLOGY_MODELS:
        morphology = extract_cell_morphology(prep["cells"])
        _, groups = make_morphology_groups(morphology, n_groups, n_pcs, random_state)
        if verbose:
            print(f"[{model_type}] morphology groups: {len(np.unique(groups))}")

    kw = dict(lambda_ridge=lambda_ridge, max_iter=max_iter, tol=tol, feature_batch_size=feature_batch_size)
    if model_type == "area_weighted":
        X = area_weighted_baseline(A, Y)
    elif model_type == "morphology_prior":
        X = group_morphology_prior(A, Y, groups)
    elif model_type == "nnls":
        X = gaussian_nonnegative_deconvolution(A, Y, None, area_weighted_baseline(A, Y), **kw)
    elif model_type == "area_gaussian":
        X = gaussian_nonnegative_deconvolution(
            A, Y, area_weighted_baseline(A, Y), lambda_prior=lambda_prior, **kw
        )
    else:  # morphology_gaussian
        X = gaussian_nonnegative_deconvolution(
            A, Y, group_morphology_prior(A, Y, groups), lambda_prior=lambda_prior, **kw
        )

    out = ad.AnnData(X=np.asarray(X, dtype=np.float32), obs=prep["obs"].copy(), var=prep["var"].copy())
    if groups is not None:
        out.obs["morphology_group"] = pd.Categorical(groups)

    out.obsm["spatial"] = prep["xy"]  # cell centroids, for sq.pl.spatial_scatter / sc.pl.spatial
    out.obsm["pixel_weights"] = prep["W_norm"]  # per-cell normalised, as in pixels_to_cells
    out.obsm["pixel_overlap_area"] = prep["W_area"]  # raw overlap areas
    out.uns["pixel_ids"] = prep["pixel_ids"]
    out.uns["spatial_coordinate_system"] = prep["coordinate_system"] or "intrinsic"

    info = {
        "model_type": model_type,
        "pixel_layer": prep["pixel_layer"],
        "cell_layer": prep["cell_layer"],
        "source_table": prep["table_name"],
        "source_layer": prep["source_layer"],
        "coordinate_system": prep["coordinate_system"] or "intrinsic",
        "normalise_per_pixel": normalise_per_pixel,
    }
    if groups is not None:
        info.update(n_groups=n_groups, n_pcs=n_pcs)
    if "gaussian" in model_type:
        info["lambda_prior"] = lambda_prior
    if model_type in {"nnls", "area_gaussian", "morphology_gaussian"}:
        info["lambda_ridge"] = lambda_ridge
    if compute_metrics:
        m = reconstruction_metrics(A, Y, X)
        info.update(rmse=m["rmse"], relative_error=m["relative_error"])
        out.var["pixel_rmse"] = m["feature_rmse"]
        if verbose:
            print(f"[{model_type}] RMSE={m['rmse']:.6g}  relative error={m['relative_error']:.6g}")
    out.uns["deconvolution"] = info  # no None values, so it can be written to zarr
    return out


def _register(sdata, out, key_added, cell_layer, overwrite):
    from spatialdata.models import TableModel

    if key_added in sdata.tables and not overwrite:
        raise ValueError(f"Table '{key_added}' already exists. Use overwrite=True.")
    out = TableModel.parse(out, region=cell_layer, region_key="region", instance_key="cell_id")
    if key_added in sdata.tables:
        del sdata.tables[key_added]
    sdata.tables[key_added] = out
    if sdata.is_backed():
        sdata.write_element(key_added, overwrite=True)
    return out


# --------------------------------------------------------------------------- #
# Single entry point
# --------------------------------------------------------------------------- #
def deconvolve_maldi_to_cells(
    sdata,
    model_type: str = "morphology_gaussian",
    key_added: str = "cell_decon",
    *,
    pixel_layer: str = "pixels",
    cell_layer: str = "cell_segmentation",
    table_name: str = "maldi_adata",
    coordinate_system: Optional[str] = "aligned",
    layer: Optional[str] = None,
    min_coverage: float = 0.0,
    n_groups: int = 20,
    n_pcs: int = 8,
    lambda_prior: float = 0.1,
    lambda_ridge: float = 1e-5,
    max_iter: int = 100,
    tol: float = 1e-4,
    feature_batch_size: int = 64,
    normalise_per_pixel: bool = True,
    compute_metrics: bool = True,
    random_state: int = 42,
    overwrite: bool = False,
    inplace: bool = True,
    verbose: bool = True,
) -> ad.AnnData:
    """
    Impute MALDI m/z intensities to single cells. Independent of `pixels_to_cells`.

    Parameters
    ----------
    sdata : SpatialData
    model_type : one of MODEL_TYPES (see module docstring).
    key_added : name of the table the result is saved as (``sdata.tables[key_added]``).
    pixel_layer, cell_layer : shapes layers with MALDI pixel / single-cell polygons.
        ``cell_layer`` is also the only source of morphology.
    table_name : table with the pixel x m/z intensities annotating ``pixel_layer``.
    coordinate_system : overlaps and morphology are computed here (None = intrinsic).
    layer : table layer to use as intensities (default ``.X``; use a normalised layer).
    min_coverage : also remove cells with less than this fraction of area on pixels
        (cells with no overlap are always removed).
    overwrite : replace ``key_added`` if it exists.
    inplace : add the table to ``sdata``; the AnnData is always returned.

    Returns
    -------
    AnnData (cells x m/z), cells with pixel overlap only.
      .obsm["spatial"]           cell centroids in ``coordinate_system``
      .obsm["pixel_weights"]     per-cell normalised overlap (as pixels_to_cells)
      .obsm["pixel_overlap_area"] raw cell-pixel overlap areas
      .obs                       n_pixels, frac_cell_covered, max_pixel_weight,
                                 cell_area, cell_id, region, morphology_group*
      .var["pixel_rmse"]         per-feature reconstruction RMSE
      .uns["deconvolution"]      model_type, parameters, rmse, relative_error
    """
    _check_model(model_type)
    if inplace and key_added in sdata.tables and not overwrite:
        raise ValueError(f"Table '{key_added}' already exists. Use overwrite=True.")

    prep = compute_pixel_overlap(
        sdata,
        pixel_layer=pixel_layer,
        cell_layer=cell_layer,
        table_name=table_name,
        coordinate_system=coordinate_system,
        layer=layer,
        min_coverage=min_coverage,
        verbose=verbose,
    )
    out = _fit_and_package(
        prep,
        model_type,
        n_groups=n_groups,
        n_pcs=n_pcs,
        lambda_prior=lambda_prior,
        lambda_ridge=lambda_ridge,
        max_iter=max_iter,
        tol=tol,
        feature_batch_size=feature_batch_size,
        normalise_per_pixel=normalise_per_pixel,
        compute_metrics=compute_metrics,
        random_state=random_state,
        verbose=verbose,
    )
    if inplace:
        out = _register(sdata, out, key_added, cell_layer, overwrite)
    return out


def compare_models(
    sdata,
    model_types: Iterable[str] = MODEL_TYPES,
    key_prefix: Optional[str] = None,
    overwrite: bool = False,
    **kwargs,
) -> tuple[Dict[str, ad.AnnData], pd.DataFrame]:
    """
    Run several model_types on one shared overlap computation.

    Returns ({model_type: AnnData}, metrics DataFrame). Nothing is added to sdata
    unless key_prefix is given, in which case tables are saved as
    ``f"{key_prefix}_{model_type}"``. Accepts the same layer / coordinate_system /
    fitting keyword arguments as deconvolve_maldi_to_cells.
    """
    model_types = list(model_types)
    for m in model_types:
        _check_model(m)
    overlap_kw = {k: kwargs.pop(k) for k in list(kwargs) if k in _OVERLAP_ARGS}
    prep = compute_pixel_overlap(sdata, verbose=kwargs.get("verbose", True), **overlap_kw)

    results, rows = {}, []
    for name in model_types:
        out = _fit_and_package(prep, name, **kwargs)
        if key_prefix:
            out = _register(sdata, out, f"{key_prefix}_{name}", prep["cell_layer"], overwrite)
        results[name] = out
        d = out.uns["deconvolution"]
        rows.append({"model_type": name, "rmse": d.get("rmse"), "relative_error": d.get("relative_error")})
    return results, pd.DataFrame(rows).set_index("model_type")




def run_decon_PCA(sdata, table_name: str = "cell_decon", n_pcs: int = 30, log_norm: bool = True, scale: bool = True, random_state: int = 42, verbose: bool = True):
    """
    Get deconvolved cell-level MALDI intensities and perform PCA + K-means clustering.

    Parameters
    ----------
    sdata : SpatialData
        The SpatialData object containing the deconvolved cell-level MALDI intensities.
    table : str, optional
        The name of the table containing the deconvolved cell-level MALDI intensities.
        Default is "cell_decon".

    Returns
    -------
    cell_adata : AnnData
        The AnnData object containing the deconvolved cell-level MALDI intensities.
    X_pca : np.ndarray
        The PCA-transformed data.
    clusters : np.ndarray
        The cluster labels assigned to each cell.
    """

    cell_adata = sdata.tables[table_name]  

    X = np.asarray(sdata.tables[table_name].X, dtype=np.float32)

    assert X.shape == (cell_adata.n_obs, len(cell_adata.var))

    # Replace invalid values and enforce non-negativity
    X = np.nan_to_num(
        X,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    X = np.maximum(X, 0)

    if verbose:
        print("Cell-level Decon matrix:", X.shape)


    # Log transformation reduces the influence of very high intensities
    X_update = X
    if log_norm:
        X_update = np.log1p(X_update)
    

    # Standardise each metabolite across cells
    if scale:
        scaler = StandardScaler()
        X_update = scaler.fit_transform(X_update)


    pca = PCA(
        n_components=n_pcs,
        random_state=random_state,
    )

    X_pca = pca.fit_transform(X_update)

    if verbose:
        print("PCA shape:", X_pca.shape)

        print(
            "Variance explained by 30 PCs:",
            pca.explained_variance_ratio_.sum()
        )

    sdata.tables[table_name].obsm["X_pca"] = X_pca

    return sdata




def _cs(info):
    cs = info.get("coordinate_system")
    return None if cs in (None, "intrinsic") else cs


def _as_categorical(adata, key):
    if key not in adata.obs:
        cats = [c for c in adata.obs.columns if not pd.api.types.is_float_dtype(adata.obs[c])]
        raise KeyError(f"'{key}' is not in .obs. Candidate columns: {cats}")
    s = adata.obs[key]
    if not isinstance(s.dtype, pd.CategoricalDtype):
        if pd.api.types.is_numeric_dtype(s) and s.nunique() > 30:
            raise ValueError(f"'{key}' looks continuous ({s.nunique()} unique values); a categorical column is needed.")
        s = s.astype("category")
    return s.cat.remove_unused_categories()


def _category_colors(adata, key, cats):
    stored = adata.uns.get(f"{key}_colors")
    if stored is not None and len(stored) == len(cats):
        return dict(zip(cats, [mcolors.to_hex(c) for c in stored]))
    n = len(cats)
    if n <= 10:
        pal = plt.get_cmap("tab10")(np.arange(n))
    elif n <= 20:
        pal = plt.get_cmap("tab20")(np.arange(n))
    else:
        pal = plt.get_cmap("nipy_spectral")(np.linspace(0.05, 0.95, n))
    return dict(zip(cats, [mcolors.to_hex(c) for c in pal]))


def _cell_polys(sdata, adata):
    """Cell polygons in the order of adata.obs (matched by cell_id)."""
    info = adata.uns["deconvolution"]
    gdf = _get_polygons(sdata, info["cell_layer"], _cs(info))
    gdf = gdf.set_axis(gdf.index.astype(str))
    return gdf.geometry.loc[adata.obs["cell_id"].astype(str).to_numpy()]


def _pixel_polys(sdata, adata):
    """Pixel polygons in the order of adata.uns['pixel_ids'] (= rows of the pixel matrix)."""
    info = adata.uns["deconvolution"]
    gdf = _get_polygons(sdata, info["pixel_layer"], _cs(info))
    gdf = gdf.set_axis(gdf.index.astype(str))
    return gdf.geometry.loc[np.asarray(adata.uns["pixel_ids"]).astype(str)]


def _get_morphology(sdata, adata):
    if "morphology" in adata.obsm and "morphology_features" in adata.uns:
        return pd.DataFrame(
            np.asarray(adata.obsm["morphology"]),
            index=adata.obs_names,
            columns=list(adata.uns["morphology_features"]),
        )
    # Tables made before morphology was stored: recompute from the polygons.
    morph = extract_cell_morphology(_cell_polys(sdata, adata))
    morph.index = adata.obs_names
    return morph


def _parts(geom):
    for p in (geom.geoms if hasattr(geom, "geoms") else [geom]):
        if hasattr(p, "exterior") and not p.is_empty:
            yield p


def _verts(geoms):
    verts, owner = [], []
    for k, g in enumerate(geoms):
        for p in _parts(g):
            verts.append(np.asarray(p.exterior.coords))
            owner.append(k)
    return verts, np.asarray(owner, dtype=int)


def _in_roi(geoms, roi):
    if roi is None:
        return np.ones(len(geoms), dtype=bool)
    xmin, ymin, xmax, ymax = roi
    b = geoms.bounds.to_numpy()
    return (b[:, 2] >= xmin) & (b[:, 0] <= xmax) & (b[:, 3] >= ymin) & (b[:, 1] <= ymax)


def _style_spatial(ax, roi, geoms_list, invert_y=True):
    ax.set_aspect("equal")
    if roi is not None:
        ax.set_xlim(roi[0], roi[2])
        lo, hi = roi[1], roi[3]
    else:
        b = pd.concat([g.bounds for g in geoms_list])
        ax.set_xlim(b["minx"].min(), b["maxx"].max())
        lo, hi = b["miny"].min(), b["maxy"].max()
    ax.set_ylim((hi, lo) if invert_y else (lo, hi))
    ax.set_xticks([])
    ax.set_yticks([])
    for sp_ in ax.spines.values():
        sp_.set_linewidth(0.4)



def compare_morphology_clusters(
    sdata,
    table: str = "cell_decon",
    group_key: str = "morphology_group",
    n_examples: int = 8,
    example_mode: str = "representative",
    order: str = "hierarchical",
    normalise_size: bool = False,
    features: Optional[Sequence[str]] = None,
    annot: bool = True,
    cmap: str = "RdBu_r",
    vmax: Optional[float] = None,
    random_state: int = 0,
    save: Optional[str] = None,
):
    """
    Compare morphology groups: dendrogram, feature heatmap, cluster sizes and example cells.

    Heatmap values are z-scores: (cluster mean - mean over all cells) / std over all cells,
    so 0 = typical cell and +1 = one SD above average for that feature.

    Parameters
    ----------
    group_key : categorical .obs column. Default "morphology_group", but any grouping works,
        e.g. a leiden clustering of the deconvolved .X, to ask whether expression clusters
        differ in shape.
    n_examples : example cells drawn per cluster.
    example_mode : "representative" (closest to the cluster centre in feature space) or "random".
    order : "hierarchical" (ward dendrogram on cluster means), "size" or "label".
    normalise_size : False draws examples at a common scale (size differences are visible);
        True rescales each cell to fill its slot (shape only).

    Returns
    -------
    (fig, stats) where stats holds "mean", "zscore" and "n_cells" per cluster.
    """
    adata = sdata.tables[table]
    g = _as_categorical(adata, group_key)
    cats = list(g.cat.categories)
    K = len(cats)
    if K < 2:
        raise ValueError("Need at least two clusters to compare.")
    morph = _get_morphology(sdata, adata)
    if features is not None:
        morph = morph[list(features)]
    feats = list(morph.columns)
    F = len(feats)

    mu, sd = morph.mean(), morph.std(ddof=0).replace(0, 1.0)
    Zc = (morph - mu) / sd
    labels = g.to_numpy()
    mean_df = morph.groupby(labels, observed=True).mean().reindex(cats)
    z_df = Zc.groupby(labels, observed=True).mean().reindex(cats)
    n_cells = g.value_counts().reindex(cats)

    use_dendro = False
    if order == "hierarchical" and K >= 3:
        Zlink = linkage(z_df.to_numpy(), method="ward")
        leaves = dendrogram(Zlink, no_plot=True)["leaves"]
        use_dendro = True
    elif order == "size":
        leaves = list(np.argsort(-n_cells.to_numpy()))
    else:
        leaves = list(range(K))
    ordered = [cats[i] for i in leaves]
    colors = _category_colors(adata, group_key, cats)

    # ---- example cells -------------------------------------------------------
    polys = _cell_polys(sdata, adata)
    rng = np.random.default_rng(random_state)
    chosen = {}
    for cat in ordered:
        pos = np.flatnonzero(labels == cat)
        n = min(n_examples, len(pos))
        if example_mode == "random":
            chosen[cat] = rng.choice(pos, n, replace=False)
        else:
            d = np.linalg.norm(Zc.to_numpy()[pos] - z_df.loc[cat].to_numpy(), axis=1)
            chosen[cat] = pos[np.argsort(d)[:n]]
    ex_geoms = {c: [polys.iloc[i] for i in idx] for c, idx in chosen.items()}
    max_extent = max(
        max(g_.bounds[2] - g_.bounds[0], g_.bounds[3] - g_.bounds[1])
        for gl in ex_geoms.values() for g_ in gl
    )

    # ---- layout in inches ----------------------------------------------------
    s = 0.8
    dw = 0.8 if use_dendro else 0.0
    left, right, top, bottom = 0.2, 0.2, 1.4, 0.95
    hw = max(F * 0.72, 2.2)
    x_d = left
    x_h = x_d + dw + 0.55
    x_c = x_h + hw + 0.08
    x_b = x_c + 0.1 + 0.15
    x_e = x_b + 1.0 + 0.35
    Wd = x_e + n_examples * s + right
    Hd = bottom + K * s + top
    fig = plt.figure(figsize=(Wd, Hd))

    def add(x, y, w, h):
        return fig.add_axes([x / Wd, y / Hd, w / Wd, h / Hd])

    H = K * s
    # heatmap
    zmat = z_df.loc[ordered].to_numpy()
    vm = vmax or max(0.5, float(np.nanmax(np.abs(zmat))))
    ax_h = add(x_h, bottom, hw, H)
    im = ax_h.imshow(zmat, cmap=cmap, vmin=-vm, vmax=vm, aspect="auto")
    ax_h.set_xticks(range(F))
    ax_h.set_xticklabels([f.replace("_", " ") for f in feats], rotation=40, ha="left", fontsize=8)
    ax_h.set_yticks(range(K))
    ax_h.set_yticklabels([str(c) for c in ordered], fontsize=8)
    ax_h.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False, length=0)
    for sp_ in ax_h.spines.values():
        sp_.set_visible(False)
    if annot:
        for i in range(K):
            for j in range(F):
                v = zmat[i, j]
                ax_h.text(j, i, f"{v:+.1f}", ha="center", va="center", fontsize=7,
                          color="white" if abs(v) > 0.6 * vm else "black")
    cax = add(x_h, 0.4, hw * 0.8, 0.12)
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.set_label("z-score vs all cells", fontsize=7)
    cb.ax.tick_params(labelsize=7)

    # dendrogram
    if use_dendro:
        ax_d = add(x_d, bottom, dw, H)
        dendrogram(Zlink, orientation="left", ax=ax_d, no_labels=True, color_threshold=0,
                   above_threshold_color="0.35")
        ax_d.set_ylim(10 * K, 0)
        ax_d.axis("off")

    # colour strip + sizes
    ax_c = add(x_c, bottom, 0.1, H)
    ax_c.imshow(np.array([[mcolors.to_rgb(colors[c])] for c in ordered]), aspect="auto")
    ax_c.axis("off")
    ax_b = add(x_b, bottom, 1.0, H)
    counts = n_cells.loc[ordered].to_numpy()
    ax_b.barh(range(K), counts, color=[colors[c] for c in ordered], height=0.7)
    for i, v in enumerate(counts):
        ax_b.text(v, i, f" {v:,}", va="center", fontsize=7)
    ax_b.set_ylim(K - 0.5, -0.5)
    ax_b.set_xlim(0, counts.max() * 1.45)
    ax_b.set_yticks([])
    ax_b.set_xticks([])
    for side in ("top", "right", "left", "bottom"):
        ax_b.spines[side].set_visible(False)
    ax_b.set_title("cells", fontsize=8, pad=4)

    # example cells
    ax_e = add(x_e, bottom, n_examples * s, H)
    ax_e.set_xlim(-0.5, n_examples - 0.5)
    ax_e.set_ylim(K - 0.5, -0.5)
    for i, cat in enumerate(ordered):
        if i:
            ax_e.axhline(i - 0.5, color="0.9", lw=0.5)
        for j, geom in enumerate(ex_geoms[cat]):
            minx, miny, maxx, maxy = geom.bounds
            ext = max(maxx - minx, maxy - miny)
            scale = 0.82 / (ext if normalise_size else max_extent)
            cx, cy = (minx + maxx) / 2, (miny + maxy) / 2
            for p in _parts(geom):
                xy = np.asarray(p.exterior.coords)
                xy = np.c_[(xy[:, 0] - cx) * scale + j, (xy[:, 1] - cy) * scale + i]
                ax_e.fill(xy[:, 0], xy[:, 1], fc=colors[cat], ec="0.2", lw=0.5, alpha=0.85)
    ax_e.set_xticks([])
    ax_e.set_yticks([])
    for sp_ in ax_e.spines.values():
        sp_.set_visible(False)
    scale_note = "each scaled to fit" if normalise_size else "common scale"
    ax_e.set_title(f"example cells ({example_mode}; {scale_note})", fontsize=8, pad=4, loc="left")

    fig.suptitle(f"Morphology comparison of '{group_key}' ({K} clusters, {len(adata):,} cells)",
                 x=0.01, ha="left", y=0.995, fontsize=10, fontweight="bold")
    if save:
        fig.savefig(save, dpi=300, bbox_inches="tight")
    stats = {"mean": mean_df.loc[ordered], "zscore": z_df.loc[ordered], "n_cells": n_cells.loc[ordered]}
    return fig, stats


def plot_pixel_pies(
    sdata,
    color: str,
    table: str = "cell_decon",
    weight: str = "area",
    roi: Optional[Sequence[float]] = None,
    radius_scale: float = 0.9,
    colors: Optional[dict] = None,
    ax=None,
    invert_y: bool = True,
    legend: bool = True,
    title: Optional[str] = None,
    max_pixels: int = 60000,
):
    """
    One pie chart per MALDI pixel showing the make-up of the cells inside it.

    Parameters
    ----------
    color : categorical .obs column of the cell table (e.g. a leiden clustering of .X,
        or "morphology_group").
    weight : "area" - each cell counts by its overlap area with the pixel (default);
             "count" - each overlapping cell counts once.
    roi : (xmin, ymin, xmax, ymax) in the table's coordinate system; strongly recommended
        for large slides, where pies are unreadable and slow to draw at full scale.
    radius_scale : pie radius as a fraction of half the pixel width.

    Returns
    -------
    (ax, props): props is a DataFrame (pixels x categories) of the plotted proportions.
    """
    if weight not in ("area", "count"):
        raise ValueError("weight must be 'area' or 'count'.")
    adata = sdata.tables[table]
    cat = _as_categorical(adata, color)
    cats = list(cat.cat.categories)
    K = len(cats)
    cmap = colors or _category_colors(adata, color, cats)

    W = sp.csr_matrix(adata.obsm["pixel_overlap_area"])  # cells x pixels
    if weight == "count":
        W = (W > 0).astype(np.float32)
    codes = cat.cat.codes.to_numpy()
    ok = codes >= 0
    Z = sp.csr_matrix((np.ones(ok.sum(), np.float32), (np.flatnonzero(ok), codes[ok])), shape=(W.shape[0], K))
    P = np.asarray((W.T @ Z).todense())  # pixels x K
    tot = P.sum(1)
    props = P / np.where(tot > 0, tot, 1.0)[:, None]

    pix = _pixel_polys(sdata, adata)
    cen = pix.centroid
    xy = np.column_stack([cen.x.to_numpy(), cen.y.to_numpy()])
    b = pix.bounds.to_numpy()
    radius = 0.5 * np.median(np.minimum(b[:, 2] - b[:, 0], b[:, 3] - b[:, 1])) * radius_scale

    show = (tot > 0) & _in_roi(pix, roi)
    if show.sum() > max_pixels:
        warnings.warn(f"Drawing {int(show.sum()):,} pies will be slow and hard to read; consider roi=(xmin, ymin, xmax, ymax).")

    wedges, fcs = [], []
    for i in np.flatnonzero(show):
        cum = 0.0
        for k in np.flatnonzero(props[i] > 0):
            wedges.append(Wedge(xy[i], radius, 360 * cum, 360 * (cum + props[i, k])))
            fcs.append(cmap[cats[k]])
            cum += props[i, k]

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 7))
    ax.add_collection(PatchCollection(wedges, facecolors=fcs, edgecolors="none", linewidths=0))
    _style_spatial(ax, roi, [pix[show]] if show.any() else [pix], invert_y)
    ax.set_title(title or f"{color}: composition of each MALDI pixel (weight={weight})", fontsize=9)
    if legend:
        ax.legend(handles=[Patch(fc=cmap[c], label=str(c)) for c in cats], loc="upper left",
                  bbox_to_anchor=(1.01, 1), frameon=False, fontsize=8, title=color,
                  ncol=1 if K <= 18 else 2)
    out = pd.DataFrame(props, index=np.asarray(adata.uns["pixel_ids"]).astype(str), columns=[str(c) for c in cats])
    return ax, out.loc[show]



def _load_pixel_matrix(sdata, adata):
    info = adata.uns["deconvolution"]
    tab = sdata.tables[info["source_table"]]
    attrs = tab.uns["spatialdata_attrs"]
    sub = tab[tab.obs[attrs["region_key"]] == info["pixel_layer"]]
    ids = sub.obs[attrs["instance_key"]].astype(str).to_numpy()
    src = info["source_layer"]
    Y = _dense_float32(sub.X if src == "X" else sub.layers[src])
    pos = pd.Series(np.arange(len(ids)), index=ids)
    return Y[pos.loc[np.asarray(adata.uns["pixel_ids"]).astype(str)].to_numpy()]


def evaluate_deconvolution(
    sdata,
    table: str = "cell_decon",
    features: Optional[Sequence[str]] = None,
    n_features: int = 3,
    roi: Optional[Sequence[float]] = None,
    max_points: int = 200_000,
    cmap: str = "magma",
    random_state: int = 0,
    save: Optional[str] = None,
):
    """
    Compare the original pixel data with the deconvolved single-cell data.

    The cell values are aggregated back to pixels (Y_hat = A @ X) and compared with the
    observed pixel intensities Y.

      Top row    (a) observed vs reconstructed intensities, all pixels x m/z
                 (b) per-feature correlation between observed and reconstructed pixel maps
                 (c) per-feature relative error vs mean intensity
                 (d) per-pixel relative error vs number of cells in the pixel
      Other rows for selected m/z: observed pixels | reconstructed pixels | imputed single
                 cells | residual (observed - reconstructed).

    Caution: this is an in-sample check. A good reconstruction shows the cell values are
    consistent with the pixel data, not that individual cell values are correct; many
    different cell-level solutions can give the same pixel sums.

    Parameters
    ----------
    features : m/z names (var_names) to map; default = top `n_features` by variance.
    roi : (xmin, ymin, xmax, ymax) zoom for the maps, recommended so single cells are visible.

    Returns
    -------
    (fig, metrics) where metrics holds global numbers and per-feature / per-pixel tables.
    """
    adata = sdata.tables[table]
    info = adata.uns["deconvolution"]
    Y = _load_pixel_matrix(sdata, adata)
    X = np.asarray(adata.X, dtype=np.float32)
    A = _build_A(adata.obsm["pixel_overlap_area"], info.get("normalise_per_pixel", True))
    Yhat = np.asarray(A @ X, dtype=np.float32)
    P, M = Y.shape
    rng = np.random.default_rng(random_state)

    res = Y - Yhat
    rel_err = float(np.linalg.norm(res) / (np.linalg.norm(Y) + 1e-12))
    Yc, Hc = Y - Y.mean(0), Yhat - Yhat.mean(0)
    num = (Yc * Hc).sum(0, dtype=np.float64)
    den = np.sqrt((Yc**2).sum(0, dtype=np.float64) * (Hc**2).sum(0, dtype=np.float64))
    r_feat = np.divide(num, den, out=np.full(M, np.nan), where=den > 0)
    feat_rel = np.linalg.norm(res, axis=0) / (np.linalg.norm(Y, axis=0) + 1e-12)
    pix_rel = np.linalg.norm(res, axis=1) / (np.linalg.norm(Y, axis=1) + 1e-12)
    n_in_pix = np.diff(A.indptr)

    n_pts = min(max_points, P * M)
    flat = rng.choice(P * M, n_pts, replace=False)
    a_, b_ = Y[flat // M, flat % M], Yhat[flat // M, flat % M]
    r_global = float(np.corrcoef(a_, b_)[0, 1])

    # feature selection
    if features is None:
        sel = list(np.argsort(-Y.var(0))[:n_features])
    else:
        sel = [adata.var_names.get_loc(f) for f in features]
    nf = len(sel)

    pix, cel = _pixel_polys(sdata, adata), _cell_polys(sdata, adata)
    pmask, cmask = _in_roi(pix, roi), _in_roi(cel, roi)
    pv, pown = _verts(pix[pmask])
    cv, cown = _verts(cel[cmask])

    fig = plt.figure(figsize=(16, 3.6 * (1 + nf)), layout="constrained")
    gs = fig.add_gridspec(1 + nf, 4)

    # (a)
    ax = fig.add_subplot(gs[0, 0])
    hb = ax.hexbin(a_, b_, gridsize=60, bins="log", mincnt=1, cmap="viridis",
                   extent=(0, np.percentile(a_, 99.9), 0, np.percentile(a_, 99.9)))
    lim = np.percentile(a_, 99.9)
    ax.plot([0, lim], [0, lim], "r--", lw=0.8)
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.set_xlabel("observed pixel intensity"); ax.set_ylabel("reconstructed (A·X̂)")
    ax.set_title(f"(a) all pixels × m/z\nr = {r_global:.3f}, relative error = {rel_err:.3f}", fontsize=9)
    fig.colorbar(hb, ax=ax, label="count (log)", shrink=0.8)

    # (b)
    ax = fig.add_subplot(gs[0, 1])
    ax.hist(r_feat[np.isfinite(r_feat)], bins=40, color="#0072B2", edgecolor="white", lw=0.3)
    ax.axvline(np.nanmedian(r_feat), color="r", ls="--", lw=1)
    ax.set_xlabel("per-feature r (observed vs reconstructed map)"); ax.set_ylabel("m/z features")
    ax.set_title(f"(b) spatial pattern recovery\nmedian r = {np.nanmedian(r_feat):.3f}", fontsize=9)

    # (c)
    ax = fig.add_subplot(gs[0, 2])
    mean_int = Y.mean(0)
    ax.scatter(mean_int, feat_rel, s=6, alpha=0.5, color="#D55E00", lw=0)
    if (mean_int > 0).all():
        ax.set_xscale("log")
    ax.set_xlabel("mean observed intensity"); ax.set_ylabel("relative error ‖y−ŷ‖/‖y‖")
    ax.set_title(f"(c) error by feature\nmedian = {np.median(feat_rel):.3f}", fontsize=9)

    # (d)
    ax = fig.add_subplot(gs[0, 3])
    bins = [1, 2, 3, 4, 5]
    groups, labs = [], []
    for k in bins:
        m = (n_in_pix == k) if k < 5 else (n_in_pix >= 5)
        if m.sum():
            groups.append(pix_rel[m]); labs.append(f"{k}{'+' if k == 5 else ''}\n(n={int(m.sum()):,})")
    ax.boxplot(groups, showfliers=False, patch_artist=True,
               boxprops=dict(facecolor="#CC79A7", alpha=0.7), medianprops=dict(color="k"))
    ax.set_xticks(range(1, len(labs) + 1))
    ax.set_xticklabels(labs, fontsize=8)
    ax.set_xlabel("cells overlapping the pixel"); ax.set_ylabel("pixel relative error")
    ax.set_title("(d) error vs cells per pixel", fontsize=9)

    # spatial rows
    titles = ["observed (pixels)", "reconstructed A·X̂ (pixels)", "imputed single cells X̂", "residual (obs − recon)"]
    for r, j in enumerate(sel, start=1):
        name = str(adata.var_names[j])
        lo, hi = np.percentile(Y[:, j], [1, 99])
        norm = mcolors.Normalize(lo, hi if hi > lo else lo + 1e-9)
        rlim = max(np.percentile(np.abs(res[:, j]), 99), 1e-9)
        axes = [fig.add_subplot(gs[r, c]) for c in range(4)]
        layers = [
            (pv, Y[:, j][pmask][pown], cmap, norm),
            (pv, Yhat[:, j][pmask][pown], cmap, norm),
            (cv, X[:, j][cmask][cown], cmap, norm),
            (pv, res[:, j][pmask][pown], "RdBu_r", mcolors.Normalize(-rlim, rlim)),
        ]
        mappables = []
        for ax, (verts, vals, cm, nm), t in zip(axes, layers, titles):
            pc = PolyCollection(verts, array=vals, cmap=cm, norm=nm, edgecolors="none", linewidths=0)
            ax.add_collection(pc)
            _style_spatial(ax, roi, [pix, cel])
            ax.set_title(t, fontsize=9)
            mappables.append(pc)
        axes[0].set_ylabel(f"m/z {name}", fontsize=10, fontweight="bold")
        fig.colorbar(mappables[0], ax=axes[:3], shrink=0.8, pad=0.01, label="intensity")
        fig.colorbar(mappables[3], ax=axes[3], shrink=0.8, pad=0.01, label="residual")

    fig.suptitle(
        f"Pixel data vs deconvolved single-cell data  |  model: {info['model_type']}  |  "
        f"{adata.n_obs:,} cells, {P:,} pixels, {M} m/z",
        fontsize=11, fontweight="bold",
    )
    if save:
        fig.savefig(save, dpi=300, bbox_inches="tight")
    metrics = {
        "relative_error": rel_err,
        "pearson_r_global": r_global,
        "median_feature_r": float(np.nanmedian(r_feat)),
        "median_feature_relative_error": float(np.median(feat_rel)),
        "per_feature": pd.DataFrame(
            {"pearson_r": r_feat, "relative_error": feat_rel, "mean_intensity": mean_int},
            index=adata.var_names,
        ),
        "per_pixel": pd.DataFrame(
            {"relative_error": pix_rel, "n_cells": n_in_pix},
            index=np.asarray(adata.uns["pixel_ids"]).astype(str),
        ),
    }
    return fig, metrics


def plot_model_comparison(metrics: pd.DataFrame, ax=None):
    """
    Bar chart of reconstruction error per model_type (output of compare_models()).

    Lower is better, but reconstruction error alone favours the least-regularised models
    (nnls fits the pixels best and is the least constrained); it does not rank accuracy.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 3))
    m = metrics.sort_values("relative_error")
    ax.barh(m.index, m["relative_error"], color="#0072B2")
    for i, v in enumerate(m["relative_error"]):
        ax.text(v, i, f" {v:.3f}", va="center", fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("relative reconstruction error  ‖Y − AX‖/‖Y‖")
    ax.set_xlim(0, m["relative_error"].max() * 1.2)
    return ax
