"""
batch_correction.py
===================

Batch correction for MALDI SpatialData objects.

Supports two methods:

* Harmony
    Operates on a PCA embedding.
    Corrected embeddings are stored in:

        adata.obsm["X_pca_harmony"]

    The raw intensity matrix (adata.X) is NOT modified.

* ComBat
    Operates directly on adata.X.
    Corrected intensities overwrite:

        adata.X


---------------------------------------------------------------------------
IMPORTANT
---------------------------------------------------------------------------

This function assumes the data has already been normalised
(e.g. TIC or RMS normalisation).

Applying batch correction to raw counts will produce misleading results.


---------------------------------------------------------------------------
USAGE
---------------------------------------------------------------------------

# Option 1 — merge inside batch_correction
merged = batch_correction(
    sdatas=[sdata1, sdata2],
    batch_names=["A", "B"],
    method="harmony",
)

# Option 2 — use an already merged object
merged = batch_correction(
    pre_merged=merged_sdata,
    method="combat",
)

"""

from __future__ import annotations

import os
import warnings

from typing import Literal, Optional

import numpy as np
import pandas as pd

from spatialdata import SpatialData

import goatpy as gp


# -----------------------------------------------------------------------------
# Logging helper
# -----------------------------------------------------------------------------

def _log(msg: str) -> None:

    try:
        import psutil

        rss = psutil.Process(os.getpid()).memory_info().rss / 1e9

        print(f"[{rss:.2f}GB] {msg}")

    except ImportError:

        print(msg)


# -----------------------------------------------------------------------------
# Harmony
# -----------------------------------------------------------------------------

def _run_harmony(
    merged: SpatialData,
    table_name: str,
    pcs: int,
    batch_col: str,
    random_state: int,
) -> SpatialData:

    try:
        import harmonypy
    except ImportError as e:
        raise ImportError(
            "harmonypy is required for Harmony batch correction.\n"
            "Install with:\n"
            "pip install harmonypy"
        ) from e

    import anndata as ad
    import scanpy as sc

    from scipy.sparse import issparse
    from sklearn.preprocessing import StandardScaler

    adata = merged.tables[table_name]

    _log(
        f"Harmony input: "
        f"{adata.n_obs:,} pixels × {adata.n_vars:,} features"
    )

    # -------------------------------------------------------------------------
    # Convert matrix
    # -------------------------------------------------------------------------

    X_raw = (
        adata.X.toarray()
        if issparse(adata.X)
        else np.asarray(adata.X, dtype=np.float64)
    )

    X_scaled = np.zeros_like(X_raw, dtype=np.float64)

    # -------------------------------------------------------------------------
    # Scale each batch independently
    # -------------------------------------------------------------------------

    unique_batches = (
        adata.obs[batch_col]
        .astype(str)
        .unique()
    )

    _log(f"Detected {len(unique_batches)} batches")

    for batch in unique_batches:

        mask = adata.obs[batch_col].astype(str) == str(batch)

        idx = np.where(mask)[0]

        if idx.size == 0:
            continue

        scaler = StandardScaler()

        X_scaled[idx] = scaler.fit_transform(X_raw[idx])

        _log(
            f"Scaled batch '{batch}' "
            f"({idx.size:,} pixels)"
        )

    # -------------------------------------------------------------------------
    # PCA object
    # -------------------------------------------------------------------------

    pca_adata = ad.AnnData(
        X=X_scaled.astype(np.float32),
        obs=adata.obs.copy(),
    )

    _log(f"Running PCA ({pcs} PCs)")

    sc.pp.pca(
        pca_adata,
        n_comps=pcs,
        random_state=random_state,
    )

    # -------------------------------------------------------------------------
    # Harmony
    # -------------------------------------------------------------------------

    _log("Running Harmony")

    harmony_out = harmonypy.run_harmony(
        pca_adata.obsm["X_pca"].astype(np.float64),
        pca_adata.obs,
        batch_col,
        max_iter_harmony=20,
    )

    corrected = harmony_out.Z_corr

    if corrected.shape[0] != adata.n_obs:
        corrected = corrected.T

    # -------------------------------------------------------------------------
    # Store embeddings
    # -------------------------------------------------------------------------

    adata.obsm["X_pca"] = pca_adata.obsm["X_pca"]

    adata.obsm["X_pca_harmony"] = corrected.astype(np.float32)

    _log(
        "Harmony complete "
        f"(embedding shape={corrected.shape})"
    )

    return merged


# -----------------------------------------------------------------------------
# ComBat
# -----------------------------------------------------------------------------

def _run_combat(
    merged: SpatialData,
    table_name: str,
    batch_col: str,
    covariates: Optional[list[str]],
) -> SpatialData:

    try:
        import scanpy as sc
    except ImportError as e:
        raise ImportError(
            "scanpy is required for ComBat batch correction.\n"
            "Install with:\n"
            "pip install scanpy"
        ) from e

    adata = merged.tables[table_name]

    X = np.asarray(adata.X, dtype=np.float64)

    # -------------------------------------------------------------------------
    # Replace invalid values
    # -------------------------------------------------------------------------

    if not np.all(np.isfinite(X)):

        n_bad = int(np.sum(~np.isfinite(X)))

        warnings.warn(
            f"Found {n_bad:,} non-finite values in adata.X.\n"
            "Replacing with 0 before ComBat.",
            stacklevel=2,
        )

        X = np.nan_to_num(
            X,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        adata.X = X.astype(np.float32)

    # -------------------------------------------------------------------------
    # Validate covariates
    # -------------------------------------------------------------------------

    covariates_arg = None

    if covariates is not None:

        missing = [
            c for c in covariates
            if c not in adata.obs.columns
        ]

        if missing:
            raise KeyError(
                "Missing covariates in adata.obs:\n"
                f"{missing}"
            )

        covariates_arg = covariates

        _log(
            f"Preserving covariates: "
            f"{covariates_arg}"
        )

    # -------------------------------------------------------------------------
    # Run ComBat
    # -------------------------------------------------------------------------

    _log(
        f"Running ComBat "
        f"(matrix shape={X.shape})"
    )

    sc.pp.combat(
        adata,
        key=batch_col,
        covariates=covariates_arg,
        inplace=True,
    )

    # -------------------------------------------------------------------------
    # Remove negatives
    # -------------------------------------------------------------------------

    X_corrected = np.asarray(
        adata.X,
        dtype=np.float32,
    )

    n_neg = int((X_corrected < 0).sum())

    if n_neg > 0:

        _log(
            f"Clipping {n_neg:,} negative values to 0"
        )

        X_corrected = np.clip(
            X_corrected,
            0.0,
            None,
        )

        adata.X = X_corrected

    _log("ComBat complete")

    return merged


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

def batch_correction(
    sdatas: Optional[list[SpatialData]] = None,
    batch_names: Optional[list[str]] = None,
    pre_merged: Optional[SpatialData] = None,
    method: Literal["harmony", "combat"] = "harmony",
    pcs: int = 30,
    covariates: Optional[list[str]] = None,
    table_name: str = "maldi_adata",
    batch_col: str = "batch",
    feature_join: str = "inner",
    random_state: int = 42,
) -> SpatialData:
    """
    Merge SpatialData objects and apply batch correction.

    Parameters
    ----------
    sdatas
        SpatialData objects to merge.

    batch_names
        Batch labels corresponding to sdatas.

    pre_merged
        Already merged SpatialData object.
        If provided, merge_spatialdata is skipped.

    method
        "harmony" or "combat"

    pcs
        Number of PCs for Harmony.

    covariates
        Covariates to preserve during ComBat.

    table_name
        AnnData table name.

    batch_col
        obs column containing batch labels.

    feature_join
        "inner" or "outer"

    random_state
        Random seed for Harmony.

    Returns
    -------
    SpatialData
    """

    # -------------------------------------------------------------------------
    # Warning
    # -------------------------------------------------------------------------

    warnings.warn(
        "\n"
        "batch_correction assumes the data has already "
        "been normalised.\n"
        "Running on raw counts may produce misleading results.",
        UserWarning,
        stacklevel=2,
    )

    # -------------------------------------------------------------------------
    # Validate method
    # -------------------------------------------------------------------------

    method = method.lower()

    if method not in ("harmony", "combat"):

        raise ValueError(
            "method must be either:\n"
            "'harmony' or 'combat'"
        )

    # -------------------------------------------------------------------------
    # Validate inputs
    # -------------------------------------------------------------------------

    if pre_merged is None:

        if sdatas is None:
            raise ValueError(
                "Either 'sdatas' or 'pre_merged' "
                "must be provided."
            )

        if batch_names is None:
            raise ValueError(
                "'batch_names' must be provided "
                "when using 'sdatas'."
            )

        if len(sdatas) != len(batch_names):
            raise ValueError(
                "'sdatas' and 'batch_names' "
                "must have the same length."
            )

    # -------------------------------------------------------------------------
    # Merge or use pre-merged
    # -------------------------------------------------------------------------

    if pre_merged is not None:

        if not isinstance(pre_merged, SpatialData):
            raise TypeError(
                "pre_merged must be a SpatialData object."
            )

        merged = pre_merged

        _log(
            "Using pre-merged SpatialData object"
        )

    else:

        _log(
            f"Merging {len(sdatas)} SpatialData objects"
        )

        merged = gp.merge_spatialdata(
            sdatas=sdatas,
            batch_names=batch_names,
            table_name=table_name,
            feature_join=feature_join,
        )

    # -------------------------------------------------------------------------
    # Validate table
    # -------------------------------------------------------------------------

    if table_name not in merged.tables:

        raise KeyError(
            f"'{table_name}' not found in merged.tables"
        )

    adata = merged.tables[table_name]

    # -------------------------------------------------------------------------
    # Validate batch column
    # -------------------------------------------------------------------------

    if batch_col not in adata.obs.columns:

        raise KeyError(
            f"'{batch_col}' not found in adata.obs"
        )

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------

    _log(
        f"Input matrix: "
        f"{adata.n_obs:,} pixels × "
        f"{adata.n_vars:,} features"
    )

    unique_batches = (
        adata.obs[batch_col]
        .astype(str)
        .unique()
    )

    _log(
        f"Detected {len(unique_batches)} batches"
    )

    # -------------------------------------------------------------------------
    # Run correction
    # -------------------------------------------------------------------------

    if method == "harmony":

        merged = _run_harmony(
            merged=merged,
            table_name=table_name,
            pcs=pcs,
            batch_col=batch_col,
            random_state=random_state,
        )

    else:

        merged = _run_combat(
            merged=merged,
            table_name=table_name,
            batch_col=batch_col,
            covariates=covariates,
        )

    # -------------------------------------------------------------------------
    # Provenance
    # -------------------------------------------------------------------------

    adata.uns["batch_correction"] = {
        "method": method,
        "pcs": pcs if method == "harmony" else None,
        "covariates": (
            covariates
            if method == "combat"
            else None
        ),
        "batch_col": batch_col,
        "feature_join": feature_join,
        "n_batches": len(unique_batches),
        "used_pre_merged": pre_merged is not None,
    }

    _log(
        f"batch_correction complete "
        f"(method='{method}')"
    )

    return merged