"""
batch_correction.py
===================
Batch correction for merged MALDI SpatialData objects.

Supports two methods:

* **Harmony** — operates on a PCA embedding (``adata.obsm["GraphPCA"]`` if
  available, otherwise a fresh PCA is computed).  Corrected embeddings are
  stored in ``adata.obsm["X_pca_harmony"]``.  The raw count matrix is
  **not** modified; downstream clustering should use the corrected embedding.

* **ComBat** — operates directly on the intensity matrix ``adata.X``.
  Corrected counts are written back to ``adata.X``.  An optional
  ``covariates`` argument accepts a list of additional ``obs`` column names
  to preserve during correction (passed as ``mod_combat`` design matrix).


-----------
Both methods assume the data has already been normalised (e.g. TIC or RMS
via ``gp.normalize_spatialdata``).  Batch correction on raw counts will
produce misleading results.

Usage
-----
>>> import goatpy as gp
>>> from goatpy.batch_correction import batch_correction

# Harmony (default) — corrects PCA embedding
>>> merged = batch_correction(
...     sdatas=[sdata1, sdata2],
...     batch_names=["sample_A", "sample_B"],
...     method="harmony",
...     pcs=30,
... )

# ComBat — corrects the intensity matrix directly
>>> merged = batch_correction(
...     sdatas=[sdata1, sdata2],
...     batch_names=["sample_A", "sample_B"],
...     method="combat",
... )

# ComBat with a covariate preserved during correction
>>> merged = batch_correction(
...     sdatas=[sdata1, sdata2],
...     batch_names=["sample_A", "sample_B"],
...     method="combat",
...     covariates=["annotation"],
... )
"""

from __future__ import annotations

import os
import warnings
from typing import Literal, Optional

import numpy as np
import pandas as pd

from spatialdata import SpatialData

import goatpy as gp


# ---------------------------------------------------------------------------
# Logging helper (mirrors auto_align.py)
# ---------------------------------------------------------------------------

def _log(msg: str) -> None:
    try:
        import psutil
        rss = psutil.Process(os.getpid()).memory_info().rss / 1e9
        print(f"[{rss:.2f}GB] {msg}")
    except ImportError:
        print(msg)


# ---------------------------------------------------------------------------
# Internal: Harmony correction
# ---------------------------------------------------------------------------

def _run_harmony(
    sdatas: list[SpatialData],
    batch_names: list[str],
    merged: SpatialData,
    table_name: str,
    pcs: int,
    batch_col: str,
    random_state: int,
    feature_join: str = "inner",
) -> SpatialData:
    """
    Run Harmony on a PCA embedding and store the result in
    ``adata.obsm["X_pca_harmony"]``.

    Works directly on the already-merged table so that all spatial
    coordinates (he_x, he_y, obsm["spatial"]) and obs_names set by
    merge_spatialdata are preserved.  Each batch's intensities are
    z-scored independently before PCA so absolute scale differences
    between instruments are removed; adata.X is never modified.
    """
    try:
        import harmonypy
    except ImportError as e:
        raise ImportError(
            "harmonypy is required for Harmony batch correction.\n"
            "Install it with:  pip install harmonypy"
        ) from e

    import anndata as ad
    import scanpy as sc
    from scipy.sparse import issparse
    from sklearn.preprocessing import StandardScaler

    # Work on the merged table — obs_names and spatial coords are already
    # correct from merge_spatialdata; don't replace this table.
    merged_adata = merged.tables[table_name]

    _log(
        f"  Merged object: "
        f"{merged_adata.n_obs:,} pixels × {merged_adata.n_vars:,} features"
    )

    # Z-score each batch independently on a temporary matrix so that
    # absolute intensity scale differences between instruments are removed.
    # merged_adata.X is never modified.
    X_raw = (
        merged_adata.X.toarray()
        if issparse(merged_adata.X)
        else np.array(merged_adata.X, dtype=np.float64)
    )
    X_scaled = np.empty_like(X_raw, dtype=np.float64)

    for batch_name in batch_names:
        mask = merged_adata.obs[batch_col].astype(str) == str(batch_name)
        idx = np.where(mask)[0]
        if idx.size == 0:
            _log(f"  WARNING: no pixels found for batch '{batch_name}'")
            continue
        scaler = StandardScaler()
        X_scaled[idx] = scaler.fit_transform(X_raw[idx])
        _log(f"  Scaled batch '{batch_name}': {idx.size:,} pixels")

    # Lightweight AnnData just for PCA — obs is shared so Harmony gets
    # the correct batch labels without touching any spatial metadata.
    pca_adata = ad.AnnData(
        X=X_scaled.astype(np.float32),
        obs=merged_adata.obs.copy(),
    )

    _log(f"  Running PCA ({pcs} PCs)...")
    sc.pp.pca(pca_adata, n_comps=pcs, random_state=random_state)

    _log("  Running Harmony...")
    x = pca_adata.obsm["X_pca"].astype(np.float64)

    harmony_out = harmonypy.run_harmony(
        x,
        pca_adata.obs,
        batch_col,
        max_iter_harmony=20,
    )

    corrected = harmony_out.Z_corr
    if corrected.shape[0] != merged_adata.n_obs:
        corrected = corrected.T

    # Attach embeddings to the merged table — everything else stays intact.
    merged_adata.obsm["X_pca_harmony"] = corrected.astype(np.float32)
    merged_adata.obsm["X_pca"] = pca_adata.obsm["X_pca"]

    _log(f"  Harmony complete. Corrected embedding shape={corrected.shape}")

    return merged

# ---------------------------------------------------------------------------
# Internal: ComBat correction
# ---------------------------------------------------------------------------

def _run_combat(
    merged: SpatialData,
    table_name: str,
    batch_col: str,
    covariates: Optional[list[str]],
) -> SpatialData:
    """
    Run ComBat on ``adata.X`` directly.

    Uses ``scanpy.pp.combat`` which wraps the Python port of the original
    Johnson et al. (2007) ComBat algorithm.  The corrected matrix is written
    back to ``adata.X``.
    """
    try:
        import scanpy as sc
    except ImportError as e:
        raise ImportError(
            "scanpy is required for ComBat batch correction.\n"
            "Install it with:  pip install scanpy"
        ) from e

    adata = merged.tables[table_name]

    # scanpy's combat modifies adata in-place; work on a view-safe copy of X
    X = np.asarray(adata.X, dtype=np.float64)

    # Check for non-finite values
    if not np.all(np.isfinite(X)):
        n_bad = int(np.sum(~np.isfinite(X)))
        warnings.warn(
            f"ComBat: {n_bad} non-finite values found in adata.X — "
            "replacing with 0 before correction.",
            stacklevel=3,
        )
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        adata.X = X.astype(np.float32)

    # Build covariate string for scanpy (comma-separated obs column names)
    covariates_arg = None
    if covariates:
        missing = [c for c in covariates if c not in adata.obs.columns]
        if missing:
            raise KeyError(
                f"Covariate columns not found in adata.obs: {missing}\n"
                f"Available columns: {list(adata.obs.columns)}"
            )
        covariates_arg = covariates
        _log(f"  ComBat: covariates preserved = {covariates_arg}")

    _log(f"  Running ComBat (batch_col='{batch_col}') on matrix {X.shape} ...")

    sc.pp.combat(adata, key=batch_col, covariates=covariates_arg, inplace=True)

    # Clip to non-negative (ComBat can produce small negatives)
    X_corrected = np.asarray(adata.X, dtype=np.float32)
    n_neg = int((X_corrected < 0).sum())
    if n_neg > 0:
        _log(
            f"  ComBat: clipping {n_neg:,} negative values to 0 "
            f"({n_neg / X_corrected.size * 100:.2f}% of matrix)"
        )
        X_corrected = np.clip(X_corrected, 0.0, None)
        adata.X = X_corrected

    _log(
        f"  ComBat done.  Corrected matrix stored in adata.X  "
        f"shape={X_corrected.shape}"
    )

    return merged


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def batch_correction(
    sdatas: list[SpatialData],
    batch_names: list[str],
    method: Literal["harmony", "combat"] = "harmony",
    pcs: int = 30,
    covariates: Optional[list[str]] = None,
    table_name: str = "maldi_adata",
    batch_col: str = "batch",
    feature_join: str = "inner",
    random_state: int = 42,
) -> SpatialData:
    """
    Merge multiple SpatialData objects and apply batch correction.

    This function assumes the data has already been normalised
    (e.g. TIC or RMS via ``gp.normalize_spatialdata``).  Running batch
    correction on raw, unnormalised counts is likely to produce misleading
    results.

    Parameters
    ----------
    sdatas : list of SpatialData
        Input objects to merge and correct.  Must all contain a table named
        ``table_name``.
    batch_names : list of str
        One label per SpatialData, used as the batch identifier.  Must be the
        same length as ``sdatas``.
    method : "harmony" | "combat", default "harmony"
        Batch correction algorithm to apply.

        * ``"harmony"`` — corrects a PCA embedding; does **not** modify
          ``adata.X``.  Corrected coordinates are stored in
          ``adata.obsm["X_pca_harmony"]``.  Requires ``harmonypy``.
        * ``"combat"`` — corrects ``adata.X`` directly using the ComBat
          algorithm (via ``scanpy.pp.combat``).  Requires ``scanpy``.
    pcs : int, default 30
        Number of principal components to use.  Only relevant for
        ``method="harmony"``.
    covariates : list of str, optional
        Additional ``obs`` columns to include in the ComBat design matrix as
        covariates of interest (i.e. biological variation to preserve).
        Only used when ``method="combat"``.  Example: ``["annotation"]``.
    table_name : str, default "maldi_adata"
        Key of the AnnData table inside each SpatialData object.
    batch_col : str, default "batch"
        Column name added to ``adata.obs`` by ``merge_spatialdata`` that
        holds the batch label.  You rarely need to change this.
    feature_join : "inner" | "outer", default "inner"
        Passed to ``merge_spatialdata``.  ``"inner"`` keeps only features
        present in all batches; ``"outer"`` keeps all features (missing
        values filled with 0).
    random_state : int, default 42
        Random seed for Harmony.

    Returns
    -------
    SpatialData
        Merged object with batch correction applied:

        * Harmony → ``adata.obsm["X_pca_harmony"]`` added.
        * ComBat  → ``adata.X`` replaced with corrected values.

        In both cases ``adata.uns["batch_correction"]`` records the method
        and parameters used.

    Raises
    ------
    ValueError
        If ``method`` is not ``"harmony"`` or ``"combat"``.
    ImportError
        If the required library for the chosen method is not installed.
    KeyError
        If a covariate column is not found in ``adata.obs`` (ComBat only).

    Examples
    --------
    >>> import goatpy as gp
    >>> from goatpy.batch_correction import batch_correction

    # First normalise each sample
    >>> sdata1 = gp.normalize_spatialdata(sdata1, table_name="maldi_adata")
    >>> sdata2 = gp.normalize_spatialdata(sdata2, table_name="maldi_adata")

    # Harmony (default)
    >>> merged = batch_correction(
    ...     sdatas=[sdata1, sdata2],
    ...     batch_names=["sample_A", "sample_B"],
    ...     method="harmony",
    ...     pcs=30,
    ... )

    # ComBat with a biological covariate
    >>> merged = batch_correction(
    ...     sdatas=[sdata1, sdata2],
    ...     batch_names=["sample_A", "sample_B"],
    ...     method="combat",
    ...     covariates=["annotation"],
    ... )
    """
    # ---- normalisation warning ----
    warnings.warn(
        "\n"
        "┌─────────────────────────────────────────────────────────┐\n"
        "│  batch_correction() assumes the data has already been   │\n"
        "│  normalised (e.g. via gp.normalize_spatialdata).        │\n"
        "│  Applying batch correction to raw counts will produce   │\n"
        "│  misleading results.                                     │\n"
        "└─────────────────────────────────────────────────────────┘",
        UserWarning,
        stacklevel=2,
    )

    # ---- validate method ----
    method = method.lower()
    if method not in ("harmony", "combat"):
        raise ValueError(
            f"method must be 'harmony' or 'combat', got '{method}'."
        )

    if covariates is not None and method != "combat":
        warnings.warn(
            "covariates= is only used with method='combat'.  "
            "It will be ignored for Harmony.",
            UserWarning,
            stacklevel=2,
        )

    # ---- merge ----
    _log(f"batch_correction: merging {len(sdatas)} SpatialData objects ...")
    merged = gp.merge_spatialdata(
        sdatas=sdatas,
        batch_names=batch_names,
        table_name=table_name,
        feature_join=feature_join,
    )

    n_pixels = merged.tables[table_name].n_obs
    n_vars   = merged.tables[table_name].n_vars
    _log(f"  Merged: {n_pixels:,} pixels × {n_vars:,} features")

    # ---- batch correction ----
    if method == "harmony":
        merged = _run_harmony(
            sdatas=sdatas,
            batch_names=batch_names,
            merged=merged,
            table_name=table_name,
            pcs=pcs,
            batch_col=batch_col,
            random_state=random_state, 
            feature_join=feature_join,
        )
    else:  # combat
        merged = _run_combat(
            merged,
            table_name=table_name,
            batch_col=batch_col,
            covariates=covariates,
        )

    # ---- record provenance ----
    merged.tables[table_name].uns["batch_correction"] = {
        "method":          method,
        "pcs":             pcs if method == "harmony" else None,
        "covariates":      covariates if method == "combat" else None,
        "batch_names":     batch_names,
        "n_batches":       len(batch_names),
        "feature_join":    feature_join,
    }

    _log(f"batch_correction complete.  method='{method}'")
    return merged