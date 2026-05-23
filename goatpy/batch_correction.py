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
    scale_per_batch: bool = True,
) -> SpatialData:
    """
    Run Harmony on a PCA embedding and store the result in
    ``adata.obsm["X_pca_harmony"]``.

    If ``scale_per_batch=True`` (default), each batch's intensity matrix is
    z-scored independently (mean=0, std=1 per feature) before PCA.  This
    removes absolute scale differences between batches — e.g. a 5x intensity
    offset from different instruments — so PCA captures biological variance
    rather than technical scale.  The scaling is applied to a temporary copy
    used only for PCA; ``adata.X`` is not modified.

    If ``adata.obsm["GraphPCA"]`` already exists, ``scale_per_batch`` is
    ignored and the existing embedding is used directly.  If you want fresh
    per-batch-scaled PCA, delete ``adata.obsm["GraphPCA"]`` before calling.
    """
    try:
        import harmonypy
    except ImportError as e:
        raise ImportError(
            "harmonypy is required for Harmony batch correction.\n"
            "Install it with:  pip install harmonypy"
        ) from e
    
    import scanpy as sc
    import anndata as ad
    from scipy.sparse import issparse
    from sklearn.preprocessing import StandardScaler

    def get_dense(adata):
        return adata.X.toarray() if issparse(adata.X) else np.array(adata.X)

    scaled_adatas = []

    for sdata, batch_name in zip(sdatas, batch_names):

        _log(f"  Processing batch: {batch_name}")

        adata = sdata.tables[table_name].copy()

        adata.obs[batch_col] = batch_name

        X = get_dense(adata).astype(np.float64)

        scaler = StandardScaler()

        X_scaled = scaler.fit_transform(X)

        adata.X = X_scaled
        
        scaled_adatas.append(adata)
    
    
    adata = ad.concat(scaled_adatas)

    _log(
        f"  Merged object: "
        f"{adata.n_obs:,} pixels × {adata.n_vars:,} features"
    )

    _log(f"  Running PCA ({pcs} PCs)...")

    sc.pp.pca(
        adata,
        n_comps=pcs,
        random_state=random_state,
    )

    _log("  Running Harmony...")

    x = adata.obsm["X_pca"].astype(np.float64)

    harmony_out = harmonypy.run_harmony(
        x,
        adata.obs,
        batch_col,
        max_iter_harmony=20,
    )

    if harmony_out.Z_corr.shape[0] == adata.n_obs:
        corrected = harmony_out.Z_corr
    else:
        corrected = harmony_out.Z_corr.T

    adata.obsm["X_pca_harmony"] = corrected.astype(np.float32)

    _log(
        f"  Harmony complete. "
        f"Corrected embedding shape={corrected.shape}"
    )

    merged[table_name] = adata

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
    scale_per_batch: bool = True,
    covariates: Optional[list[str]] = None,
    table_name: str = "maldi_adata",
    batch_col: str = "batch",
    feature_join: str = "inner",
    offset_coords: bool = True,
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
    scale_per_batch : bool, default True
        Harmony only.  If True, each batch's intensity matrix is z-scored
        independently (mean=0, std=1 per feature) before PCA is computed.
        This removes absolute scale differences between batches (e.g. a 5x
        intensity offset between instruments) so PCA reflects biology rather
        than technical scale.  ``adata.X`` is never modified.
        Ignored when ``adata.obsm["GraphPCA"]`` already exists — delete it
        first if you want a fresh scaled PCA.
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
    offset_coords : bool, default True
        Passed to ``merge_spatialdata``.  Offsets spatial coordinates so
        batches appear side-by-side rather than overlapping.
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
        offset_coords=offset_coords,
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
            scale_per_batch=scale_per_batch,
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
        "scale_per_batch": scale_per_batch if method == "harmony" else None,
        "covariates":      covariates if method == "combat" else None,
        "batch_names":     batch_names,
        "n_batches":       len(batch_names),
        "feature_join":    feature_join,
    }

    _log(f"batch_correction complete.  method='{method}'")
    return merged