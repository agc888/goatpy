from __future__ import annotations
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scanpy as sc
from sklearn.decomposition import PCA

from pathlib import Path
from typing import Optional, Tuple, Union

import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
from spatialdata import SpatialData


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scanpy as sc


def check_batch(
    sdata,
    table_name: str = "maldi_adata",
    batch_col: str = "batch"
):

    adata = sdata[table_name].copy()

    # -------------------------------------------------------
    # Stable batch ordering
    # -------------------------------------------------------
    batch_order = sorted(
        adata.obs[batch_col].unique()
    )

    adata.obs[batch_col] = pd.Categorical(
        adata.obs[batch_col],
        categories=batch_order,
        ordered=True
    )

    # -------------------------------------------------------
    # Convert sparse matrix if needed
    # -------------------------------------------------------
    X = adata.X

    if not isinstance(X, np.ndarray):
        X = X.toarray()

    # -------------------------------------------------------
    # Total intensity
    # -------------------------------------------------------
    total_intensity = X.sum(axis=1)

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(25, 5)
    )

    # =======================================================
    # Panel 1: Total intensity by batch
    # =======================================================
    df1 = pd.DataFrame({
        "intensity": total_intensity,
        "batch": adata.obs[batch_col].values
    })

    for b in batch_order:

        axes[0].hist(
            df1[df1["batch"] == b]["intensity"],
            bins=50,
            alpha=0.5,
            label=str(b)
        )

    axes[0].legend()
    axes[0].set_title(
        "Total intensity by batch"
    )

    # =======================================================
    # Panel 2: Log intensity distribution
    # =======================================================
    logX = np.log1p(X)

    n_cells, n_features = logX.shape

    sample_cells = min(
        5000,
        n_cells
    )

    # Randomly sample cells
    cell_idx = np.random.choice(
        n_cells,
        sample_cells,
        replace=False
    )

    df2 = pd.DataFrame({
        "value": logX[cell_idx].flatten(),
        "batch": np.repeat(
            adata.obs[batch_col]
            .values[cell_idx],
            n_features
        )
    })

    for b in batch_order:

        axes[1].hist(
            df2[df2["batch"] == b]["value"],
            bins=100,
            alpha=0.5,
            label=str(b)
        )

    axes[1].legend()

    axes[1].set_title(
        "Log intensity distribution by batch"
    )

    # =======================================================
    # Panel 3: PCA
    # =======================================================
    sc.pp.scale(
        adata,
        max_value=10
    )

    sc.tl.pca(
        adata,
        n_comps=2
    )

    sc.pl.pca(
        adata,
        color=batch_col,
        ax=axes[2],
        show=False
    )

    axes[2].set_title(
        "PCA by batch"
    )

    plt.tight_layout()
    plt.show()

"""
Spectral visualisation utilities for goatpy.

Provides ``plot_spectrum``, which plots the mean binned spectrum from a
SpatialData object and optionally overlays the original curated peak list
from a CSV file (the same format as PEAKS.csv / glycan_list.csv).

Usage
-----
>>> from goatpy.plot_spectrum import plot_spectrum

# Basic — just the mean binned spectrum
>>> plot_spectrum(sdata)

# With the original curated peak CSV overlaid
>>> plot_spectrum(sdata, peaks_csv="goatpy/data/PEAKS.csv")

# With glycan annotations
>>> plot_spectrum(sdata, peaks_csv="goatpy/data/glycan_list.csv",
...              label_col="Composition", mz_col="Theoretical m/z [M+Na]")

# Restrict m/z window
>>> plot_spectrum(sdata, mz_range=(1200, 2000))

# Plot a single pixel instead of the mean
>>> plot_spectrum(sdata, pixel_index=42)
"""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _read_peaks_csv(
    csv_path: Union[str, Path],
    mz_col: Optional[str] = None,
    label_col: Optional[str] = None,
) -> pd.DataFrame:
    """
    Read a peak CSV and return a DataFrame with columns ``mz`` and ``label``.

    Handles both goatpy CSV formats:
      - PEAKS.csv          : space-separated, column 2 is m/z, no labels
      - glycan_list.csv    : comma-separated, columns = [m/z, Composition]

    Parameters
    ----------
    csv_path   : path to CSV file
    mz_col     : explicit column name for m/z.  Auto-detected when None.
    label_col  : explicit column name for labels.  Optional.

    Returns
    -------
    DataFrame with columns:
        ``mz``    : float
        ``label`` : str (empty string when no label column found)
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Peaks CSV not found: {path}")

    # Try to detect separator
    with open(path, "r") as f:
        header = f.readline()
    sep = "," if "," in header else r"\s+"

    df = pd.read_csv(path, sep=sep, engine="python")
    df.columns = [str(c).strip().strip('"') for c in df.columns]

    # ---- auto-detect m/z column ----
    if mz_col is None:
        candidates = [
            c for c in df.columns
            if any(k in c.lower() for k in ["mz", "m/z", "mass", "theoretical"])
        ]
        if candidates:
            mz_col = candidates[0]
        else:
            # Fall back to first numeric column
            numeric_cols = df.select_dtypes(include=[float, int]).columns.tolist()
            if not numeric_cols:
                # Try coercing
                for c in df.columns:
                    try:
                        pd.to_numeric(df[c], errors="raise")
                        numeric_cols = [c]
                        break
                    except Exception:
                        continue
            if not numeric_cols:
                raise ValueError(
                    f"Could not find an m/z column in {path}. "
                    f"Columns: {list(df.columns)}. "
                    f"Pass mz_col= explicitly."
                )
            mz_col = numeric_cols[0]

    mz = pd.to_numeric(df[mz_col], errors="coerce").dropna()
    idx = mz.index

    # ---- auto-detect label column ----
    if label_col is None:
        label_candidates = [
            c for c in df.columns
            if c != mz_col and any(
                k in c.lower()
                for k in ["composition", "name", "label", "glycan", "annotation"]
            )
        ]
        label_col = label_candidates[0] if label_candidates else None

    if label_col is not None and label_col in df.columns:
        labels = df[label_col].iloc[idx].fillna("").astype(str).tolist()
    else:
        labels = [""] * len(mz)

    return pd.DataFrame({"mz": mz.values, "label": labels})


def _get_spectrum(
    sdata: SpatialData,
    table_name: str,
    pixel_index: Optional[int],
    reduce: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract a 1-D spectrum (m/z, intensity) from a SpatialData table.

    Parameters
    ----------
    sdata        : SpatialData object (output of bin_imzml or load_and_align)
    table_name   : key in sdata.tables
    pixel_index  : row index to plot; None → reduce across all pixels
    reduce       : "mean" | "median" | "max" | "sum"  (used when pixel_index is None)

    Returns
    -------
    (mz_values, intensities) : np.ndarray, np.ndarray
    """
    if table_name not in sdata.tables:
        raise KeyError(
            f"Table '{table_name}' not found. "
            f"Available: {list(sdata.tables.keys())}"
        )

    adata = sdata.tables[table_name]

    try:
        mz_values = np.array(adata.var_names, dtype=float)
    except ValueError:
        raise ValueError(
            f"var_names could not be converted to float.  "
            f"Ensure the table was created by bin_imzml (var_names = m/z strings)."
        )

    X = np.asarray(adata.X, dtype=np.float32)

    if pixel_index is not None:
        if pixel_index < 0 or pixel_index >= X.shape[0]:
            raise IndexError(
                f"pixel_index={pixel_index} out of range "
                f"[0, {X.shape[0] - 1}]."
            )
        intensities = X[pixel_index]
    else:
        fn = {"mean": np.mean, "median": np.median,
              "max": np.max, "sum": np.sum}.get(reduce)
        if fn is None:
            raise ValueError(f"reduce must be mean/median/max/sum, got '{reduce}'")
        intensities = fn(X, axis=0)

    return mz_values, intensities.astype(np.float64)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def plot_spectrum(
    sdata: SpatialData,
    peaks_csv: Optional[Union[str, Path]] = None,
    mz_col: Optional[str] = None,
    label_col: Optional[str] = None,
    table_name: str = "maldi_adata",
    pixel_index: Optional[int] = None,
    reduce: str = "mean",
    mz_range: Optional[Tuple[float, float]] = None,
    tolerance: float = 0.1,
    label_top_n: int = 20,
    label_min_intensity_pct: float = 5.0,
    figsize: Tuple[float, float] = (14, 5),
    spectrum_color: str = "#3a6fa8",
    peak_color: str = "#d9534f",
    peak_linewidth: float = 1.2,
    spectrum_linewidth: float = 0.8,
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
    show: bool = True,
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Axes:
    """
    Plot the binned spectrum from a SpatialData object, with optional overlay
    of a curated peak list CSV.

    Parameters
    ----------
    sdata : SpatialData
        Output of ``bin_imzml``, ``bin_and_align``, or ``load_and_align``.
    peaks_csv : str or Path, optional
        Path to a CSV file with curated peak m/z values (PEAKS.csv format or
        glycan_list.csv format).  When supplied, each curated peak is marked
        on the spectrum with a vertical dashed line.
    mz_col : str, optional
        Column name for m/z values in ``peaks_csv``.  Auto-detected when None.
    label_col : str, optional
        Column name for peak labels in ``peaks_csv`` (e.g. "Composition").
        Labels are drawn above the highest-intensity annotated peaks.
    table_name : str, default "maldi_adata"
        Table to read from ``sdata.tables``.
    pixel_index : int, optional
        Plot a single pixel's spectrum.  Plots the mean across all pixels
        when None.
    reduce : "mean" | "median" | "max" | "sum", default "mean"
        Reduction applied across all pixels when ``pixel_index`` is None.
    mz_range : (lo, hi), optional
        Restrict the x-axis to this m/z window.
    tolerance : float, default 0.1
        m/z tolerance (Da) used to match curated peaks to binned bin centres.
        A curated peak at 1257.5 Da will be matched if a bin centre lies
        within ±``tolerance`` Da.
    label_top_n : int, default 20
        Maximum number of peak labels drawn (to avoid overlap).  The
        highest-intensity matched peaks are labelled first.
    label_min_intensity_pct : float, default 5.0
        Only draw a label if the matched bin's intensity is ≥ this percentage
        of the maximum spectrum intensity.
    figsize : (width, height), default (14, 5)
    spectrum_color : str, default "#3a6fa8"
        Line colour for the binned spectrum.
    peak_color : str, default "#d9534f"
        Colour for curated peak overlay lines and labels.
    peak_linewidth : float, default 1.2
    spectrum_linewidth : float, default 0.8
    ax : matplotlib Axes, optional
        Plot into an existing Axes.  A new figure is created when None.
    title : str, optional
        Axes title.  Auto-generated when None.
    show : bool, default True
        Call ``plt.show()`` at the end.
    save_path : str or Path, optional
        If supplied, save the figure to this path before showing.

    Returns
    -------
    ax : matplotlib Axes

    Examples
    --------
    >>> plot_spectrum(sdata)
    >>> plot_spectrum(sdata, peaks_csv="goatpy/data/PEAKS.csv",
    ...              mz_range=(900, 2000))
    >>> plot_spectrum(sdata, peaks_csv="goatpy/data/glycan_list.csv",
    ...              mz_col="Theoretical m/z [M+Na]",
    ...              label_col="Composition",
    ...              label_top_n=30)
    >>> ax = plot_spectrum(sdata, show=False)   # embed in larger figure
    """

    # ------------------------------------------------------------------
    # 1. Extract spectrum
    # ------------------------------------------------------------------
    mz_values, intensities = _get_spectrum(sdata, table_name, pixel_index, reduce)

    # ------------------------------------------------------------------
    # 2. Optional m/z window filter
    # ------------------------------------------------------------------
    if mz_range is not None:
        lo, hi = mz_range
        mask = (mz_values >= lo) & (mz_values <= hi)
        mz_values = mz_values[mask]
        intensities = intensities[mask]
        if len(mz_values) == 0:
            raise ValueError(f"No bins found in mz_range={mz_range}.")

    # ------------------------------------------------------------------
    # 3. Load curated peaks CSV
    # ------------------------------------------------------------------
    peaks_df = None
    if peaks_csv is not None:
        peaks_df = _read_peaks_csv(peaks_csv, mz_col=mz_col, label_col=label_col)
        if mz_range is not None:
            peaks_df = peaks_df[
                (peaks_df["mz"] >= mz_range[0]) & (peaks_df["mz"] <= mz_range[1])
            ].copy()

    # ------------------------------------------------------------------
    # 4. Build figure
    # ------------------------------------------------------------------
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    # Spectrum
    ax.plot(mz_values, intensities,
            color=spectrum_color,
            linewidth=spectrum_linewidth,
            alpha=0.85,
            zorder=2,
            label="Binned spectrum")

    # Light shading under curve
    ax.fill_between(mz_values, intensities,
                    color=spectrum_color, alpha=0.07, zorder=1)

    # ------------------------------------------------------------------
    # 5. Curated peak overlay
    # ------------------------------------------------------------------
    if peaks_df is not None and len(peaks_df) > 0:
        max_intensity = float(intensities.max()) if intensities.max() > 0 else 1.0
        min_label_int = label_min_intensity_pct / 100.0 * max_intensity

        matched_peaks = []  # (mz_curated, mz_bin, intensity, label)

        for _, row in peaks_df.iterrows():
            curated_mz = float(row["mz"])
            label = str(row["label"])

            # Find nearest bin within tolerance
            diffs = np.abs(mz_values - curated_mz)
            best_idx = int(np.argmin(diffs))
            if diffs[best_idx] <= tolerance:
                matched_peaks.append((
                    curated_mz,
                    float(mz_values[best_idx]),
                    float(intensities[best_idx]),
                    label,
                ))

        # Draw vertical dashed lines for all matched peaks
        _drawn_vlines = 0
        for curated_mz, bin_mz, intensity, label in matched_peaks:
            ax.axvline(
                x=curated_mz,
                color=peak_color,
                linewidth=peak_linewidth,
                linestyle="--",
                alpha=0.6,
                zorder=3,
            )
            _drawn_vlines += 1

        # Draw labels for top-N by intensity
        label_candidates = sorted(
            [(intensity, curated_mz, bin_mz, label)
             for curated_mz, bin_mz, intensity, label in matched_peaks
             if intensity >= min_label_int and label],
            reverse=True,
        )

        # Simple collision avoidance: track last labelled x positions
        labelled_x: list[float] = []
        min_x_gap = (mz_values[-1] - mz_values[0]) * 0.015

        drawn_labels = 0
        for intensity, curated_mz, bin_mz, label in label_candidates:
            if drawn_labels >= label_top_n:
                break

            # Skip if too close to an already-labelled peak
            if any(abs(curated_mz - lx) < min_x_gap for lx in labelled_x):
                continue

            y_pos = intensity + max_intensity * 0.03
            ax.annotate(
                label,
                xy=(curated_mz, intensity),
                xytext=(curated_mz, y_pos),
                fontsize=7.5,
                color=peak_color,
                ha="center",
                va="bottom",
                rotation=90,
                annotation_clip=True,
                arrowprops=None,
                zorder=5,
            )
            labelled_x.append(curated_mz)
            drawn_labels += 1

        print(
            f"  Curated peaks: {len(peaks_df)} in CSV  |  "
            f"{len(matched_peaks)} matched within ±{tolerance} Da  |  "
            f"{drawn_labels} labelled"
        )

    # ------------------------------------------------------------------
    # 6. Formatting
    # ------------------------------------------------------------------
    ax.set_xlabel("m/z (Da)", fontsize=12)
    ax.set_ylabel(
        f"{'Pixel ' + str(pixel_index) if pixel_index is not None else reduce.capitalize()} intensity",
        fontsize=12,
    )

    if title is None:
        n_pixels = sdata.tables[table_name].shape[0]
        n_bins   = sdata.tables[table_name].shape[1]
        if pixel_index is not None:
            title = f"Pixel {pixel_index} spectrum — {n_bins:,} bins"
        else:
            title = (
                f"{reduce.capitalize()} spectrum  —  "
                f"{n_pixels:,} pixels × {n_bins:,} bins"
            )
    ax.set_title(title, fontsize=13, pad=10)

    # Legend
    legend_handles = [
        Line2D([0], [0], color=spectrum_color, linewidth=1.5,
               label="Binned spectrum"),
    ]
    if peaks_df is not None and len(peaks_df) > 0:
        legend_handles.append(
            Line2D([0], [0], color=peak_color, linewidth=peak_linewidth,
                   linestyle="--", label=f"Curated peaks (n={len(matched_peaks)})")
        )
    ax.legend(handles=legend_handles, fontsize=10, framealpha=0.6,
              loc="upper right")

    ax.xaxis.set_minor_locator(mticker.AutoMinorLocator(5))
    ax.tick_params(axis="both", which="major", labelsize=10)
    ax.set_xlim(mz_values[0], mz_values[-1])
    ax.set_ylim(bottom=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved to {save_path}")

    if show:
        plt.show()

    return ax


# ---------------------------------------------------------------------------
# Convenience: plot multiple pixels as overlaid spectra
# ---------------------------------------------------------------------------

def plot_spectra_comparison(
    sdata: SpatialData,
    pixel_indices: list[int],
    peaks_csv: Optional[Union[str, Path]] = None,
    mz_col: Optional[str] = None,
    label_col: Optional[str] = None,
    table_name: str = "maldi_adata",
    mz_range: Optional[Tuple[float, float]] = None,
    tolerance: float = 0.1,
    label_top_n: int = 15,
    figsize: Tuple[float, float] = (14, 5),
    show: bool = True,
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Axes:
    """
    Overlay spectra from multiple pixels on a single axes.

    Parameters
    ----------
    sdata : SpatialData
    pixel_indices : list of int
        Row indices to overlay.
    peaks_csv, mz_col, label_col, table_name, mz_range,
    tolerance, label_top_n, figsize, show, save_path : same as ``plot_spectrum``

    Returns
    -------
    ax : matplotlib Axes
    """
    cmap = plt.get_cmap("tab10")
    fig, ax = plt.subplots(figsize=figsize)

    mz_values_all = None
    for i, px in enumerate(pixel_indices):
        mz_values, intensities = _get_spectrum(sdata, table_name, px, "mean")

        if mz_range is not None:
            lo, hi = mz_range
            mask = (mz_values >= lo) & (mz_values <= hi)
            mz_values = mz_values[mask]
            intensities = intensities[mask]

        ax.plot(mz_values, intensities,
                linewidth=0.9, alpha=0.8,
                color=cmap(i % 10),
                label=f"Pixel {px}",
                zorder=2 + i)
        if mz_values_all is None:
            mz_values_all = mz_values

    # Curated peak overlay using mean spectrum for matching
    if peaks_csv is not None and mz_values_all is not None:
        _, mean_int = _get_spectrum(sdata, table_name, None, "mean")
        if mz_range is not None:
            lo, hi = mz_range
            mask = (mz_values_all >= lo) & (mz_values_all <= hi)
            mean_int = mean_int[mask]

        # Reuse plot_spectrum's peak overlay by passing a temp axes
        # (simpler than duplicating logic)
        _tmp_fig, _tmp_ax = plt.subplots()
        plot_spectrum(
            sdata, peaks_csv=peaks_csv,
            mz_col=mz_col, label_col=label_col,
            table_name=table_name,
            mz_range=mz_range,
            tolerance=tolerance,
            label_top_n=label_top_n,
            ax=_tmp_ax,
            show=False,
        )
        # Copy peak vlines and annotations to main axes
        for line in _tmp_ax.lines:
            if line.get_linestyle() == "--":
                ax.axvline(
                    x=line.get_xdata()[0],
                    color="#d9534f", linewidth=1.2,
                    linestyle="--", alpha=0.5, zorder=1,
                )
        for ann in _tmp_ax.texts:
            ax.add_artist(ann)
        plt.close(_tmp_fig)

    ax.set_xlabel("m/z (Da)", fontsize=12)
    ax.set_ylabel("Intensity", fontsize=12)
    ax.set_title(
        f"Spectral comparison — {len(pixel_indices)} pixels",
        fontsize=13, pad=10,
    )
    ax.legend(fontsize=9, framealpha=0.6)
    ax.xaxis.set_minor_locator(mticker.AutoMinorLocator(5))
    ax.set_ylim(bottom=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()

    return ax