"""
binning.py
==========
Data-driven m/z binning for MALDI imzML files.

Instead of loading a pre-defined peak list, this module reads every spectrum
in the imzML file, discovers all m/z values that appear across the dataset,
and bins them into a uniform grid.

Key concepts
------------
- **bin width (tolerance)**: all m/z values within `tolerance` Da of a bin
  centre are summed into that bin. A common starting point is 0.05–0.1 Da
  for unit-resolution MALDI instruments; use 0.005–0.02 for high-resolution.
- **bin centres**: derived from the observed m/z range across all spectra so
  no prior knowledge is required.
- **reduce function**: defaults to ``max`` (intensity of the tallest peak
  within the bin window), matching pyimzml's ``getionimage`` convention.
  ``sum`` is also supported and can improve SNR on dense spectra.

Usage
-----
>>> import goatpy as gp
>>> sdata = gp.bin_and_load(
...     imzml_path = "sample.imzML",
...     he_path    = "sample.svs",    # optional H&E
...     tolerance  = 0.05,
...     mz_range   = (900, 3600),     # optional subset
... )

Or just build the binned matrix without H&E registration:

>>> from goatpy.binning import bin_imzml
>>> sdata = bin_imzml("sample.imzML", tolerance=0.05)
"""

from __future__ import annotations

import gc
import os
from functools import partial
from typing import Callable, Literal, Optional, Tuple

import numpy as np
import pandas as pd
import anndata as ad
import geopandas as gpd
from shapely.geometry import box as shapely_box

from pyimzml.ImzMLParser import ImzMLParser

from spatialdata import SpatialData
from spatialdata.models import (
    Image2DModel, PointsModel, ShapesModel, TableModel,
)
from spatialdata.transformations import Identity


# ---------------------------------------------------------------------------
# Logging helper (reuses auto_align pattern)
# ---------------------------------------------------------------------------

def _log(msg: str) -> None:
    try:
        import psutil
        rss = psutil.Process(os.getpid()).memory_info().rss / 1e9
        print(f"[{rss:.2f}GB] {msg}")
    except ImportError:
        print(msg)


# ---------------------------------------------------------------------------
# discover global m/z axis
# ---------------------------------------------------------------------------

def _discover_mz_axis(
    parser: ImzMLParser,
    mz_range: Optional[Tuple[float, float]],
    tolerance: float,
    sample_fraction: float = 1.0,
) -> np.ndarray:
    """
    Scan every spectrum (or a random fraction of them) to find the full m/z
    range actually present in the file, then build a uniform bin grid.

    Parameters
    ----------
    parser : ImzMLParser
        Already-opened parser object.
    mz_range : (lo, hi) or None
        Restrict bins to this m/z window.  ``None`` uses the full range.
    tolerance : float
        Bin half-width in Da.  Bin centres are spaced ``2 * tolerance`` apart.
    sample_fraction : float
        Fraction of spectra to scan for range discovery (0–1].  Use < 1 for
        very large files where scanning every spectrum is slow.

    Returns
    -------
    bin_centres : np.ndarray, shape (n_bins,)
        Uniformly spaced bin centre m/z values.
    """
    n_spectra = len(parser.coordinates)
    step = max(1, int(1 / sample_fraction))

    global_lo = np.inf
    global_hi = -np.inf

    _log(f"  Scanning {n_spectra} spectra for m/z range "
         f"(sample_fraction={sample_fraction:.2f}) ...")

    for idx in range(0, n_spectra, step):
        mzs, _ = parser.getspectrum(idx)
        if len(mzs) == 0:
            continue
        lo, hi = float(mzs[0]), float(mzs[-1])
        if lo < global_lo:
            global_lo = lo
        if hi > global_hi:
            global_hi = hi

    if np.isinf(global_lo):
        raise ValueError("No m/z data found in file — all spectra are empty.")

    _log(f"  Observed m/z range: [{global_lo:.4f}, {global_hi:.4f}]")

    # Optionally restrict
    if mz_range is not None:
        global_lo = max(global_lo, mz_range[0])
        global_hi = min(global_hi, mz_range[1])
        _log(f"  Restricted to mz_range=[{global_lo:.4f}, {global_hi:.4f}]")

    # Build uniform grid; bin width = 2 * tolerance
    bin_width = 2.0 * tolerance
    bin_centres = np.arange(global_lo, global_hi + bin_width, bin_width)
    _log(f"  {len(bin_centres):,} bins  (width={bin_width:.4f} Da)")
    return bin_centres


# ---------------------------------------------------------------------------
# bin a single spectrum
# ---------------------------------------------------------------------------

def _bin_spectrum(
    mzs: np.ndarray,
    intensities: np.ndarray,
    bin_centres: np.ndarray,
    tolerance: float,
    reduce: Literal["max", "sum"] = "max",
) -> np.ndarray:
    """
    Map one spectrum onto the global bin grid.

    For each bin centre, collect all intensity values whose m/z falls within
    [centre - tolerance, centre + tolerance] and reduce them to a scalar.

    This is a vectorised implementation using np.searchsorted — it is O(n log n)
    rather than O(n_bins * n_peaks).

    Parameters
    ----------
    mzs : np.ndarray
        Sorted m/z values for this spectrum.
    intensities : np.ndarray
        Corresponding intensity values.
    bin_centres : np.ndarray
        Global bin centre array (sorted, uniformly spaced).
    tolerance : float
        Half-width of each bin in Da.
    reduce : "max" | "sum"
        How to combine multiple peaks within a bin.

    Returns
    -------
    binned : np.ndarray, shape (n_bins,), dtype float32
    """
    mzs = np.asarray(mzs, dtype=np.float64)
    intensities = np.asarray(intensities, dtype=np.float64)

    n_bins = len(bin_centres)
    binned = np.zeros(n_bins, dtype=np.float32)

    if len(mzs) == 0:
        return binned

    # For each bin, find peaks within [centre-tol, centre+tol] using searchsorted
    lo_mz = bin_centres - tolerance
    hi_mz = bin_centres + tolerance

    lo_idx = np.searchsorted(mzs, lo_mz, side="left")
    hi_idx = np.searchsorted(mzs, hi_mz, side="right")

    if reduce == "max":
        for i in range(n_bins):
            a, b = lo_idx[i], hi_idx[i]
            if b > a:
                binned[i] = np.max(intensities[a:b])
    elif reduce == "sum":
        for i in range(n_bins):
            a, b = lo_idx[i], hi_idx[i]
            if b > a:
                binned[i] = np.sum(intensities[a:b])
    else:
        raise ValueError(f"reduce must be 'max' or 'sum', got '{reduce}'")

    return binned


# ---------------------------------------------------------------------------
# load all spectra into a dense matrix
# ---------------------------------------------------------------------------

def _load_binned_matrix(
    imzml_path: str,
    bin_centres: np.ndarray,
    tolerance: float,
    reduce: Literal["max", "sum"] = "max",
    chunk_size: int = 500,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load every spectrum from the imzML and bin it onto the global m/z grid.

    Parameters
    ----------
    imzml_path : str
    bin_centres : np.ndarray
    tolerance : float
    reduce : "max" | "sum"
    chunk_size : int
        Log progress every this many spectra.

    Returns
    -------
    X : np.ndarray, shape (n_pixels, n_bins), float32
        Binned intensity matrix.
    coords : np.ndarray, shape (n_pixels, 2), int
        (x, y) MALDI pixel coordinates (1-based from imzML).
    bin_centres : np.ndarray, shape (n_bins,)
        The bin centres used (passed through for convenience).
    """
    parser = ImzMLParser(imzml_path)
    n_spectra = len(parser.coordinates)
    n_bins = len(bin_centres)

    X = np.zeros((n_spectra, n_bins), dtype=np.float32)
    coords = np.zeros((n_spectra, 2), dtype=np.int32)

    _log(f"  Binning {n_spectra:,} spectra × {n_bins:,} bins ...")

    for idx, (x, y, _) in enumerate(parser.coordinates):
        mzs, intensities = parser.getspectrum(idx)
        X[idx] = _bin_spectrum(mzs, intensities, bin_centres, tolerance, reduce)
        coords[idx] = [x, y]

        if (idx + 1) % chunk_size == 0 or idx == n_spectra - 1:
            _log(f"    {idx + 1:,} / {n_spectra:,}  "
                 f"({(idx + 1) / n_spectra * 100:.1f}%)")

    _log(f"  Matrix shape: {X.shape}  ({X.nbytes / 1e6:.0f} MB)")
    return X, coords, bin_centres


# ---------------------------------------------------------------------------
# optional peak filtering
# ---------------------------------------------------------------------------

def _filter_peaks(
    X: np.ndarray,
    bin_centres: np.ndarray,
    min_frequency: float = 0.0,
    min_intensity: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Drop bins that are too sparse or too weak across the dataset.

    Parameters
    ----------
    X : np.ndarray, shape (n_pixels, n_bins)
    bin_centres : np.ndarray, shape (n_bins,)
    min_frequency : float
        Drop bins detected in fewer than this fraction of pixels (0–1).
        E.g. 0.01 drops bins present in < 1 % of pixels.
    min_intensity : float
        Drop bins whose maximum intensity across all pixels is below this value.

    Returns
    -------
    X_filtered : np.ndarray
    bin_centres_filtered : np.ndarray
    """
    mask = np.ones(X.shape[1], dtype=bool)

    if min_frequency > 0.0:
        freq = (X > 0).mean(axis=0)
        mask &= freq >= min_frequency
        _log(f"  After min_frequency={min_frequency}: "
             f"{mask.sum():,} / {len(mask):,} bins retained")

    if min_intensity > 0.0:
        peak_max = X.max(axis=0)
        mask &= peak_max >= min_intensity
        _log(f"  After min_intensity={min_intensity}: "
             f"{mask.sum():,} / {len(mask):,} bins retained")

    return X[:, mask], bin_centres[mask]


# ---------------------------------------------------------------------------
# Assemble SpatialData (no H&E)
# ---------------------------------------------------------------------------

def _build_spatialdata_no_he(
    X: np.ndarray,
    bin_centres: np.ndarray,
    coords: np.ndarray,
    pixel_size: float = 1.0,
) -> SpatialData:
    """
    Build a minimal SpatialData object from binned spectra without H&E.

    Each MALDI pixel becomes a square polygon of side ``pixel_size``.

    Parameters
    ----------
    X : np.ndarray, shape (n_pixels, n_bins)
    bin_centres : np.ndarray, shape (n_bins,)
    coords : np.ndarray, shape (n_pixels, 2)  — (x, y) 1-based
    pixel_size : float
        Edge length of each pixel square in arbitrary units.

    Returns
    -------
    SpatialData
    """
    n_pixels = X.shape[0]

    # Shift to 0-based
    x = coords[:, 0].astype(float) - 1
    y = coords[:, 1].astype(float) - 1

    adata = ad.AnnData(X, dtype=np.float32)
    adata.var_names = [f"{mz:.4f}" for mz in bin_centres]
    adata.obs_names = [str(i) for i in range(n_pixels)]

    adata.obs["x"] = x
    adata.obs["y"] = y
    adata.obs["MPI"] = np.ravel(X.sum(axis=1))
    adata.obsm["spatial"] = np.column_stack([x, y])

    pixel_idx = adata.obs_names.to_numpy()
    half = pixel_size / 2.0
    geoms = [
        shapely_box(float(xi) - half, float(yi) - half,
                    float(xi) + half, float(yi) + half)
        for xi, yi in zip(x, y)
    ]
    gdf = gpd.GeoDataFrame({"cell_id": pixel_idx}, geometry=geoms)
    shapes = ShapesModel.parse(gdf, transformations={"global": Identity()})

    pts_df = pd.DataFrame({"x": x, "y": y, "cell_id": pixel_idx})
    centroids = PointsModel.parse(pts_df)

    sdata = SpatialData(
        points={"centroids": centroids},
        shapes={"pixels": shapes},
    )

    adata.obs["instance_id"] = sdata["pixels"].index
    adata.obs["region"] = pd.Categorical(["pixels"] * n_pixels)

    table = TableModel.parse(
        adata, region="pixels",
        region_key="region", instance_key="instance_id",
    )
    sdata["maldi_adata"] = table
    return sdata



def bin_imzml(
    imzml_path: str,
    tolerance: float = 0.05,
    mz_range: Optional[Tuple[float, float]] = None,
    reduce: Literal["max", "sum"] = "max",
    min_frequency: float = 0.0,
    min_intensity: float = 0.0,
    sample_fraction: float = 1.0,
    chunk_size: int = 500,
) -> SpatialData:
    """
    Load a MALDI imzML file and bin all spectra onto a uniform m/z grid.

    This replaces the manual peak-list workflow (``glyco_spatialdata`` / the
    bundled PEAKS.csv) with a data-driven approach that retains every
    detectable signal. The result is a SpatialData object compatible with all
    other goatpy functions.

    Parameters
    ----------
    imzml_path : str
        Path to the .imzML file.
    tolerance : float, default 0.05
        Half-width of each m/z bin in Da.  All peaks within
        ``[centre - tolerance, centre + tolerance]`` are collapsed to one bin.

        Recommended values:
        - Low-resolution (unit-res) MALDI:  0.1 – 0.5 Da
        - Medium-resolution:                0.02 – 0.1 Da
        - High-resolution (Orbitrap/FT):    0.002 – 0.01 Da
    mz_range : (lo, hi) or None
        Restrict the output to this m/z window, e.g. ``(900.0, 3600.0)``.
        ``None`` uses the full range found in the file.
    reduce : "max" | "sum", default "max"
        How to combine multiple peaks within one bin.

        - ``"max"``  — tallest peak wins  (matches pyimzml ``getionimage``)
        - ``"sum"``  — integrate all peaks (better SNR for dense spectra)
    min_frequency : float, default 0.0
        After binning, drop m/z bins detected in fewer than this fraction of
        pixels (0.0 = keep all).  E.g. ``0.01`` drops bins present in < 1 % of
        pixels.
    min_intensity : float, default 0.0
        Drop m/z bins whose maximum intensity across all pixels is below this
        value.
    sample_fraction : float, default 1.0
        Fraction of spectra to scan when discovering the m/z range.  Values
        < 1 speed up the range-discovery step on large files but may miss
        rare m/z values.
    chunk_size : int, default 500
        Progress is logged every ``chunk_size`` spectra.

    Returns
    -------
    SpatialData with:
        shapes["pixels"]        — one square per MALDI pixel
        points["centroids"]     — centroid of each pixel
        tables["maldi_adata"]   — AnnData, rows = pixels, columns = m/z bins

    Examples
    --------
    >>> from goatpy.binning import bin_imzml
    >>> sdata = bin_imzml("sample.imzML", tolerance=0.05, mz_range=(900, 3600))
    >>> sdata["maldi_adata"].shape
    (4200, 13500)   # depends on your data

    # Then normalise, reduce, cluster as usual:
    >>> import goatpy as gp
    >>> sdata = gp.normalize_spatialdata(sdata, table_name="maldi_adata")
    >>> sdata = gp.graphpca_spatialdata(sdata, n_components=30)
    >>> sdata = gp.get_kmean_clusters(sdata, n_clusters=8)

    # Or pass directly to load_and_align for H&E registration — just supply
    # the pre-binned sdata instead of using the imzml_path loading path.
    # (H&E registration still runs on the TIC image built from the binned matrix.)
    """
    _log(f"bin_imzml: {imzml_path}")
    _log(f"  tolerance={tolerance} Da  reduce={reduce}  mz_range={mz_range}")

    # ------------------------------------------------------------------
    # 1. Open file and discover m/z axis
    # ------------------------------------------------------------------
    _log("Step 1/4 — discovering m/z axis ...")
    parser = ImzMLParser(imzml_path)
    bin_centres = _discover_mz_axis(
        parser, mz_range, tolerance, sample_fraction
    )
    del parser
    gc.collect()

    # ------------------------------------------------------------------
    # 2. Load and bin all spectra
    # ------------------------------------------------------------------
    _log("Step 2/4 — loading and binning spectra ...")
    X, coords, bin_centres = _load_binned_matrix(
        imzml_path, bin_centres, tolerance, reduce, chunk_size
    )

    # ------------------------------------------------------------------
    # 3. Optional peak filtering
    # ------------------------------------------------------------------
    if min_frequency > 0.0 or min_intensity > 0.0:
        _log("Step 3/4 — filtering low-quality bins ...")
        X, bin_centres = _filter_peaks(X, bin_centres, min_frequency, min_intensity)
    else:
        _log("Step 3/4 — skipping filtering (min_frequency=0, min_intensity=0)")

    # ------------------------------------------------------------------
    # 4. Assemble SpatialData
    # ------------------------------------------------------------------
    _log("Step 4/4 — assembling SpatialData ...")
    sdata = _build_spatialdata_no_he(X, bin_centres, coords)

    _log(
        f"Done.  {sdata['maldi_adata'].shape[0]:,} pixels × "
        f"{sdata['maldi_adata'].shape[1]:,} m/z bins"
    )
    return sdata


def bin_and_align(
    imzml_path: str,
    he_path: str,
    tolerance: float = 0.05,
    mz_range: Optional[Tuple[float, float]] = None,
    reduce: Literal["max", "sum"] = "max",
    min_frequency: float = 0.0,
    min_intensity: float = 0.0,
    geojson_path: Optional[str] = None,
    maldi_pixel_um: Optional[float] = None,
    he_pixel_um: Optional[float] = None,
    img_upscaling: int = 10,
    buffer_px: int = 150,
    coarse_rotation_step: int = 15,
    fine_rotation_range: float = 5.0,
    fine_rotation_step: float = 1.0,
    **kwargs,
) -> SpatialData:
    """
    Bin all spectra from an imzML file and register against an H&E image.

    This is the data-driven equivalent of ``load_and_align``:  instead of
    loading a fixed peak list it bins the entire spectral space first, then
    passes the binned TIC image to the registration engine.

    Parameters
    ----------
    imzml_path, he_path : str
        Paths to imzML and H&E files.
    tolerance : float
        Bin half-width in Da — see ``bin_imzml`` for guidance.
    mz_range : (lo, hi) or None
        Restrict to this m/z window before registration.
    reduce : "max" | "sum"
        Bin reduction function.
    min_frequency, min_intensity : float
        Post-binning quality filters — see ``bin_imzml``.
    geojson_path : str or None
        Optional QuPath annotation export.
    maldi_pixel_um, he_pixel_um : float or None
        Physical pixel sizes.  Auto-detected when None.
    img_upscaling : int
        Upscaling factor for the output canvas.
    buffer_px : int
        Canvas padding at registration resolution.
    coarse_rotation_step, fine_rotation_range, fine_rotation_step : float
        Registration search parameters.

    Returns
    -------
    SpatialData
        Same structure as ``load_and_align`` output.
    """
    from .auto_align import (
        _log, _read_native_mpp, _load_he_at_resolution,
        _crop_offsets, _maldi_to_grayscale, _he_to_grayscale,
        _register, _build_affine_and_canvas, _transform_geojson,
        _build_spatialdata, _read_maldi_pixel_size,
    )
    from PIL import Image as _Image
    from pyimzml.ImzMLParser import ImzMLParser as _Parser
    from spatialdata.models import ShapesModel as _Shapes
    from spatialdata.transformations import Identity as _Id

    # ------------------------------------------------------------------
    # 1. Bin spectra
    # ------------------------------------------------------------------
    _log("=== bin_and_align: Step 1 — binning spectra ===")
    sdata_binned = bin_imzml(
        imzml_path,
        tolerance=tolerance,
        mz_range=mz_range,
        reduce=reduce,
        min_frequency=min_frequency,
        min_intensity=min_intensity,
    )
    adata = sdata_binned["maldi_adata"]
    X = np.asarray(adata.X, dtype=np.float32)
    n_pixels, n_bins = X.shape

    # Reconstruct maldi_h x maldi_w grid from obs x/y
    xs = adata.obs["x"].astype(int).to_numpy()
    ys = adata.obs["y"].astype(int).to_numpy()
    maldi_w = int(xs.max()) + 1
    maldi_h = int(ys.max()) + 1

    spectra_all = np.zeros((maldi_h, maldi_w, n_bins), dtype=np.float32)
    spectra_all[ys, xs, :] = X

    # ------------------------------------------------------------------
    # 2. MALDI pixel size
    # ------------------------------------------------------------------
    if maldi_pixel_um is None:
        detected = _read_maldi_pixel_size(imzml_path)
        maldi_pixel_um = detected if detected is not None else 10.0
        _log(f"  maldi_pixel_um={maldi_pixel_um}")

    # ------------------------------------------------------------------
    # 3. H&E pixel size
    # ------------------------------------------------------------------
    if he_pixel_um is None:
        he_pixel_um = _read_native_mpp(he_path)
        if he_pixel_um is None:
            he_pixel_um = 0.2527
            _log(f"  WARNING: H&E pixel size unknown, assuming {he_pixel_um} um/px.")
        else:
            _log(f"  H&E native pixel size: {he_pixel_um:.4f} um/px")

    # ------------------------------------------------------------------
    # 4. Registration (same pipeline as load_and_align)
    # ------------------------------------------------------------------
    _log("=== bin_and_align: Step 2 — H&E registration ===")
    tic = spectra_all.sum(axis=-1).astype(np.float32)
    maldi_gray = _maldi_to_grayscale(tic)

    he_img, loaded_mpp = _load_he_at_resolution(he_path, maldi_pixel_um, he_pixel_um)
    he_reg_w, he_reg_h = he_img.width, he_img.height
    he_gray = _he_to_grayscale(he_img)

    best_rot, best_idx = _register(
        he_gray, maldi_gray,
        src_w=he_reg_w, src_h=he_reg_h,
        coarse_step=coarse_rotation_step,
        fine_range=fine_rotation_range,
        fine_step=fine_rotation_step,
        buffer_px=buffer_px,
    )
    del he_gray, maldi_gray
    gc.collect()

    he_canvas, M_stored, canvas_pr, canvas_pc = _build_affine_and_canvas(
        he_img=he_img, src_w=he_reg_w, src_h=he_reg_h,
        rotation_deg=best_rot, buffer_px=buffer_px,
    )
    del he_img
    gc.collect()

    # ------------------------------------------------------------------
    # 5. Annotations
    # ------------------------------------------------------------------
    annotation_gdf = None
    if geojson_path is not None:
        annotation_gdf = _transform_geojson(
            geojson_path=geojson_path,
            he_pixel_um=he_pixel_um,
            reg_mpp=loaded_mpp,
            M_stored=M_stored,
            img_upscaling=img_upscaling,
        )

    # ------------------------------------------------------------------
    # 6. Build SpatialData with registered canvas
    # ------------------------------------------------------------------
    _log("=== bin_and_align: Step 3 — building SpatialData ===")
    sdata = _build_spatialdata(
        spectra_all=spectra_all,
        peaks=[float(v) for v in adata.var_names],
        maldi_pixel_um=maldi_pixel_um,
        he_canvas=he_canvas,
        maldi_offset_in_canvas=best_idx,
        reg_mpp=loaded_mpp,
        crop_r=0,
        crop_c=0,
        img_upscaling=img_upscaling,
    )

    if annotation_gdf is not None:
        ann_shapes = _Shapes.parse(annotation_gdf, transformations={"global": _Id()})
        sdata.shapes["annotations"] = ann_shapes

    sdata["maldi_adata"].uns["he_transform"] = {
        "rotation_deg":     float(best_rot),
        "maldi_offset":     [int(best_idx[0]), int(best_idx[1])],
        "he_pixel_um":      float(he_pixel_um),
        "maldi_pixel_um":   float(maldi_pixel_um),
        "reg_mpp":          float(loaded_mpp),
        "buffer_px":        int(buffer_px),
        "img_upscaling":    int(img_upscaling),
        "canvas_shape":     list(he_canvas.shape[:2]),
        "affine_matrix":    M_stored.tolist(),
        "binning": {
            "tolerance":      tolerance,
            "reduce":         reduce,
            "mz_range":       list(mz_range) if mz_range else None,
            "min_frequency":  min_frequency,
            "min_intensity":  min_intensity,
            "n_bins":         int(n_bins),
        },
    }

    _log(
        f"bin_and_align done.  "
        f"{sdata['maldi_adata'].shape[0]:,} pixels × "
        f"{sdata['maldi_adata'].shape[1]:,} m/z bins"
    )
    return sdata
