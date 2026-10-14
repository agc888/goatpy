"""
mass_alignment.py

Pixel-wise mass recalibration / alignment for MALDI mass spectrometry
imaging (MSI) data.

This mirrors the approach used by SCiLS Lab's "Mass recalibration /
alignment" tool and by peak-based imzML calibrators (e.g. Fangyh28's
imzml_calibrator): for every pixel, match a reference m/z list against
that pixel's own spectrum, quantify how far off each matched peak is,
fit a model describing that error across the mass range, and use the
model to correct the pixel's full m/z axis.

CHANGES vs. the original version (to match Bruker flexControl / SCiLS
batch calibration -- see the "Calibration" panel: Mass Control List =
5 glycans, Peak Assignment Tolerance = 350 ppm (User Defined), Mode =
Quadratic):

  1. tolerance / tolerance_unit default to Bruker's own setting
     (350 ppm) instead of a flat 0.1 Da window. ppm scales with mass,
     Da doesn't -- a flat window is systematically tighter (relative
     to instrument error) at high mass and looser at low mass.

  2. method defaults to "quadratic" (Bruker's Mode), not "linear".

  3. min_matches now defaults to requiring (almost) every reference to
     be found before fitting a *per-pixel* model. Quadratic has 3 free
     parameters -- fitting it from exactly 3 matched points (the old
     min_matches=3 default) is an EXACT fit with zero residual and zero
     robustness: one mis-picked peak (noise spike, neighboring ion) and
     the whole curve swings, often worst away from the 3 points it was
     anchored on. That produces exactly the kind of spatially-*structured*
     noise (patches tied to tissue regions with weaker signal / competing
     ions) you'd see vs. Bruker's much cleaner, more uniform error map.
     Requiring close to a full match before trusting a per-pixel fit,
     and falling back to the pooled/global model otherwise, is far more
     stable -- the fallback model is fit from hundreds of matches, not 3.

  4. Real SNR gating. match_reference_peaks always supported min_snr /
     noise_level, but nothing ever computed noise_level, so a noise
     spike inside the search window could win the "most intense" pick
     just as easily as the real peak. estimate_noise_level() below is
     now called before every match_reference_peaks call and passed in.

  5. coarse_prealign's tolerance is ppm-scaled by default too (instead
     of a flat Da value), for the same reason as (1) -- and so its wide
     window doesn't disproportionately increase the odds of grabbing
     the wrong peak at the low-mass end of the reference list.

Why per-pixel and not one global shift
---------------------------------------
In MALDI-TOF/MALDI-TOF-imaging, mass error is rarely a single constant
offset. It usually varies smoothly with m/z (different regions of the
flight-tube / detector response) and drifts pixel-to-pixel due to laser
energy, sample thickness, charging, etc. So the correction is fit
independently per pixel (or per pixel-block), not applied uniformly.

Two scales of error, two passes
--------------------------------
There's a second effect on top of the pixel-to-pixel drift described
above: a *systematic* offset that's roughly the same for every pixel
(e.g. the instrument's default calibration is just off, or a whole
region of the mass axis is shifted). That offset can easily be *bigger*
than the narrow +/- tolerance window used to match individual peaks
per-pixel -- and if it is, the per-pixel matching silently fails for
that reference on every pixel (no match found = dropped, not
corrected), which looks exactly like "this peak never gets calibrated
and drifts differently pixel to pixel".

So this module runs calibration in two passes:
    0. coarse_prealign   -> find the ~dataset-wide offset with a WIDE
                             search window, using a sample of pixels
    1. match_reference_peaks -> find each reference m/z in this pixel,
                             now searching where the coarse pass says to
                             look, with the normal narrow tolerance
    2. fit_calibration_model -> fit residual error(m/z) for this pixel
    3. apply_calibration      -> correct the pixel's full m/z axis
    4. calibrate_dataset      -> do 0-3 for every pixel + QC + fallback

Only numpy/scipy/pandas are required.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import os
from dataclasses import dataclass, field
from typing import Callable, Literal, Sequence

Method = Literal["constant", "linear", "quadratic", "loess"]
# coarse_prealign's fitted params get handed to worker processes in
# calibrate_imzml_file's parallel matching pass, so they must be plain,
# picklable numbers -- no 'loess' (which is a closure over data arrays).
CoarseMethod = Literal["constant", "linear", "quadratic"]

# Bruker "Glycans calibration list" from the Calibration panel -- kept
# here as a convenience default so calls don't need to repeat it, but
# always pass your own reference_mz explicitly; this is just documentation
# of what the reference run in the screenshot used.
BRUKER_GLYCAN_REFERENCE_MZ = [1257.4230, 1419.4750, 1581.5280, 1809.6390, 1905.6340]


# --------------------------------------------------------------------------- #
# 0. Noise estimation (new)
# --------------------------------------------------------------------------- #

def estimate_noise_level(intensity: np.ndarray, baseline_percentile: float = 25.0) -> float:
    """
    Cheap, robust noise-floor estimate for a single spectrum.

    Takes the bottom `baseline_percentile`% of intensity values (i.e. the
    points that are baseline/noise, not real peaks) and returns their
    median + a robust spread estimate (MAD, scaled to be comparable to a
    standard deviation). This is deliberately simple -- it doesn't try to
    model peak shapes -- it just needs to be good enough to tell "this is
    baseline noise" from "this is a real peak" for the min_snr gate in
    match_reference_peaks.

    Without this, a wide search window (especially in coarse_prealign, or
    in low-signal tissue regions) can just as easily pick a noise spike as
    the "most intense point in the window" as it can the real peak -- and
    that mis-pick then gets treated as ground truth for that pixel's whole
    calibration curve.
    """
    intensity = np.asarray(intensity, dtype=float)
    if intensity.size == 0:
        return 0.0
    cutoff = np.percentile(intensity, baseline_percentile)
    baseline = intensity[intensity <= cutoff]
    if baseline.size == 0:
        return 0.0
    med = np.median(baseline)
    mad = np.median(np.abs(baseline - med))
    return float(med + 1.4826 * mad)  # 1.4826 scales MAD ~= std for normal-ish noise


# --------------------------------------------------------------------------- #
# 1. Peak matching
# --------------------------------------------------------------------------- #

def match_reference_peaks(
    mz: np.ndarray,
    intensity: np.ndarray,
    reference_mz: Sequence[float],
    tolerance: float = 350.0,
    tolerance_unit: Literal["Da", "ppm"] = "ppm",
    min_snr: float = 3.0,
    noise_level: float | None = None,
) -> pd.DataFrame:
    """
    For each reference m/z, look for the best-matching peak in this
    pixel's own spectrum within +/- tolerance, and return the observed
    m/z actually found.

    Defaults now match the Bruker "User Defined" Peak Assignment
    Tolerance shown in the Calibration panel (350 ppm), and gate on SNR
    by default (min_snr=3.0) -- pass noise_level explicitly (e.g. from
    estimate_noise_level) or it's computed for you when left as None.

    Parameters
    ----------
    mz, intensity : 1D arrays for a single pixel's spectrum (profile or
        already-picked/centroided peak list both work).
    reference_mz : the "ground truth" feature list (e.g. theoretical
        glycan masses, or a consensus list built from the whole dataset).
        NOTE: this is also used as the *search center* for each window --
        see coarse_prealign below for what to do when the true peak is
        further from this than your tolerance allows.
    tolerance : search half-window around each reference mass.
    tolerance_unit : 'ppm' (default, matches Bruker's mass-scaled window
        and is recommended in general since real MS error scales with
        m/z), or 'Da' for a flat absolute window.
    min_snr, noise_level : simple SNR gate, on by default. Candidate
        peaks with intensity < min_snr * noise_level are ignored, so a
        noise spike inside the window can't win the "most intense point"
        pick over a genuine (but possibly lower, in a weak-signal pixel)
        real peak nearby. If noise_level is None it's estimated
        automatically via estimate_noise_level(intensity). Pass
        min_snr=0 to fully disable this gate (old behavior).

    Returns
    -------
    DataFrame with columns: reference_mz, observed_mz, observed_intensity,
    matched (bool), error_da, error_ppm.
    Unmatched references get observed_mz = NaN.
    """
    mz = np.asarray(mz, dtype=float)
    intensity = np.asarray(intensity, dtype=float)

    if min_snr > 0 and noise_level is None:
        noise_level = estimate_noise_level(intensity)

    # Candidate points to search within each window. For each reference
    # mass we simply take the most intense sample point inside the
    # tolerance window as the matched peak apex. This works uniformly for
    # profile spectra (the apex of a real peak is its highest sample) and
    # for already-centroided/peak-picked lists (every point already is a
    # peak). Note: scipy.find_peaks was deliberately NOT used here -- it
    # can never mark the first/last point of an array as a peak, which
    # silently drops real matches in short centroided peak lists.
    peak_mz = mz
    peak_int = intensity

    rows = []
    for ref in reference_mz:
        win = tolerance if tolerance_unit == "Da" else tolerance * ref / 1e6
        lo, hi = ref - win, ref + win
        in_window = (peak_mz >= lo) & (peak_mz <= hi)

        if noise_level is not None and min_snr > 0:
            in_window &= peak_int >= (min_snr * noise_level)

        if not np.any(in_window):
            rows.append(dict(reference_mz=ref, observed_mz=np.nan,
                              observed_intensity=np.nan, matched=False,
                              error_da=np.nan, error_ppm=np.nan))
            continue

        cand_mz = peak_mz[in_window]
        cand_int = peak_int[in_window]
        # pick the most intense candidate peak in the window (the
        # "real" apex is virtually always the dominant one there, now
        # that the SNR gate has already thrown out sub-noise-floor points)
        best = np.argmax(cand_int)
        obs_mz = cand_mz[best]
        err_da = obs_mz - ref
        rows.append(dict(reference_mz=ref, observed_mz=obs_mz,
                          observed_intensity=cand_int[best], matched=True,
                          error_da=err_da, error_ppm=1e6 * err_da / ref))

    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# 1.5 Coarse pre-alignment (fixes shifts bigger than the matching tolerance)
# --------------------------------------------------------------------------- #

def coarse_prealign(
    pixel_spectra: Sequence[tuple[np.ndarray, np.ndarray]],
    reference_mz: Sequence[float],
    tolerance: float = 1500.0,
    tolerance_unit: Literal["Da", "ppm"] = "ppm",
    method: CoarseMethod = "linear",
    min_snr: float = 3.0,
    n_sample: int = 500,
    random_state: int = 0,
) -> tuple["CalibrationModel", pd.DataFrame]:
    """
    Estimate ONE global correction for a systematic mis-assignment that's
    too large for the normal (narrow) per-pixel matching tolerance to see.

    Tolerance now defaults to ppm (1500 ppm, ~4x the 350 ppm fine-pass
    tolerance) rather than a flat Da value. A flat Da window is much
    wider *relatively* at low mass than high mass -- e.g. the old 2 Da
    default is ~1590 ppm at m/z 1257 but only ~1050 ppm at m/z 1905 --
    which made the coarse pass more likely to grab a neighboring ion
    (rather than the true, shifted reference peak) at the low-mass end
    of a reference list. A ppm window keeps that risk roughly uniform
    across the mass range. SNR gating (min_snr) is also applied here now,
    for the same "don't let the window grab a noise spike" reason as in
    match_reference_peaks -- this matters more here, not less, since the
    window is wider.

    The problem this solves
    ------------------------
    match_reference_peaks only looks +/- `tolerance` around each reference
    mass. If the *true* instrument error is bigger than that window, every
    pixel's window search comes up empty for that reference. It isn't fit
    imprecisely -- it's silently dropped (observed_mz = NaN) for every
    pixel, so nothing ever corrects it, and the pixel-to-pixel jitter on
    top of it never gets touched either, because the fine per-pixel model
    never even sees that reference.

    How
    ---
    1. Take a random sample of pixels (a few hundred is plenty -- this
       only needs the *typical* offset, not a per-pixel one).
    2. Match against `reference_mz` with a WIDE `tolerance` so peaks that
       are genuinely far from their theoretical mass can still be found.
    3. Take the MEDIAN observed m/z per reference mass across the sampled
       pixels (median, not a pooled mean, so a handful of noisy or
       wrong-window pixels can't drag the estimate around).
    4. Fit one global `method` model to those medians.

    Inspect the returned `diagnostics` table: if some references show a
    ~0 Da offset and one shows a large one, that's usually not a single
    coarse shift affecting everything -- it's more likely a typo'd
    theoretical mass in your reference list, which no calibration model
    can fix; correct the reference list itself instead.

    Returns
    -------
    model : CalibrationModel describing the coarse, dataset-wide offset.
    diagnostics : one row per reference mass found, with its median
        observed m/z and error, for sanity-checking the estimate above.

    Raises
    ------
    ValueError if nothing was found even with the widened tolerance --
    widen `tolerance` further, or check reference_mz against the raw
    spectra by eye.
    """
    if method not in ("constant", "linear", "quadratic"):
        raise ValueError("coarse_prealign only supports 'constant', 'linear', or 'quadratic'")

    n = len(pixel_spectra)
    if n > n_sample:
        rng = np.random.default_rng(random_state)
        sample_idx = rng.choice(n, size=n_sample, replace=False)
    else:
        sample_idx = np.arange(n)

    matches = []
    for i in sample_idx:
        mz, intensity = pixel_spectra[i]
        matches.append(match_reference_peaks(mz, intensity, reference_mz,
                                              tolerance=tolerance, tolerance_unit=tolerance_unit,
                                              min_snr=min_snr))
    pooled = pd.concat(matches, ignore_index=True).dropna(subset=["observed_mz"])

    if pooled.empty:
        raise ValueError(
            f"coarse_prealign found nothing within +/-{tolerance} {tolerance_unit} of any "
            "reference mass across the sampled pixels. Widen `tolerance` (try a few Da), "
            "or double check reference_mz against the raw spectra by eye first."
        )

    diagnostics = (
        pooled.groupby("reference_mz")
        .agg(observed_mz=("observed_mz", "median"), n_pixels_found=("observed_mz", "size"))
        .reset_index()
    )
    diagnostics["error_da"] = diagnostics["observed_mz"] - diagnostics["reference_mz"]
    diagnostics["error_ppm"] = 1e6 * diagnostics["error_da"] / diagnostics["reference_mz"]

    model = fit_calibration_model(diagnostics, method=method)
    return model, diagnostics


def _shift_reference_for_matching(
    reference_mz: Sequence[float], coarse_model: "CalibrationModel | None"
) -> np.ndarray:
    """
    Turn the TRUE reference m/z list into "where to actually search for
    each peak in the raw, not-yet-calibrated spectrum", using a coarse
    model -- without touching the pixel's own m/z axis at all. This lets
    a narrow matching tolerance find peaks that are systematically far
    from their theoretical mass, while every downstream error/fit is
    still computed against the true reference_mz (see
    _restore_true_reference).
    """
    reference_mz = np.asarray(reference_mz, dtype=float)
    if coarse_model is None:
        return reference_mz
    return reference_mz + coarse_model.predict_error(reference_mz)


def _restore_true_reference(matched: pd.DataFrame, reference_mz: Sequence[float]) -> pd.DataFrame:
    """
    After matching against a coarse-shifted reference list, swap the
    'reference_mz' column back to the TRUE reference masses (same order,
    so this is a straight positional replace) and recompute error_da /
    error_ppm against them, so every downstream step still measures true
    error, not error-relative-to-the-shifted-search-window.
    """
    matched = matched.copy()
    matched["reference_mz"] = np.asarray(reference_mz, dtype=float)
    matched["error_da"] = matched["observed_mz"] - matched["reference_mz"]
    matched["error_ppm"] = 1e6 * matched["error_da"] / matched["reference_mz"]
    return matched


# --------------------------------------------------------------------------- #
# 2. Calibration models
# --------------------------------------------------------------------------- #

@dataclass
class CalibrationModel:
    """A fitted correction function for one pixel."""
    method: Method
    predict_error: Callable[[np.ndarray], np.ndarray]  # error(mz) in Da
    n_points: int
    params: dict = field(default_factory=dict)


def _lowess_predict(x_fit, y_fit, frac=0.5):
    """
    Minimal LOESS (locally weighted linear regression), no external
    dependency beyond numpy. Returns a callable predict(x_query).
    Good enough for smooth, slowly-varying calibration curves; not a
    general-purpose statistics tool.
    """
    x_fit = np.asarray(x_fit, dtype=float)
    y_fit = np.asarray(y_fit, dtype=float)
    n = len(x_fit)
    k = max(3, int(np.ceil(frac * n)))

    def predict(x_query):
        x_query = np.atleast_1d(np.asarray(x_query, dtype=float))
        out = np.empty_like(x_query)
        for i, xq in enumerate(x_query):
            d = np.abs(x_fit - xq)
            idx = np.argsort(d)[:k]
            bw = d[idx].max() if d[idx].max() > 0 else 1.0
            w = (1 - np.clip(d[idx] / bw, 0, 1) ** 3) ** 3  # tricube weights
            X = np.vstack([np.ones(k), x_fit[idx]]).T
            W = np.diag(w)
            try:
                beta = np.linalg.lstsq(W @ X, W @ y_fit[idx], rcond=None)[0]
            except np.linalg.LinAlgError:
                beta = [y_fit[idx].mean(), 0.0]
            out[i] = beta[0] + beta[1] * xq
        return out

    return predict


def fit_calibration_model(
    matched: pd.DataFrame,
    method: Method = "quadratic",
    loess_frac: float = 0.5,
) -> CalibrationModel:
    """
    Fit error(m/z) = observed - reference from matched reference peaks
    for one pixel.

    method (default now "quadratic", matching Bruker's Mode setting):
        'constant'  - single median offset (use when you truly expect
                      one shift across the whole mass range)
        'linear'    - offset that varies linearly with m/z (captures a
                      slope/gain error)
        'quadratic' - adds curvature, useful over wide mass ranges.
                      NOTE: needs >=4 matched points to be an
                      overdetermined (robust) fit rather than an exact
                      3-point fit -- see min_matches in calibrate_dataset
                      / calibrate_imzml_file.
        'loess'     - smooth, non-parametric; needs more matched points
                      (~6+) to be stable
    """
    d = matched.dropna(subset=["observed_mz"])
    ref = d["reference_mz"].to_numpy()
    err = d["error_da"].to_numpy()
    n = len(ref)

    if n == 0:
        raise ValueError("No matched reference peaks to fit a calibration on.")

    if method == "constant" or n < 2:
        c = np.median(err)
        return CalibrationModel("constant", lambda mz: np.full_like(np.atleast_1d(mz), c, dtype=float),
                                 n, {"offset": c})

    if method == "linear" or (method == "quadratic" and n < 3) or (method == "loess" and n < 4):
        coeffs = np.polyfit(ref, err, 1)
        return CalibrationModel("linear", lambda mz, c=coeffs: np.polyval(c, mz),
                                 n, {"slope": coeffs[0], "intercept": coeffs[1]})

    if method == "quadratic":
        coeffs = np.polyfit(ref, err, 2)
        return CalibrationModel("quadratic", lambda mz, c=coeffs: np.polyval(c, mz),
                                 n, {"coeffs": coeffs.tolist()})

    if method == "loess":
        order = np.argsort(ref)
        predict = _lowess_predict(ref[order], err[order], frac=loess_frac)
        return CalibrationModel("loess", predict, n, {})

    raise ValueError(f"Unknown method: {method}")


# --------------------------------------------------------------------------- #
# 3. Applying the correction
# --------------------------------------------------------------------------- #

def apply_calibration(mz: np.ndarray, model: CalibrationModel) -> np.ndarray:
    """
    Correct an m/z axis using a fitted CalibrationModel.
    error(mz) is defined as observed - reference, so the corrected axis
    is mz - error(mz).
    """
    mz = np.asarray(mz, dtype=float)
    return mz - model.predict_error(mz)


# --------------------------------------------------------------------------- #
# 4. Whole-dataset alignment
# --------------------------------------------------------------------------- #

def calibrate_dataset(
    pixel_spectra: Sequence[tuple[np.ndarray, np.ndarray]],
    reference_mz: Sequence[float],
    tolerance: float = 350.0,
    tolerance_unit: Literal["Da", "ppm"] = "ppm",
    method: Method = "quadratic",
    min_matches: int | None = None,
    min_snr: float = 3.0,
    loess_frac: float = 0.5,
    coarse_prealign_enabled: bool = True,
    coarse_tolerance: float = 1500.0,
    coarse_tolerance_unit: Literal["Da", "ppm"] = "ppm",
    coarse_method: CoarseMethod = "linear",
    coarse_n_sample: int = 500,
):
    """
    Recalibrate every pixel in a dataset against a shared reference m/z
    list. Pixels with fewer than `min_matches` matched reference peaks
    fall back to a global model (fit from the pooled matches of all
    "good" pixels), so no pixel is left uncorrected.

    Runs a coarse pre-alignment pass first (see coarse_prealign) unless
    `coarse_prealign_enabled=False`, to catch systematic shifts bigger
    than `tolerance` before the fine per-pixel matching runs. Turn it off
    only if you've already confirmed every reference lands within
    `tolerance` of the raw, uncorrected data.

    Parameters
    ----------
    pixel_spectra : sequence of (mz_array, intensity_array), one per pixel.
    reference_mz : reference feature list, shared across all pixels.
    tolerance, tolerance_unit : see match_reference_peaks. This is the
        FINE, per-pixel matching window -- default 350 ppm matches the
        Bruker "User Defined" Peak Assignment Tolerance.
    method : see fit_calibration_model. Default "quadratic" matches
        Bruker's Mode setting.
    min_matches : minimum matched reference peaks required to fit a
        pixel-specific model; below this, the fallback model is used.
        Defaults to len(reference_mz) - 1 when left as None -- with
        method="quadratic" (3 free params) this means a per-pixel fit
        only happens when it's overdetermined (at least 1 spare point),
        never an exact/unstable 3-point fit. Lower this (e.g. to 3) only
        if you've checked the fallback rate isn't unreasonably high.
    min_snr : SNR gate passed to match_reference_peaks / coarse_prealign;
        set to 0 to disable.
    coarse_prealign_enabled, coarse_tolerance, coarse_tolerance_unit,
    coarse_method, coarse_n_sample : see coarse_prealign. coarse_tolerance
        defaults to ppm now (1500 ppm, ~4x the fine tolerance) so its
        width scales consistently across the reference mass range instead
        of being relatively wider at low mass; widen it if coarse_prealign
        raises a "found nothing" error.

    Returns
    -------
    calibrated_spectra : list of (calibrated_mz, intensity) per pixel,
        same order as input.
    qc : pandas.DataFrame, one row per pixel, with columns:
        n_matched, used_fallback,
        rmse_ppm_before, rmse_ppm_after,
        median_ppm_before, median_ppm_after
        -- this is the same kind of summary SCiLS' "Mass variation
        statistics" / QC View panel reports. If your spatial error map
        still looks patchy after these defaults, cross-check it against
        `qc['used_fallback']` plotted spatially -- if the noisy patches
        line up with used_fallback==False regions that had exactly
        min_matches points, that confirms unstable per-pixel fits are
        still the driver and min_matches should be raised further (or
        method dropped to "linear" in those regions).
    coarse_diagnostics : the per-reference diagnostics table from
        coarse_prealign (None if coarse_prealign_enabled=False). Check
        this first if calibration still looks wrong -- it tells you the
        dataset-wide offset that was estimated for each reference mass.
    """
    if min_matches is None:
        min_matches = max(2, len(reference_mz) - 1)

    n_pixels = len(pixel_spectra)

    # --- pass 0: coarse, wide-tolerance pre-alignment ---
    coarse_model = None
    coarse_diagnostics = None
    if coarse_prealign_enabled:
        coarse_model, coarse_diagnostics = coarse_prealign(
            pixel_spectra, reference_mz, tolerance=coarse_tolerance,
            tolerance_unit=coarse_tolerance_unit, method=coarse_method,
            min_snr=min_snr, n_sample=coarse_n_sample,
        )
    search_reference_mz = _shift_reference_for_matching(reference_mz, coarse_model)

    matches_per_pixel: list[pd.DataFrame] = []
    models: list[CalibrationModel | None] = [None] * n_pixels

    # --- pass 1: match (now searching where the coarse pass says to
    # look) + fit per-pixel models where possible ---
    all_good_matches = []
    for mz, intensity in pixel_spectra:
        m = match_reference_peaks(mz, intensity, search_reference_mz,
                                   tolerance=tolerance, tolerance_unit=tolerance_unit,
                                   min_snr=min_snr)
        m = _restore_true_reference(m, reference_mz)
        matches_per_pixel.append(m)
        n_matched = int(m["matched"].sum())
        if n_matched >= min_matches:
            models[len(matches_per_pixel) - 1] = fit_calibration_model(
                m, method=method, loess_frac=loess_frac)
            all_good_matches.append(m[m["matched"]])

    # --- fallback model from pooled matches of well-behaved pixels ---
    fallback_model = None
    if all_good_matches:
        pooled = pd.concat(all_good_matches, ignore_index=True)
        fallback_model = fit_calibration_model(pooled, method=method, loess_frac=loess_frac)
    else:
        # degenerate case: nothing matched well anywhere -> no correction
        fallback_model = CalibrationModel("constant", lambda mz: np.zeros_like(np.atleast_1d(mz), dtype=float), 0, {})

    # --- pass 2: apply models, compute QC ---
    calibrated_spectra = []
    qc_rows = []
    for i, (mz, intensity) in enumerate(pixel_spectra):
        model = models[i] if models[i] is not None else fallback_model
        used_fallback = models[i] is None
        cal_mz = apply_calibration(mz, model)
        calibrated_spectra.append((cal_mz, intensity))

        m = matches_per_pixel[i]
        matched = m[m["matched"]]
        if len(matched):
            ppm_before = matched["error_ppm"].to_numpy()
            # recompute error after calibration for QC
            corrected_obs = matched["observed_mz"].to_numpy() - model.predict_error(
                matched["observed_mz"].to_numpy())
            ppm_after = 1e6 * (corrected_obs - matched["reference_mz"].to_numpy()) / matched["reference_mz"].to_numpy()
            rmse_before, rmse_after = np.sqrt(np.mean(ppm_before ** 2)), np.sqrt(np.mean(ppm_after ** 2))
            med_before, med_after = np.median(ppm_before), np.median(ppm_after)
        else:
            rmse_before = rmse_after = med_before = med_after = np.nan

        qc_rows.append(dict(
            pixel_index=i,
            n_matched=int(matched.shape[0]),
            used_fallback=used_fallback,
            rmse_ppm_before=rmse_before,
            rmse_ppm_after=rmse_after,
            median_ppm_before=med_before,
            median_ppm_after=med_after,
        ))

    qc = pd.DataFrame(qc_rows)
    return calibrated_spectra, qc, coarse_diagnostics


# --------------------------------------------------------------------------- #
# 5. imzML file I/O (requires: pip install pyimzml)
# --------------------------------------------------------------------------- #

def load_imzml(path: str):
    """
    Load every spectrum from an .imzML file (its paired .ibd file must sit
    next to it, same base name).

    Returns
    -------
    pixel_spectra : list of (mz_array, intensity_array), one per pixel,
        in file order -- directly usable with calibrate_dataset().
    coordinates : list of (x, y, z) tuples, same order, so you can map
        results (e.g. QC ppm error) back onto the pixel grid afterwards.
    """
    from pyimzml.ImzMLParser import ImzMLParser

    parser = ImzMLParser(path)
    pixel_spectra = []
    for i, coords in enumerate(parser.coordinates):
        mz, intensity = parser.getspectrum(i)
        pixel_spectra.append((np.asarray(mz, dtype=float), np.asarray(intensity, dtype=float)))
    return pixel_spectra, list(parser.coordinates)


import re

def copy_pixel_size_metadata(source_imzml_path, target_imzml_path):
    """
    Copy the <scanSettings> pixel-size cvParams from source_imzml_path
    into target_imzml_path's scanSettingsList, since pyimzml's
    ImzMLWriter doesn't preserve them when writing a new file.
    """
    with open(source_imzml_path, "r", encoding="utf-8") as f:
        source_xml = f.read()

    # grab the whole <scanSettings ...>...</scanSettings> block from source
    m = re.search(r"<scanSettings[^>]*>.*?</scanSettings>", source_xml, re.DOTALL)
    if not m:
        raise ValueError(f"No <scanSettings> block found in {source_imzml_path}")
    source_scan_settings = m.group(0)

    with open(target_imzml_path, "r", encoding="utf-8") as f:
        target_xml = f.read()

    target_xml, n = re.subn(
        r"<scanSettings[^>]*>.*?</scanSettings>",
        source_scan_settings,
        target_xml,
        flags=re.DOTALL,
    )
    if n == 0:
        raise ValueError(f"No <scanSettings> block found in {target_imzml_path} to replace")

    with open(target_imzml_path, "w", encoding="utf-8") as f:
        f.write(target_xml)

def save_imzml(
    path: str,
    calibrated_spectra: Sequence[tuple[np.ndarray, np.ndarray]],
    coordinates: Sequence[tuple],
    spec_type: str = "centroid",
    source_imzml_path: str | None = None,
):
    """
    Write calibrated spectra back out to a new .imzML/.ibd pair.

    Written in 'processed' mode (each pixel keeps its own m/z axis),
    since per-pixel calibration means the axes are no longer identical
    across pixels -- 'continuous' mode would require resampling onto a
    shared axis first, which this function does not do.

    spec_type : 'centroid' if your input was already peak-picked,
        'profile' if you passed raw profile spectra through.
    source_imzml_path : path to the ORIGINAL, uncalibrated .imzML this
        data came from. If given, its <scanSettings> block (pixel size /
        pixel spacing) is copied into the output afterward -- ImzMLWriter
        writes its own default scanSettingsList and does not preserve
        the original's, which otherwise silently breaks downstream tools
        (e.g. goatpy auto-alignment) that read pixel size from the header.
        Leave as None only if you don't need that metadata preserved.
    """
    from pyimzml.ImzMLWriter import ImzMLWriter

    with ImzMLWriter(path, mode="processed", spec_type=spec_type) as w:
        for (mz, intensity), coords in zip(calibrated_spectra, coordinates):
            w.addSpectrum(mz, intensity, coords)

    if source_imzml_path is not None:
        copy_pixel_size_metadata(source_imzml_path, path)


def _progress(iterable, total=None, show=True, desc=""):
    """Lightweight progress reporting; uses tqdm if installed, else a
    plain periodic print so this has no hard dependency on tqdm."""
    if not show:
        yield from iterable
        return
    try:
        from tqdm import tqdm
        yield from tqdm(iterable, total=total, desc=desc)
        return
    except ImportError:
        pass
    step = max(1, (total or 0) // 20) if total else 1000
    for i, item in enumerate(iterable):
        if i % step == 0:
            pct = f" ({100 * i / total:.0f}%)" if total else ""
            print(f"{desc}: {i}{'/' + str(total) if total else ''}{pct}", flush=True)
        yield item
    print(f"{desc}: done ({total if total else i + 1})", flush=True)


# Per-worker-process globals for parallel matching (avoids re-opening /
# re-parsing the imzML header on every single spectrum -- ImzMLParser
# construction reads and indexes the whole file, which is expensive to
# repeat per spectrum but cheap to do once per worker process).
_worker_parser = None


def _init_matching_worker(path):
    global _worker_parser
    from pyimzml.ImzMLParser import ImzMLParser
    _worker_parser = ImzMLParser(path)


def _match_one(args):
    idx, search_reference_mz, tolerance, tolerance_unit, min_snr = args
    mz, intensity = _worker_parser.getspectrum(idx)
    m = match_reference_peaks(
        np.asarray(mz, dtype=float), np.asarray(intensity, dtype=float),
        search_reference_mz, tolerance=tolerance, tolerance_unit=tolerance_unit,
        min_snr=min_snr,
    )
    return idx, m


def calibrate_imzml_file(
    input_path: str,
    reference_mz: Sequence[float],
    output_path: str | None = None,
    tolerance: float = 350.0,
    tolerance_unit: Literal["Da", "ppm"] = "ppm",
    method: Method = "quadratic",
    min_matches: int | None = None,
    min_snr: float = 3.0,
    loess_frac: float = 0.5,
    spec_type: str = "centroid",
    n_jobs: int = 1,
    chunksize: int = 200,
    return_spectra: bool = False,
    show_progress: bool = True,
    coarse_prealign_enabled: bool = True,
    coarse_tolerance: float = 1500.0,
    coarse_tolerance_unit: Literal["Da", "ppm"] = "ppm",
    coarse_method: CoarseMethod = "linear",
    coarse_n_sample: int = 500,
    save_qc_folder: str | None = None,
):
    """
    End-to-end, memory-safe pipeline: calibrate every pixel of an .imzML
    file against `reference_mz`, streaming spectra through one at a time
    rather than holding the whole dataset in RAM, and (optionally) write
    the corrected data straight back out to a new .imzML file.

    Defaults now mirror the Bruker Calibration panel: tolerance=350 ppm,
    method="quadratic". min_matches defaults to len(reference_mz) - 1 so
    a per-pixel quadratic fit only happens when it's overdetermined (see
    calibrate_dataset for the full reasoning) -- pixels that don't reach
    that fall back to the pooled global model, which is much more stable
    than an exact-fit quadratic from the bare minimum 3 points.

    A coarse, wide-tolerance pre-alignment pass runs first (on a sampled
    subset of pixels, see coarse_prealign) unless
    `coarse_prealign_enabled=False`. This exists because the fine,
    narrow-tolerance per-pixel matching below can only find a reference
    peak if it's within `tolerance` of the raw, uncalibrated data --
    systematic offsets bigger than that (a mis-assigned/mis-calibrated
    peak, not just per-pixel jitter) are otherwise silently dropped from
    every pixel's fit, not corrected.

    This replaces the earlier load-everything-into-a-list approach, which
    can exhaust memory (and crash the kernel) on whole-slide datasets with
    tens of thousands of pixels -- especially at full profile resolution,
    where holding every raw AND every calibrated spectrum in memory at
    once can easily reach tens of GB.

    Passes made over the file:
      0. Coarse pre-alignment (sampled, in-memory): estimate one global
         offset from a random subset of pixels using a wide search window.
      1. Matching pass (parallelizable, CPU-bound): for every pixel, find
         the reference peaks in its own spectrum, searching where the
         coarse pass says to look. Only the small matched table (a
         handful of rows per pixel) is kept in memory here -- not the
         full m/z/intensity arrays.
      2. Apply + write pass (sequential, I/O-bound): re-reads each
         spectrum one at a time, applies its fitted calibration model,
         writes it straight to the output file (if given), and computes
         its QC row -- then discards the arrays before moving to the
         next pixel.

    Parameters
    ----------
    n_jobs : number of worker processes for the matching pass.
        1 = no multiprocessing (safest default, works everywhere).
        Set to e.g. os.cpu_count() - 1 for a large dataset. Each worker
        opens its own file handle, so this is safe with pyimzml.
    chunksize : spectra per task batch sent to each worker (only used
        when n_jobs > 1); larger values reduce IPC overhead for very
        large pixel counts.
    return_spectra : if True, also returns every calibrated (mz, intensity)
        pair in memory as `calibrated_spectra`. Leave this False for large
        datasets -- it defeats the whole point of streaming. Only turn it
        on for small files/quick inspection.
    show_progress : print/tqdm progress for both passes.
    coarse_prealign_enabled, coarse_tolerance, coarse_tolerance_unit,
    coarse_method, coarse_n_sample : see coarse_prealign. coarse_method
        is restricted to 'constant'/'linear'/'quadratic' so its fitted
        params (plain floats) can be sent to worker processes.

    Returns
    -------
    calibrated_spectra : list of (mz, intensity) if return_spectra=True,
        else None.
    coordinates : list of (x, y, z) per pixel, file order.
    qc : per-pixel QC DataFrame (with x/y columns), same as calibrate_dataset.
    coarse_diagnostics : per-reference diagnostics from coarse_prealign
        (None if coarse_prealign_enabled=False) -- check this first if
        something still looks off.
    """
    if min_matches is None:
        min_matches = max(2, len(reference_mz) - 1)

    from pyimzml.ImzMLParser import ImzMLParser

    # cheap: just reads the index / coordinate list, not spectral data
    coordinates = list(ImzMLParser(input_path).coordinates)
    n_spectra = len(coordinates)

    # ---------------- Pass 0: coarse, wide-tolerance pre-alignment ----------------
    coarse_model = None
    coarse_diagnostics = None
    if coarse_prealign_enabled:
        n_coarse_sample = min(coarse_n_sample, n_spectra)
        rng = np.random.default_rng(0)
        coarse_idx = (rng.choice(n_spectra, size=n_coarse_sample, replace=False)
                      if n_spectra > n_coarse_sample else np.arange(n_spectra))
        coarse_parser = ImzMLParser(input_path)
        coarse_sample_spectra = []
        for i in coarse_idx:
            mz, intensity = coarse_parser.getspectrum(i)
            coarse_sample_spectra.append((np.asarray(mz, dtype=float), np.asarray(intensity, dtype=float)))
        coarse_model, coarse_diagnostics = coarse_prealign(
            coarse_sample_spectra, reference_mz, tolerance=coarse_tolerance,
            tolerance_unit=coarse_tolerance_unit, method=coarse_method,
            min_snr=min_snr, n_sample=n_coarse_sample,
        )
        del coarse_parser, coarse_sample_spectra

    search_reference_mz = _shift_reference_for_matching(reference_mz, coarse_model)

    # ---------------- Pass 1: match reference peaks per pixel ----------------
    matches_per_pixel: list[pd.DataFrame] = [None] * n_spectra
    tasks = ((i, search_reference_mz, tolerance, tolerance_unit, min_snr) for i in range(n_spectra))

    if n_jobs and n_jobs > 1:
        import concurrent.futures as cf
        with cf.ProcessPoolExecutor(max_workers=n_jobs, initializer=_init_matching_worker,
                                     initargs=(input_path,)) as ex:
            for idx, m in _progress(ex.map(_match_one, tasks, chunksize=chunksize),
                                     total=n_spectra, show=show_progress, desc="Matching (parallel)"):
                matches_per_pixel[idx] = _restore_true_reference(m, reference_mz)
    else:
        _init_matching_worker(input_path)
        for args in _progress(tasks, total=n_spectra, show=show_progress, desc="Matching"):
            idx, m = _match_one(args)
            matches_per_pixel[idx] = _restore_true_reference(m, reference_mz)

    # ---------------- fit per-pixel models + pooled fallback ----------------
    models: list[CalibrationModel | None] = [None] * n_spectra
    good_matches = []
    for i, m in enumerate(matches_per_pixel):
        if int(m["matched"].sum()) >= min_matches:
            models[i] = fit_calibration_model(m, method=method, loess_frac=loess_frac)
            good_matches.append(m[m["matched"]])

    if good_matches:
        pooled = pd.concat(good_matches, ignore_index=True)
        fallback_model = fit_calibration_model(pooled, method=method, loess_frac=loess_frac)
    else:
        fallback_model = CalibrationModel(
            "constant", lambda mz: np.zeros_like(np.atleast_1d(mz), dtype=float), 0, {})

    # ---------------- Pass 2: apply calibration + stream to disk ----------------
    from pyimzml.ImzMLWriter import ImzMLWriter

    parser = ImzMLParser(input_path)
    writer = ImzMLWriter(output_path, mode="processed", spec_type=spec_type) if output_path else None
    calibrated_spectra = [] if return_spectra else None
    qc_rows = []

    try:
        for i in _progress(range(n_spectra), total=n_spectra, show=show_progress, desc="Calibrating + writing"):
            mz, intensity = parser.getspectrum(i)
            mz = np.asarray(mz, dtype=float)
            intensity = np.asarray(intensity, dtype=float)

            model = models[i] if models[i] is not None else fallback_model
            used_fallback = models[i] is None
            cal_mz = apply_calibration(mz, model)

            if writer is not None:
                writer.addSpectrum(cal_mz, intensity, coordinates[i])
            if calibrated_spectra is not None:
                calibrated_spectra.append((cal_mz, intensity))

            m = matches_per_pixel[i]
            matched = m[m["matched"]]
            if len(matched):
                ppm_before = matched["error_ppm"].to_numpy()
                corrected_obs = matched["observed_mz"].to_numpy() - model.predict_error(
                    matched["observed_mz"].to_numpy())
                ppm_after = (1e6 * (corrected_obs - matched["reference_mz"].to_numpy())
                             / matched["reference_mz"].to_numpy())
                rmse_before, rmse_after = np.sqrt(np.mean(ppm_before ** 2)), np.sqrt(np.mean(ppm_after ** 2))
                med_before, med_after = np.median(ppm_before), np.median(ppm_after)
            else:
                rmse_before = rmse_after = med_before = med_after = np.nan

            qc_rows.append(dict(
                pixel_index=i, n_matched=int(matched.shape[0]), used_fallback=used_fallback,
                rmse_ppm_before=rmse_before, rmse_ppm_after=rmse_after,
                median_ppm_before=med_before, median_ppm_after=med_after,
                x=coordinates[i][0], y=coordinates[i][1],
            ))

            # free arrays for this pixel before moving to the next
            del mz, intensity, cal_mz
    finally:
        if writer is not None:
            writer.close()

    # ImzMLWriter doesn't carry over <scanSettings> (pixel size / pixel
    # spacing cvParams) from the source file -- it writes its own default
    # scanSettingsList. Without this, downstream tools that read pixel
    # size from the imzML header (e.g. goatpy's auto-alignment) silently
    # get the wrong value from the calibrated output, even though the
    # spectra themselves are correct. Patch it back in from the original.
    if output_path is not None:
        copy_pixel_size_metadata(input_path, output_path)

    qc = pd.DataFrame(qc_rows)

    print(f"Calibration complete: ")
    print(coarse_diagnostics)
    
    if save_qc_folder is not None:
            os.makedirs(save_qc_folder, exist_ok=True)

            qc.to_csv(os.path.join(save_qc_folder, "calibration_qc_new_update.csv"), index=False)
            coarse_diagnostics.to_csv(os.path.join(save_qc_folder, "coarse_prealign_diagnostics_new_update.csv"),index=False,)
            plot_qc_spatial(qc, save_path=os.path.join(save_qc_folder, "qc_spatial.png"))

    



# --------------------------------------------------------------------------- #
# 6. Spatial QC plotting
# --------------------------------------------------------------------------- #
def plot_qc_spatial(
    qc: pd.DataFrame,
    columns: Sequence[str] = ("rmse_ppm_before", "rmse_ppm_after",
                               "median_ppm_before", "median_ppm_after"),
    x_col: str = "x",
    y_col: str = "y",
    save_path: str | None = None,
    figsize_per_panel: float = 4.0,
    dpi: int = 150,
    invert_y: bool = True,
    shared_scale: bool = True,
):
    """
    Render QC metrics (from calibrate_dataset / calibrate_imzml_file) back
    onto the pixel grid as spatial heatmaps -- one panel per column.

    Pixel coordinates don't need to be contiguous or start at (0, 0); this
    builds a dense grid spanning min..max of x and y and leaves missing
    pixels blank (NaN, shown as background color).

    'rmse_*' columns (always >= 0) use a sequential colormap from 0 to
    the 99th percentile. 'median_*' columns (signed, centered on 0 for a
    good calibration) use a diverging colormap centered at 0, with
    symmetric limits.

    shared_scale : if True (default), columns are grouped by their base
        name with any trailing '_before'/'_after' stripped (e.g.
        'median_ppm_before' and 'median_ppm_after' both become
        'median_ppm'), and every column in a group shares ONE vmin/vmax,
        computed from the combined data across the group. This is what
        makes before/after panels visually comparable -- a color at a
        given pixel means the same ppm value in both panels, so you can
        actually see whether the calibration improved things rather than
        each panel silently rescaling to its own (likely much smaller,
        post-calibration) range. Set False to restore independent
        per-column scaling.
    save_path : if given, saves the figure to this path (e.g. 'qc.png').
    invert_y : True flips the y-axis so it matches typical MSI image
        orientation (y increasing downward, as in image/pixel coordinates).

    Returns
    -------
    fig : the matplotlib Figure (also saved to save_path if given).
    """
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    import re

    x = qc[x_col].to_numpy()
    y = qc[y_col].to_numpy()
    x0, y0 = x.min(), y.min()
    width, height = x.max() - x0 + 1, y.max() - y0 + 1

    def build_grid(col):
        grid = np.full((height, width), np.nan)
        grid[y - y0, x - x0] = qc[col].to_numpy()
        return grid

    grids = {col: build_grid(col) for col in columns}

    # --- group columns by base name (strip trailing _before/_after) ---
    def base_name(col):
        return re.sub(r"_(before|after)$", "", col)

    groups: dict[str, list[str]] = {}
    for col in columns:
        groups.setdefault(base_name(col), []).append(col)

    # --- compute vmin/vmax per group (or per column if shared_scale=False) ---
    limits: dict[str, tuple[float, float]] = {}
    for col in columns:
        key = base_name(col) if shared_scale else col
        if key in limits:
            continue
        member_cols = groups[base_name(col)] if shared_scale else [col]
        combined = np.concatenate([grids[c].ravel() for c in member_cols])
        combined = combined[~np.isnan(combined)]

        if "rmse" in col:
            vmin, vmax = 0, np.percentile(combined, 99) if combined.size else 1.0
        else:
            lim = np.percentile(np.abs(combined), 99) if combined.size else 1.0
            vmin, vmax = -lim, lim
        limits[key] = (vmin, vmax)

    n = len(columns)
    ncols = min(2, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(figsize_per_panel * ncols, figsize_per_panel * nrows),
                              squeeze=False)
    axes = axes.ravel()

    for ax, col in zip(axes, columns):
        grid = grids[col]
        key = base_name(col) if shared_scale else col
        vmin, vmax = limits[key]

        cmap = (mpl.colormaps["viridis"] if "rmse" in col else mpl.colormaps["RdBu_r"]).copy()
        cmap.set_bad(color="lightgray")

        im = ax.imshow(grid, cmap=cmap, vmin=vmin, vmax=vmax, origin="upper" if invert_y else "lower",
                        interpolation="nearest")
        ax.set_title(col)
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="ppm")

    for ax in axes[n:]:
        ax.axis("off")

    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    return fig