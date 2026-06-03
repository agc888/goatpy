"""
auto_align.py
=============
Auto-registration of a MALDI imzML dataset against a whole-slide H&E image,
or against a multichannel IHC TIFF / OME-TIFF image.

Key design decisions
--------------------
1.  Registration is performed at MALDI native pixel size (default 10 um/px)
    so the H&E thumbnail is tiny and peak RAM stays manageable.

2.  Matching uses normalised grayscale cross-correlation (TM_CCOEFF_NORMED)
    on the TIC image vs inverted H&E/IHC intensity.

3.  A two-pass rotation search (coarse 0-360, then fine) finds the correct
    slide orientation without assuming any starting angle.

4.  Registration, canvas building, and annotation transforms all use
    cv2.warpAffine with the same analytically derived affine matrix.
    This guarantees that best_idx, canvas pixels, and annotation coordinates
    all live in the same coordinate system.

IHC support
-----------
When ``ihc_path`` is supplied to ``load_and_align``, the function loads a
multichannel TIFF or OME-TIFF (any format readable by tifffile) instead of
an H&E image.  Registration uses a single channel selected by ``ihc_channel``
(default: the channel with the highest mean intensity, which is usually the
DAPI / nuclear stain — a good proxy for tissue shape).  The full multichannel
canvas is stored in ``sdata.images["ihc_image"]``; the channel used for
registration is stored in ``sdata.images["ihc_reg_channel"]`` for inspection.

Coordinate system
-----------------
All three steps share the same affine matrix M_stored:

    M_stored maps: reg-resolution image coords -> canvas coords

    _match_at_rotation  uses cv2.warpAffine(M_stored) to build the search canvas
    _build_affine_and_canvas uses cv2.warpAffine(M_stored) to build output canvas
    _transform_geojson  applies M_up @ M_stored @ M_scale to annotation vertices

Because the same matrix is used everywhere, best_idx from matchTemplate is
directly valid as the MALDI placement offset in the output canvas.
"""

from __future__ import annotations

import os
import gc
import json
from functools import partial
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import pandas as pd
import anndata as ad
import geopandas as gpd
import cv2 as cv
from PIL import Image
from shapely.geometry import box, shape
from shapely import transform as shapely_transform

from spatialdata import SpatialData
from spatialdata.models import Image2DModel, PointsModel, ShapesModel, TableModel
from spatialdata.transformations import Identity

from .io import parmap, getimage, rd_peaks, rd_peaks_from_package
from pyimzml.ImzMLParser import ImzMLParser


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def _log(msg: str) -> None:
    try:
        import psutil
        rss = psutil.Process(os.getpid()).memory_info().rss / 1e9
        print(f"[{rss:.2f}GB] {msg}")
    except ImportError:
        print(msg)


# ---------------------------------------------------------------------------
# H&E loading
# ---------------------------------------------------------------------------

def _read_native_mpp(he_path: str) -> Optional[float]:
    """Read native microns-per-pixel from file metadata without loading pixels."""
    try:
        import openslide
        slide = openslide.OpenSlide(he_path)
        mpp_x = slide.properties.get(openslide.PROPERTY_NAME_MPP_X)
        mpp_y = slide.properties.get(openslide.PROPERTY_NAME_MPP_Y)
        slide.close()
        if mpp_x and mpp_y:
            return (float(mpp_x) + float(mpp_y)) / 2.0
    except Exception:
        pass
    return None


def _load_he_at_resolution(he_path: str,
                            target_mpp: float,
                            native_mpp: float) -> tuple[Image.Image, float]:
    """
    Load H&E at (or just finer than) target_mpp using openslide pyramid
    selection, then resize to exactly target_mpp.
    """
    ext = os.path.splitext(he_path)[1].lower()
    wsi_exts = {'.svs', '.ndpi', '.scn', '.czi', '.mrxs'}

    try:
        import openslide
        slide = openslide.OpenSlide(he_path)

        best_level = 0
        best_mpp   = native_mpp
        for lvl in range(slide.level_count):
            lvl_mpp = native_mpp * slide.level_downsamples[lvl]
            if lvl_mpp <= target_mpp * 1.05:
                best_level = lvl
                best_mpp   = lvl_mpp

        dims   = slide.level_dimensions[best_level]
        region = slide.read_region((0, 0), best_level, dims)
        img    = region.convert('RGB')
        slide.close()
        _log(f"  openslide level {best_level}: {dims[0]}x{dims[1]}  "
             f"{best_mpp:.3f} um/px  ({img.width*img.height*3/1e6:.0f} MB)")

        if abs(best_mpp - target_mpp) / target_mpp > 0.02:
            scale = best_mpp / target_mpp
            nw    = max(1, round(img.width  * scale))
            nh    = max(1, round(img.height * scale))
            img   = img.resize((nw, nh), Image.Resampling.LANCZOS)
            best_mpp = target_mpp
            _log(f"  Resized to {nw}x{nh}  {target_mpp:.3f} um/px")

        return img, best_mpp

    except Exception as e:
        if ext in wsi_exts:
            raise RuntimeError(
                f"\nFailed to open '{he_path}' with openslide: {e}\n\n"
                "SVS/NDPI files require openslide:\n"
                "  conda install -c conda-forge openslide openslide-python\n"
            ) from e

    img = Image.open(he_path).convert('RGB')
    _log(f"  PIL: {img.width}x{img.height}")
    scale = native_mpp / target_mpp
    nw    = max(1, round(img.width  * scale))
    nh    = max(1, round(img.height * scale))
    img   = img.resize((nw, nh), Image.Resampling.LANCZOS)
    _log(f"  Resized to {nw}x{nh}  {target_mpp:.3f} um/px  ({nw*nh*3/1e6:.0f} MB)")
    return img, target_mpp


# ---------------------------------------------------------------------------
# IHC loading (tifffile / OME-TIFF)
# ---------------------------------------------------------------------------

def _read_ihc_native_mpp(ihc_path: str) -> Optional[float]:
    """
    Read native microns-per-pixel from an IHC TIFF / OME-TIFF.

    Tries (in order):
      1. OME-XML PhysicalSizeX / PhysicalSizeY metadata
      2. TIFF XResolution tag with ResolutionUnit = inch (2) or centimetre (3)
      3. TIFF XResolution tag with ResolutionUnit = 1 (no absolute unit) AND
         ImageDescription contains 'unit=µm' / 'unit=um' (ImageJ convention) —
         in this case XResolution is px/µm, so mpp = 1 / xres_px_per_um
      4. Returns None if nothing found
    """
    import tifffile

    try:
        with tifffile.TiffFile(ihc_path) as tif:
            # ---- OME-XML ----
            if tif.is_ome and tif.ome_metadata:
                import xml.etree.ElementTree as ET
                root = ET.fromstring(tif.ome_metadata)
                for elem in root.iter():
                    if elem.tag.endswith('Pixels'):
                        sx = elem.get('PhysicalSizeX')
                        sy = elem.get('PhysicalSizeY')
                        unit = elem.get('PhysicalSizeXUnit', 'µm')
                        if sx and sy:
                            sx, sy = float(sx), float(sy)
                            unit_factors = {
                                'µm': 1.0, 'um': 1.0, 'micrometer': 1.0,
                                'nm': 0.001, 'mm': 1000.0, 'cm': 10000.0,
                            }
                            factor = unit_factors.get(
                                unit.lower().replace('μ', 'µ').replace('\u00b5', 'µ'), 1.0
                            )
                            mpp = (sx + sy) / 2.0 * factor
                            _log(f"  IHC pixel size from OME-XML: {mpp:.4f} um/px")
                            return mpp

            # ---- TIFF resolution tags ----
            page = tif.pages[0]
            tag_xres = page.tags.get('XResolution')
            tag_unit = page.tags.get('ResolutionUnit')
            tag_desc = page.tags.get('ImageDescription')

            if tag_xres is not None:
                xres = tag_xres.value
                xres = xres[0] / xres[1] if isinstance(xres, tuple) else float(xres)
                if xres == 0:
                    return None

                unit_code = tag_unit.value if tag_unit else 2
                # tifffile returns an enum for ResolutionUnit — get its integer value
                if hasattr(unit_code, 'value'):
                    unit_code = unit_code.value
                unit_code = int(unit_code)

                if unit_code == 3:          # centimetre: xres is px/cm
                    mpp = 10000.0 / xres
                    _log(f"  IHC pixel size from TIFF tags (cm): {mpp:.4f} um/px")
                    return mpp

                elif unit_code == 2:        # inch: xres is px/inch
                    mpp = 25400.0 / xres
                    _log(f"  IHC pixel size from TIFF tags (inch): {mpp:.4f} um/px")
                    return mpp

                elif unit_code == 1:
                    # ResolutionUnit=1 means "no absolute unit defined by TIFF spec".
                    # ImageJ writes XResolution as px/µm and puts 'unit=µm' (or 'unit=um')
                    # in the ImageDescription tag.  The description may contain the µ
                    # symbol as a literal UTF-8 character, a backslash-escaped unicode
                    # sequence (\u00B5), or the ASCII approximation 'um'.
                    desc = ""
                    if tag_desc is not None:
                        raw = str(tag_desc.value)
                        # Decode any backslash-escaped unicode sequences
                        try:
                            desc = raw.encode('utf-8').decode('unicode_escape').lower()
                        except Exception:
                            desc = raw.lower()
                    is_um = (
                        'unit=\u00b5m' in desc   # µm literal
                        or 'unit=um' in desc       # ASCII approximation
                        or 'unit=µm' in desc       # another µ variant
                    )
                    if is_um:
                        mpp = 1.0 / xres
                        _log(
                            f"  IHC pixel size from ImageJ TIFF tags "
                            f"(ResolutionUnit=1, unit=µm): {mpp:.4f} um/px  "
                            f"(XResolution={xres:.6f} px/µm)"
                        )
                        return mpp

    except Exception as e:
        _log(f"  WARNING: could not read IHC pixel size ({e})")

    return None


def _load_ihc_at_resolution(
    ihc_path: str,
    target_mpp: float,
    native_mpp: float,
    ihc_channel: Optional[int] = None,
) -> tuple[np.ndarray, np.ndarray, float, list[str]]:
    """
    Load an IHC TIFF/OME-TIFF and return the full multichannel array plus
    a single-channel registration image, both downsampled to ``target_mpp``.

    The returned arrays are at **registration resolution only** — no upscaling
    is applied here.  Upscaling for the output canvas happens later inside
    ``_build_spatialdata``, consistent with the H&E path.

    Handles the common TIFF axis layouts emitted by tifffile:
      - (Y, X)        — single-channel greyscale
      - (Y, X, S)     — RGB/RGBA packed (tifffile axes string contains 'S')
      - (C, Y, X)     — channel-first (fluorescence multi-channel)
      - (Y, X, C)     — channel-last with C > 4  (rare)
      - (Z, C, Y, X)  — z-stack; first Z-plane used

    Parameters
    ----------
    ihc_path    : path to TIFF / OME-TIFF
    target_mpp  : registration resolution (um/px) — typically MALDI pixel size
    native_mpp  : native pixel size of the IHC image (um/px)
    ihc_channel : index of the channel to use for cross-correlation registration.
                  None → auto-select the channel with the highest mean intensity
                  (usually the nuclear/DAPI stain, which captures tissue shape well)

    Returns
    -------
    ihc_full_reg  : np.ndarray (C, H, W) uint8, at target_mpp
                    Full multichannel array, each channel normalised to [0, 255].
    reg_channel   : np.ndarray (H, W) float32, at target_mpp
                    The single channel used for registration (raw, un-normalised).
    loaded_mpp    : float — equals target_mpp
    channel_names : list[str]
    """
    import tifffile

    _log(f"  Loading IHC: {ihc_path}")

    with tifffile.TiffFile(ihc_path) as tif:
        img   = tif.asarray()
        axes  = tif.series[0].axes if tif.series else ""
        _log(f"  Raw array shape: {img.shape}  dtype={img.dtype}  axes='{axes}'")

        # Try to get channel names from OME metadata
        channel_names: list[str] = []
        if tif.is_ome and tif.ome_metadata:
            try:
                import xml.etree.ElementTree as ET
                root = ET.fromstring(tif.ome_metadata)
                for elem in root.iter():
                    if elem.tag.endswith('Channel'):
                        name = elem.get('Name') or elem.get('ID', '')
                        channel_names.append(name)
            except Exception:
                pass

        # Try to read channel names from ImageJ metadata Labels
        if not channel_names:
            try:
                ij_meta = tif.imagej_metadata or {}
                labels = ij_meta.get('Labels', '')
                if labels:
                    # ImageJ Labels format:
                    #   Line 0: overview description (e.g. "Overview Image_2.vsi - CF_405, CF_488, ...")
                    #   Lines 1+: per-channel entries (may be blank or a path)
                    # Extract channel names from the overview line if it contains
                    # a dash followed by a comma-separated list of channel names.
                    parts = [l.strip() for l in labels.split('\n') if l.strip()]
                    if parts:
                        overview = parts[0]
                        if ' - ' in overview:
                            ch_part = overview.split(' - ', 1)[1]
                            channel_names = [c.strip() for c in ch_part.split(',')]
            except Exception:
                pass

    # ---- Normalise to (C, H, W) ----
    img = np.squeeze(img)  # remove singleton Z, T dims

    if img.ndim == 2:
        # (Y, X) single channel
        img = img[np.newaxis, :, :]
    elif img.ndim == 3:
        # Distinguish (Y, X, S/C) from (C, Y, X).
        # tifffile uses axis label 'S' for RGB/RGBA sample dimensions.
        # Heuristic: if the last dim is <= 4 AND the first dim is much larger,
        # this is (Y, X, S) or (Y, X, C) — transpose to (S/C, Y, X).
        # This correctly handles the common (H, W, 3) RGB-packed IHC export.
        if img.shape[-1] <= 4 and img.shape[0] > img.shape[-1]:
            img = np.transpose(img, (2, 0, 1))   # (Y,X,C) -> (C,Y,X)
        # else already (C, Y, X)
    elif img.ndim == 4:
        # (Z, C, Y, X) — take first Z plane
        _log(f"  4-D array — using first plane along axis 0")
        img = img[0]
        if img.shape[-1] <= 4 and img.shape[0] > img.shape[-1]:
            img = np.transpose(img, (2, 0, 1))
    else:
        raise ValueError(
            f"Unsupported IHC array shape after squeeze: {img.shape}. "
            "Expected 2-D, 3-D, or 4-D array."
        )

    n_channels, native_h, native_w = img.shape
    _log(f"  Normalised to (C,H,W): ({n_channels}, {native_h}, {native_w})")

    # Fill channel names
    while len(channel_names) < n_channels:
        channel_names.append(f"channel_{len(channel_names)}")
    channel_names = channel_names[:n_channels]
    _log(f"  Channel names: {channel_names}")

    # ---- Select registration channel ----
    if ihc_channel is None:
        channel_means = [float(img[c].mean()) for c in range(n_channels)]
        ihc_channel = int(np.argmax(channel_means))
        _log(
            f"  Auto-selected registration channel {ihc_channel} "
            f"('{channel_names[ihc_channel]}', mean={channel_means[ihc_channel]:.1f})"
        )
    else:
        if ihc_channel >= n_channels:
            raise ValueError(
                f"ihc_channel={ihc_channel} out of range; "
                f"image has {n_channels} channels (0–{n_channels-1})."
            )
        _log(
            f"  Using registration channel {ihc_channel} "
            f"('{channel_names[ihc_channel]}')"
        )

    # ---- Resize to target_mpp (registration resolution) ----
    scale = native_mpp / target_mpp
    new_w = max(1, round(native_w * scale))
    new_h = max(1, round(native_h * scale))
    _log(f"  Resizing: ({native_h}, {native_w}) -> ({new_h}, {new_w})  "
         f"scale={scale:.4f}  ({native_mpp:.3f} -> {target_mpp:.3f} um/px)")

    # Registration channel — keep as float32 for grayscale conversion
    reg_ch_resized = cv.resize(
        img[ihc_channel].astype(np.float32),
        (new_w, new_h),
        interpolation=cv.INTER_LINEAR,
    )

    # Full multichannel — normalise each channel to uint8 for the output canvas
    ihc_full_resized = np.zeros((n_channels, new_h, new_w), dtype=np.uint8)
    for c in range(n_channels):
        ch = img[c].astype(np.float32)
        ch_small = cv.resize(ch, (new_w, new_h), interpolation=cv.INTER_LINEAR)
        ch_min, ch_max = ch_small.min(), ch_small.max()
        if ch_max > ch_min:
            ihc_full_resized[c] = (
                (ch_small - ch_min) / (ch_max - ch_min) * 255
            ).astype(np.uint8)
        # else leave as zeros

    _log(
        f"  IHC at reg resolution: ({n_channels}, {new_h}, {new_w})  "
        f"{target_mpp:.3f} um/px  ({ihc_full_resized.nbytes / 1e6:.1f} MB)"
    )

    return ihc_full_resized, reg_ch_resized, target_mpp, channel_names


def _ihc_to_grayscale(
    reg_channel: np.ndarray,
    invert: bool = False,
) -> np.ndarray:
    """
    Normalise an IHC registration channel to float32 [0, 1].

    Parameters
    ----------
    reg_channel : (H, W) float32 array at registration resolution
    invert      : if True, invert the image before normalisation.
                  H&E images are inverted so that tissue (dark) becomes
                  bright and matches the bright MALDI TIC signal.
                  For fluorescence IHC, tissue is already bright, so
                  inversion is usually NOT needed (invert=False).

    Returns
    -------
    (H, W) float32 in [0, 1]
    """
    gray = reg_channel.astype(np.float32)
    if invert:
        # Invert relative to the channel's own range
        mn, mx = gray.min(), gray.max()
        if mx > mn:
            gray = (mx - gray)   # flip so bright->dark and dark->bright
        else:
            gray = gray * 0.0

    mn, mx = gray.min(), gray.max()
    norm = (gray - mn) / (mx - mn) if mx > mn else gray * 0.0
    _log(
        f"  IHC grayscale: {norm.shape}  mean={norm.mean():.3f}  "
        f"(invert={invert})"
    )
    return norm.astype(np.float32)


def _build_ihc_canvas(
    ihc_full: np.ndarray,
    src_w: int,
    src_h: int,
    rotation_deg: float,
    buffer_px: int,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """
    Build the IHC output canvas by applying the same affine transform used
    for registration.  The canvas is at **registration resolution** — no
    upscaling is applied here.  Upscaling happens later inside
    ``_build_spatialdata``, exactly as for the H&E canvas.

    Parameters
    ----------
    ihc_full     : (C, H, W) uint8 at registration resolution
    src_w, src_h : dimensions of the registration-resolution IHC image
    rotation_deg : best rotation from _register()
    buffer_px    : canvas padding, must match value used in _register()

    Returns
    -------
    canvas      : (C, H_canvas, W_canvas) uint8 — warped, reg-resolution
    M_stored    : (3, 3) float64 — reg-res coords -> canvas coords
    canvas_pr   : int — row placement offset
    canvas_pc   : int — col placement offset
    """
    M_stored, canvas_w, canvas_h, canvas_pc, canvas_pr = _build_affine_matrix(
        src_w, src_h, rotation_deg, buffer_px
    )

    M_cv = M_stored[:2, :]
    n_channels = ihc_full.shape[0]
    canvas = np.zeros((n_channels, canvas_h, canvas_w), dtype=np.uint8)

    for c in range(n_channels):
        canvas[c] = cv.warpAffine(
            ihc_full[c],
            M_cv,
            (canvas_w, canvas_h),
            flags=cv.INTER_LINEAR,
            borderValue=0,
        )

    _log(
        f"  IHC canvas (cv2): {canvas_w}x{canvas_h}  "
        f"channels={n_channels}  pr={canvas_pr}, pc={canvas_pc}  "
        f"rotation={rotation_deg}"
    )

    return canvas, M_stored, canvas_pr, canvas_pc


# ---------------------------------------------------------------------------
# MALDI loading
# ---------------------------------------------------------------------------

def _load_spectra(imzml_path: str,
                  peaks: list[float],
                  chunk_size: int = 10,
                  crop_r: int = 0,
                  crop_c: int = 0,
                  tol: float = 0.1,
                  reduce_func: Any = sum) -> np.ndarray:
    from pyimzml.ImzMLParser import getionimage
    p0    = ImzMLParser(imzml_path)
    probe = getionimage(p0, peaks[0], tol=tol, reduce_func=reduce_func)
    h     = probe.shape[0] - crop_r
    w     = probe.shape[1] - crop_c
    del probe

    out = np.zeros((h, w, len(peaks)), dtype=np.float32)
    for start in range(0, len(peaks), chunk_size):
        batch = peaks[start: start + chunk_size]
        imgs  = parmap(partial(getimage, path=imzml_path, tol=tol, reduce_func=reduce_func), batch,
                       nprocs=min(len(batch), 4))
        for j, img in enumerate(imgs):
            out[:, :, start + j] = img[crop_r:, crop_c:]
        del imgs
        _log(f"  Peaks {start+1}-{min(start+len(batch), len(peaks))} / {len(peaks)}")
    return out


def _read_maldi_pixel_size(imzml_path: str) -> Optional[float]:
    try:
        p = ImzMLParser(imzml_path)
        for key in ['pixel size (x)', 'pixel size x', 'pixel size']:
            val = p.imzmldict.get(key)
            if val is not None:
                return float(val)
    except Exception:
        pass
    return None


def _crop_offsets(spectra_sum: np.ndarray, cutoff: float = 0.5) -> tuple[int, int]:
    try:
        crop_c = int(max(np.where(np.sum(spectra_sum, axis=0) < cutoff)[0]))
        crop_r = int(max(np.where(np.sum(spectra_sum, axis=1) < cutoff)[0]))
        return crop_r, crop_c
    except (ValueError, IndexError):
        return 0, 0


# ---------------------------------------------------------------------------
# Image preparation
# ---------------------------------------------------------------------------

def _maldi_to_grayscale(tic: np.ndarray) -> np.ndarray:
    blurred = cv.GaussianBlur(tic, (3, 3), 0)
    mn, mx  = blurred.min(), blurred.max()
    norm    = (blurred - mn) / (mx - mn) if mx > mn else blurred * 0.0
    _log(f"  MALDI grayscale: {norm.shape}  mean={norm.mean():.3f}")
    return norm.astype(np.float32)


def _he_to_grayscale(he_img: Image.Image) -> np.ndarray:
    gray = np.array(he_img.convert('L'), dtype=np.float32)
    inv  = 255.0 - gray
    mn, mx = inv.min(), inv.max()
    norm   = (inv - mn) / (mx - mn) if mx > mn else inv * 0.0
    _log(f"  H&E grayscale: {norm.shape}  mean={norm.mean():.3f}")
    return norm.astype(np.float32)


# ---------------------------------------------------------------------------
# Affine matrix construction (shared by registration and canvas building)
# ---------------------------------------------------------------------------

def _build_affine_matrix(
    src_w: int,
    src_h: int,
    rotation_deg: float,
    buffer_px: int,
    min_w: int = 0,
    min_h: int = 0,
) -> tuple[np.ndarray, int, int, int, int]:
    """
    Compute the affine matrix that:
        1. Rotates CCW around the image centre ((src_w-1)/2, (src_h-1)/2)
        2. Shifts so the rotated bounding box starts at (0, 0)
        3. Centres the result in a buffer canvas, which is guaranteed to be
           at least (min_w, min_h) pixels so matchTemplate never fails.

    Returns
    -------
    M_stored  : np.ndarray (3, 3) -- reg-res coords -> canvas coords
    canvas_w  : int
    canvas_h  : int
    canvas_pc : int -- col offset
    canvas_pr : int -- row offset
    """
    theta = np.deg2rad(rotation_deg)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    cx = (src_w - 1) / 2.0
    cy = (src_h - 1) / 2.0

    corners = np.array([
        [0.0,       0.0      ],
        [src_w - 1, 0.0      ],
        [src_w - 1, src_h - 1],
        [0.0,       src_h - 1],
    ], dtype=np.float64)

    dx = corners[:, 0] - cx
    dy = corners[:, 1] - cy
    rot_x = cos_t * dx - sin_t * dy + cx
    rot_y = sin_t * dx + cos_t * dy + cy

    expand_x = rot_x.min()
    expand_y = rot_y.min()

    rot_w = int(np.ceil(rot_x.max() - rot_x.min()))
    rot_h = int(np.ceil(rot_y.max() - rot_y.min()))

    canvas_w  = max(rot_w + buffer_px, min_w)
    canvas_h  = max(rot_h + buffer_px, min_h)
    canvas_pc = (canvas_w - rot_w) // 2
    canvas_pr = (canvas_h - rot_h) // 2

    tx = -cos_t * cx + sin_t * cy + cx - expand_x + canvas_pc
    ty = -sin_t * cx - cos_t * cy + cy - expand_y + canvas_pr

    M_stored = np.array([
        [ cos_t, -sin_t, tx],
        [ sin_t,  cos_t, ty],
        [   0.0,    0.0, 1.0],
    ], dtype=np.float64)

    return M_stored, canvas_w, canvas_h, canvas_pc, canvas_pr


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

def _match_at_rotation(
    he_gray: np.ndarray,
    maldi_gray: np.ndarray,
    rotation: float,
    buffer_px: int,
) -> tuple[float, tuple[int, int]]:
    src_h, src_w = he_gray.shape
    tmpl_h, tmpl_w = maldi_gray.shape

    M_stored, canvas_w, canvas_h, _, _ = _build_affine_matrix(
        src_w, src_h, rotation, buffer_px,
        min_w=tmpl_w, min_h=tmpl_h,
    )

    M_cv = M_stored[:2, :]
    canvas = cv.warpAffine(
        he_gray,
        M_cv,
        (canvas_w, canvas_h),
        flags=cv.INTER_LINEAR,
        borderValue=0.0,
    )

    result           = cv.matchTemplate(canvas, maldi_gray, cv.TM_CCOEFF_NORMED)
    _, score, _, loc = cv.minMaxLoc(result)
    return float(score), (int(loc[1]), int(loc[0]))   # (row, col)


def _register(
    he_gray: np.ndarray,
    maldi_gray: np.ndarray,
    src_w: int,
    src_h: int,
    coarse_step: int = 15,
    fine_range: float = 5.0,
    fine_step: float = 1.0,
    buffer_px: int = 150,
) -> tuple[float, tuple[int, int]]:
    coarse_rots = list(range(0, 360, coarse_step))
    _log(f"  Coarse: {len(coarse_rots)} rotations (0-360 step {coarse_step}) ...")

    best_score = -np.inf
    best_rot   = 0.0
    best_idx   = (0, 0)

    for rot in coarse_rots:
        score, idx = _match_at_rotation(he_gray, maldi_gray, rot, buffer_px)
        _log(f"    {rot:5.1f}  score={score:.4f}")
        if score > best_score:
            best_score, best_rot, best_idx = score, float(rot), idx

    _log(f"  Best coarse: {best_rot}  score={best_score:.4f}")

    fine_rots = sorted({
        round(best_rot + d, 1)
        for d in np.arange(-fine_range, fine_range + fine_step, fine_step)
        if abs(d) > 1e-6
    })
    _log(f"  Fine: {len(fine_rots)} rotations (+-{fine_range} step {fine_step}) ...")

    for rot in fine_rots:
        score, idx = _match_at_rotation(he_gray, maldi_gray, rot, buffer_px)
        _log(f"    {rot:5.1f}  score={score:.4f}")
        if score > best_score:
            best_score, best_rot, best_idx = score, rot, idx

    _log(f"  Final: {best_rot}  score={best_score:.4f}  offset={best_idx}")
    return best_rot, best_idx


# ---------------------------------------------------------------------------
# Build H&E output canvas
# ---------------------------------------------------------------------------

def _build_affine_and_canvas(
    he_img: Image.Image,
    src_w: int,
    src_h: int,
    rotation_deg: float,
    buffer_px: int,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    M_stored, canvas_w, canvas_h, canvas_pc, canvas_pr = _build_affine_matrix(
        src_w, src_h, rotation_deg, buffer_px
    )

    img_np = np.array(he_img, dtype=np.uint8)
    M_cv   = M_stored[:2, :]
    canvas = cv.warpAffine(
        img_np,
        M_cv,
        (canvas_w, canvas_h),
        flags=cv.INTER_LINEAR,
        borderValue=(0, 0, 0),
    )

    _log(f"  H&E canvas (cv2): {canvas_w}x{canvas_h}  "
         f"pr={canvas_pr}, pc={canvas_pc}  rotation={rotation_deg}")

    return canvas, M_stored, canvas_pr, canvas_pc


# ---------------------------------------------------------------------------
# Annotation transform
# ---------------------------------------------------------------------------

def _transform_geojson(
    geojson_path: Union[str, Path],
    he_pixel_um: float,
    reg_mpp: float,
    M_stored: np.ndarray,
    img_upscaling: int,
    classification_key: str = "classification",
) -> gpd.GeoDataFrame:
    with open(geojson_path, "r") as f:
        geojson = json.load(f)

    features = geojson if isinstance(geojson, list) else geojson.get("features", [])
    if not features:
        raise ValueError(f"No features found in {geojson_path}")

    scale_to_reg = he_pixel_um / reg_mpp
    us           = float(img_upscaling)

    M_scale = np.array([
        [scale_to_reg, 0.0,          0.0],
        [0.0,          scale_to_reg, 0.0],
        [0.0,          0.0,          1.0],
    ], dtype=np.float64)

    M_up = np.array([
        [us,  0.0, 0.0],
        [0.0, us,  0.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)

    M = M_up @ M_stored @ M_scale

    a, b, tx = M[0, 0], M[0, 1], M[0, 2]
    d, e, ty = M[1, 0], M[1, 1], M[1, 2]

    def _apply(coords: np.ndarray) -> np.ndarray:
        x, y = coords[:, 0], coords[:, 1]
        return np.column_stack([
            a * x + b * y + tx,
            d * x + e * y + ty,
        ])

    geoms, labels, names = [], [], []
    for feat in features:
        geom_raw = feat.get("geometry")
        if geom_raw is None:
            continue
        geoms.append(shapely_transform(shape(geom_raw), _apply))
        props  = feat.get("properties") or {}
        clf    = props.get("classification") or {}
        labels.append(clf.get("name", "unknown") if isinstance(clf, dict) else str(clf))
        names.append(props.get("name", ""))

    return gpd.GeoDataFrame(
        {classification_key: labels, "name": names},
        geometry=geoms,
    )


# ---------------------------------------------------------------------------
# SpatialData construction
# ---------------------------------------------------------------------------

def _build_spatialdata(spectra_all: np.ndarray,
                       peaks: list[float],
                       maldi_pixel_um: float,
                       he_canvas: np.ndarray,
                       maldi_offset_in_canvas: tuple[int, int],
                       reg_mpp: float,
                       crop_r: int,
                       crop_c: int,
                       img_upscaling: int = 10,
                       library_id: str = "spatial",
                       ihc_canvas: Optional[np.ndarray] = None,
                       ihc_channel_names: Optional[list[str]] = None) -> SpatialData:

    maldi_h, maldi_w, n_peaks = spectra_all.shape
    scale         = maldi_pixel_um / reg_mpp
    local_off_r, local_off_c = maldi_offset_in_canvas

    us      = img_upscaling
    he_up_h = he_canvas.shape[0] * us
    he_up_w = he_canvas.shape[1] * us
    he_up   = np.array(
        Image.fromarray(he_canvas).resize(
            (he_up_w, he_up_h), Image.Resampling.NEAREST
        ),
        dtype=np.uint8,
    )
    _log(f"  H&E upscaled {us}x: {he_up_w}x{he_up_h}  ({he_up.nbytes/1e6:.0f} MB)")

    grid_r, grid_c = np.mgrid[0: maldi_h, 0: maldi_w]
    he_r = ((local_off_r + (grid_r.flatten() + 0.5) * scale) * us)
    he_c = ((local_off_c + (grid_c.flatten() + 0.5) * scale) * us)

    adata = ad.AnnData(spectra_all.reshape(-1, n_peaks).copy(), dtype=np.float32)
    adata.var_names = np.array([str(pk) for pk in peaks])
    adata.obs_names = np.array([str(i) for i in range(maldi_h * maldi_w)])

    yy, xx = np.mgrid[crop_r: maldi_h + crop_r, crop_c: maldi_w + crop_c]
    adata.obs["x"]   = xx.flatten()
    adata.obs["y"]   = yy.flatten()
    adata.obs["MPI"] = np.ravel(adata.X.sum(axis=1))

    adata.obsm["spatial"] = np.column_stack([he_c, he_r])
    adata.obs["he_x"]     = he_c
    adata.obs["he_y"]     = he_r

    adata.uns["spatial"] = {
        library_id: {
            "images": {"hires": he_up},
            "use_quality": "hires",
            "scalefactors": {
                "tissue_hires_scalef": 1.0,
                "spot_diameter_fullres": float(us),
            },
        }
    }

    pixel_idx = np.arange(maldi_h * maldi_w).astype(str)
    half      = us / 2.0
    geoms     = [
        box(float(c) - half, float(r) - half,
            float(c) + half, float(r) + half)
        for r, c in zip(he_r, he_c)
    ]
    gdf    = gpd.GeoDataFrame({"cell_id": pixel_idx}, geometry=geoms)
    shapes = ShapesModel.parse(gdf, transformations={"global": Identity()})

    pts_df    = pd.DataFrame({"x": he_c, "y": he_r, "cell_id": pixel_idx})
    centroids = PointsModel.parse(pts_df)

    # H&E image model (CYX)
    image_cyx = np.transpose(he_up, (2, 0, 1))
    img_model = Image2DModel.parse(
        image_cyx, dims=("c", "y", "x"),
        transformations={"global": Identity()},
    )

    images_dict = {"he_image": img_model}

    # IHC image model: canvas is (C, H, W) at reg resolution — upscale exactly
    # as H&E, then store.  The coordinate system is identical because both canvases
    # were produced by _build_affine_matrix with the same arguments.
    if ihc_canvas is not None:
        ihc_up_h = ihc_canvas.shape[1] * us
        ihc_up_w = ihc_canvas.shape[2] * us
        n_ihc_ch = ihc_canvas.shape[0]
        ihc_up = np.zeros((n_ihc_ch, ihc_up_h, ihc_up_w), dtype=np.uint8)
        for c in range(n_ihc_ch):
            ihc_up[c] = cv.resize(
                ihc_canvas[c], (ihc_up_w, ihc_up_h), interpolation=cv.INTER_NEAREST
            )
        _log(f"  IHC upscaled {us}x: ({n_ihc_ch}, {ihc_up_h}, {ihc_up_w})  "
             f"({ihc_up.nbytes/1e6:.0f} MB)")

        ihc_channel_names = ihc_channel_names or [
            f"channel_{i}" for i in range(n_ihc_ch)
        ]
        ihc_model = Image2DModel.parse(
            ihc_up,
            dims=("c", "y", "x"),
            c_coords=ihc_channel_names[:n_ihc_ch],
            transformations={"global": Identity()},
        )
        images_dict["ihc_image"] = ihc_model
        _log(f"  IHC image stored: channels={ihc_channel_names[:n_ihc_ch]}")

        del images_dict["he_image"]


    sdata = SpatialData(
        images=images_dict,
        points={"centroids": centroids},
        shapes={"pixels": shapes},
    )

    adata.obs["instance_id"] = sdata["pixels"].index
    adata.obs["region"]      = "pixels"
    adata.obs["region"]      = adata.obs["region"].astype("category")

    table = TableModel.parse(
        adata, region="pixels",
        region_key="region", instance_key="instance_id",
    )
    sdata["maldi_adata"] = table
    return sdata


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_and_align(
    imzml_path: str,
    he_path: Optional[str] = None,
    peaks_path: Optional[str] = None,
    geojson_path: Optional[Union[str, Path]] = None,
    geojson_shapes_key: str = "annotations",
    geojson_classification_key: str = "classification",
    maldi_pixel_um: Optional[float] = None,
    he_pixel_um: Optional[float] = None,
    spectra_chunk_size: int = 10,
    coarse_rotation_step: int = 15,
    fine_rotation_range: float = 5.0,
    fine_rotation_step: float = 1.0,
    buffer_px: int = 150,
    img_upscaling: int = 10,
    tol: float = 0.1,
    reduce_func: Any = sum,
    # ---- IHC parameters ----
    ihc_path: Optional[str] = None,
    ihc_pixel_um: Optional[float] = None,
    ihc_channel: Optional[int] = None,
    ihc_invert: bool = False,
) -> SpatialData:
    """
    Load a MALDI imzML dataset and an H&E or IHC image, auto-register them,
    and return a merged SpatialData object.

    Exactly one of ``he_path`` or ``ihc_path`` must be supplied.  When
    ``ihc_path`` is given the function loads a multichannel TIFF / OME-TIFF
    (any format readable by ``tifffile``) and uses a single channel for
    registration.  The full multichannel canvas is stored in
    ``sdata.images["ihc_image"]``; the H&E slot is left empty.

    Parameters
    ----------
    imzml_path : str
        Path to the .imzML file.
    he_path : str or None
        Path to an H&E image (SVS, NDPI, TIFF, PNG, …).
        Supply either ``he_path`` or ``ihc_path``, not both.
    peaks_path : str or None
        Path to peaks CSV.  Uses bundled PEAKS.csv when None.
    geojson_path : str, Path, or None
        Optional QuPath GeoJSON annotation export.
    geojson_shapes_key : str, default "annotations"
    geojson_classification_key : str, default "classification"
    maldi_pixel_um : float or None
        MALDI pixel size in µm.  Auto-read from imzML when None.
    he_pixel_um : float or None
        H&E native pixel size in µm.  Auto-read from metadata when None.
        Also used as the pixel size for IHC when ``ihc_pixel_um`` is None.
    spectra_chunk_size : int
    coarse_rotation_step : int
    fine_rotation_range : float
    fine_rotation_step : float
    buffer_px : int
    img_upscaling : int
    tol : float
    reduce_func : callable

    IHC parameters
    --------------
    ihc_path : str or None
        Path to a multichannel IHC TIFF / OME-TIFF.
        Supply either ``he_path`` or ``ihc_path``, not both.
    ihc_pixel_um : float or None
        Native IHC pixel size in µm.  Auto-read from OME-XML / TIFF tags
        when None.  Falls back to ``he_pixel_um`` and then to 0.2527 µm/px.
    ihc_channel : int or None
        Index of the channel to use for registration.
        None (default) → auto-select the channel with the highest mean
        intensity (typically the nuclear stain, which captures tissue shape).
    ihc_invert : bool, default False
        Invert the registration channel before matching.
        H&E images are automatically inverted (tissue is dark → bright after
        inversion, matching the bright MALDI TIC signal).
        For brightfield IHC, set ``ihc_invert=True``.
        For fluorescence IHC, leave as False (tissue is already bright).

    Returns
    -------
    SpatialData with:
        images['he_image']             -- H&E canvas (if he_path supplied)
        images['ihc_image']            -- multichannel IHC canvas (if ihc_path supplied)
        shapes['pixels']               -- one square per MALDI pixel
        shapes[geojson_shapes_key]     -- annotations (if geojson_path given)
        points['centroids']            -- centroid of each MALDI pixel
        tables['maldi_adata']          -- AnnData with ion intensities
    """
    # ---- validate input ----
    if he_path is None and ihc_path is None:
        raise ValueError(
            "Supply either he_path (H&E image) or ihc_path (IHC TIFF/OME-TIFF)."
        )
    if he_path is not None and ihc_path is not None:
        raise ValueError(
            "Supply either he_path or ihc_path, not both. "
            "To register against H&E and keep an IHC image for inspection, "
            "register first (he_path only), then add the IHC manually via "
            "sdata.images['ihc_image'] = ..."
        )

    use_ihc = ihc_path is not None

    # ------------------------------------------------------------------
    # 0. Peaks
    # ------------------------------------------------------------------
    _log("Loading peaks ...")
    peaks = rd_peaks(peaks_path) if peaks_path else rd_peaks_from_package()
    peaks = sorted(peaks)
    _log(f"  {len(peaks)} peaks")

    # ------------------------------------------------------------------
    # 0b. Reference image native pixel size
    # ------------------------------------------------------------------
    if use_ihc:
        # Determine ihc_pixel_um
        if ihc_pixel_um is None:
            ihc_pixel_um = _read_ihc_native_mpp(ihc_path)
            if ihc_pixel_um is None and he_pixel_um is not None:
                ihc_pixel_um = he_pixel_um
                _log(f"  IHC pixel size not in metadata, using he_pixel_um={ihc_pixel_um}")
            if ihc_pixel_um is None:
                ihc_pixel_um = 0.2527
                _log(f"  WARNING: IHC pixel size unknown, assuming {ihc_pixel_um} um/px.")
            else:
                _log(f"  IHC native pixel size: {ihc_pixel_um:.4f} um/px")

        ref_pixel_um = ihc_pixel_um

        # Get native IHC dimensions for pixel size validation
        import tifffile
        with tifffile.TiffFile(ihc_path) as tif:
            page = tif.pages[0]
            _ihc_native_h, _ihc_native_w = page.shape[:2]
        ref_phys_w_um = _ihc_native_w * ref_pixel_um
        ref_phys_h_um = _ihc_native_h * ref_pixel_um

    else:
        # H&E path — existing logic unchanged
        if he_pixel_um is None:
            he_pixel_um = _read_native_mpp(he_path)
            if he_pixel_um is None:
                try:
                    _img = Image.open(he_path)
                    tag_info = getattr(_img, 'tag_v2', {})
                    xres = tag_info.get(282)
                    unit = tag_info.get(296, 2)
                    if xres is not None:
                        xres = xres[0] / xres[1] if isinstance(xres, tuple) else float(xres)
                        he_pixel_um = (10000.0 / xres) if unit == 3 else (25400.0 / xres)
                        _log(f"  H&E pixel size from TIFF tags: {he_pixel_um:.4f} um/px")
                    _img.close()
                except Exception:
                    pass
            if he_pixel_um is None:
                he_pixel_um = 0.2527
                _log(f"  WARNING: H&E pixel size unknown, assuming {he_pixel_um} um/px.")
            else:
                _log(f"  H&E native pixel size: {he_pixel_um:.4f} um/px")

        ref_pixel_um = he_pixel_um

        try:
            _he_probe = Image.open(he_path)
            _he_native_w, _he_native_h = _he_probe.size
            _he_probe.close()
        except Exception:
            _he_native_w, _he_native_h = 10000, 10000
        ref_phys_w_um = _he_native_w * ref_pixel_um
        ref_phys_h_um = _he_native_h * ref_pixel_um

    # ------------------------------------------------------------------
    # 0c. MALDI pixel size
    # ------------------------------------------------------------------
    if maldi_pixel_um is None:
        detected = _read_maldi_pixel_size(imzml_path)
        if detected is not None:
            _log(f"  MALDI pixel size from imzML metadata: {detected} um/px")
            _p_probe = ImzMLParser(imzml_path)
            _maldi_h = _p_probe.imzmldict.get('max count of pixels y', 1)
            _maldi_w = _p_probe.imzmldict.get('max count of pixels x', 1)
            _ref_thumb_w = ref_phys_w_um / detected
            _ref_thumb_h = ref_phys_h_um / detected
            if _ref_thumb_w >= _maldi_w and _ref_thumb_h >= _maldi_h:
                maldi_pixel_um = detected
                _log(f"  Validated: ref thumbnail ({_ref_thumb_w:.0f}x{_ref_thumb_h:.0f} px) "
                     f">= MALDI ({_maldi_w}x{_maldi_h} px)")
            else:
                _log(f"  WARNING: imzML pixel size {detected} um makes ref thumbnail "
                     f"({_ref_thumb_w:.0f}x{_ref_thumb_h:.0f} px) smaller than MALDI "
                     f"({_maldi_w}x{_maldi_h} px) -- likely wrong.")
                for candidate in [10.0, 20.0, 50.0, 100.0, 200.0]:
                    _cw = ref_phys_w_um / candidate
                    _ch = ref_phys_h_um / candidate
                    if _cw >= _maldi_w * 0.5 and _ch >= _maldi_h * 0.5:
                        maldi_pixel_um = candidate
                        _log(f"  Auto-selected maldi_pixel_um={candidate} um/px")
                        break
                if maldi_pixel_um is None:
                    maldi_pixel_um = 10.0
                    _log(f"  Falling back to maldi_pixel_um=10.0 um/px.")
        else:
            maldi_pixel_um = 10.0
            _log(f"  Pixel size not in imzML, defaulting to {maldi_pixel_um} um/px.")
    else:
        _log(f"  MALDI pixel size (supplied): {maldi_pixel_um} um/px")

    _log(f"  maldi_pixel_um={maldi_pixel_um}  ref_pixel_um={ref_pixel_um:.4f}")

    # ------------------------------------------------------------------
    # 2. MALDI crop offsets
    # ------------------------------------------------------------------
    _log("Computing MALDI crop offsets ...")
    tic_probe = np.nansum(
        np.stack([getimage(pk, path=imzml_path, tol=tol, reduce_func=reduce_func) for pk in peaks[:5]], axis=-1),
        axis=-1,
    )
    crop_r, crop_c = _crop_offsets(tic_probe)
    _log(f"  Crop: row={crop_r}, col={crop_c}")
    del tic_probe
    gc.collect()

    # ------------------------------------------------------------------
    # 3. Load spectra
    # ------------------------------------------------------------------
    _log(f"Loading {len(peaks)} ion images (chunk={spectra_chunk_size}) with {tol} Da tolerance per peak ...")
    spectra_all = _load_spectra(
        imzml_path, peaks,
        chunk_size=spectra_chunk_size,
        crop_r=crop_r, crop_c=crop_c,
        tol=tol,
        reduce_func=reduce_func,
    )
    _log(f"  spectra_all: {spectra_all.shape}  ({spectra_all.nbytes/1e6:.0f} MB)")

    # ------------------------------------------------------------------
    # 4. MALDI registration image
    # ------------------------------------------------------------------
    _log("Preparing MALDI template ...")
    maldi_tic  = spectra_all.sum(axis=-1).astype(np.float32)
    maldi_gray = _maldi_to_grayscale(maldi_tic)
    del maldi_tic
    gc.collect()

    # ------------------------------------------------------------------
    # 5 & 6. Load reference image and build registration grayscale
    # ------------------------------------------------------------------
    ihc_full_reg   = None
    ihc_ch_names   = None
    loaded_mpp     = None

    if use_ihc:
        _log(f"Loading IHC at {maldi_pixel_um} um/px ...")
        ihc_full_reg, ihc_reg_ch, loaded_mpp, ihc_ch_names = _load_ihc_at_resolution(
            ihc_path=ihc_path,
            target_mpp=maldi_pixel_um,
            native_mpp=ihc_pixel_um,
            ihc_channel=ihc_channel,
        )
        ref_reg_w = ihc_full_reg.shape[2]
        ref_reg_h = ihc_full_reg.shape[1]
        _log("Preparing IHC registration image ...")
        ref_gray = _ihc_to_grayscale(ihc_reg_ch, invert=ihc_invert)
        del ihc_reg_ch

    else:
        _log(f"Loading H&E at {maldi_pixel_um} um/px ...")
        he_img, loaded_mpp = _load_he_at_resolution(he_path, maldi_pixel_um, he_pixel_um)
        _log(f"  H&E: {he_img.width}x{he_img.height}  ({he_img.width*he_img.height*3/1e6:.0f} MB)")
        ref_reg_w = he_img.width
        ref_reg_h = he_img.height
        _log("Preparing H&E registration image ...")
        ref_gray = _he_to_grayscale(he_img)

    # ------------------------------------------------------------------
    # 7. Registration
    # ------------------------------------------------------------------
    _log("Running registration ...")
    best_rot, best_idx = _register(
        ref_gray, maldi_gray,
        src_w=ref_reg_w,
        src_h=ref_reg_h,
        coarse_step=coarse_rotation_step,
        fine_range=fine_rotation_range,
        fine_step=fine_rotation_step,
        buffer_px=buffer_px,
    )
    del ref_gray, maldi_gray
    gc.collect()

    # ------------------------------------------------------------------
    # 8. Build output canvas
    # ------------------------------------------------------------------
    _log("Building output canvas ...")

    if use_ihc:
        ihc_canvas_reg, M_stored, canvas_pr, canvas_pc = _build_ihc_canvas(
            ihc_full=ihc_full_reg,
            src_w=ref_reg_w,
            src_h=ref_reg_h,
            rotation_deg=best_rot,
            buffer_px=buffer_px,
        )
        del ihc_full_reg
        gc.collect()

        # _build_spatialdata always expects an (H, W, 3) uint8 H&E canvas at
        # registration resolution for its coordinate and upscaling calculations.
        # Build a minimal greyscale RGB dummy from the first IHC channel so all
        # coordinate maths stays identical to the H&E path.
        first_ch = ihc_canvas_reg[0]   # (H_canvas, W_canvas) uint8, reg-res
        he_canvas_dummy = np.stack([first_ch, first_ch, first_ch], axis=-1)  # (H,W,3)

    else:
        he_canvas, M_stored, canvas_pr, canvas_pc = _build_affine_and_canvas(
            he_img       = he_img,
            src_w        = ref_reg_w,
            src_h        = ref_reg_h,
            rotation_deg = best_rot,
            buffer_px    = buffer_px,
        )
        del he_img
        gc.collect()
        ihc_canvas_up = None
        he_canvas_dummy = he_canvas

    # ------------------------------------------------------------------
    # 9. Annotations
    # ------------------------------------------------------------------
    annotation_gdf = None
    if geojson_path is not None:
        _log(f"Transforming annotations: {geojson_path} ...")
        annotation_gdf = _transform_geojson(
            geojson_path       = geojson_path,
            he_pixel_um        = ref_pixel_um,
            reg_mpp            = loaded_mpp,
            M_stored           = M_stored,
            img_upscaling      = img_upscaling,
            classification_key = geojson_classification_key,
        )
        unique = annotation_gdf[geojson_classification_key].unique().tolist()
        _log(f"  {len(annotation_gdf)} annotations  |  classes: {unique}")

    # ------------------------------------------------------------------
    # 10. Assemble SpatialData
    # ------------------------------------------------------------------
    _log("Building SpatialData ...")
    sdata = _build_spatialdata(
        spectra_all            = spectra_all,
        peaks                  = peaks,
        maldi_pixel_um         = maldi_pixel_um,
        he_canvas              = he_canvas_dummy,
        maldi_offset_in_canvas = best_idx,
        reg_mpp                = loaded_mpp,
        crop_r                 = crop_r,
        crop_c                 = crop_c,
        img_upscaling          = img_upscaling,
        ihc_canvas             = ihc_canvas_reg if use_ihc else None,
        ihc_channel_names      = ihc_ch_names,
    )

    if annotation_gdf is not None:
        ann_shapes = ShapesModel.parse(
            annotation_gdf,
            transformations={"global": Identity()},
        )
        ann_shapes[geojson_classification_key] = ann_shapes[geojson_classification_key].astype("category")
        sdata.shapes[geojson_shapes_key] = ann_shapes
        _log(f"  Annotations added -> sdata.shapes['{geojson_shapes_key}']")

    # ------------------------------------------------------------------
    # 11. Store registration transform
    # ------------------------------------------------------------------
    transform_meta = {
        "rotation_deg":     float(best_rot),
        "maldi_offset":     [int(best_idx[0]), int(best_idx[1])],
        "maldi_pixel_um":   float(maldi_pixel_um),
        "reg_mpp":          float(loaded_mpp),
        "buffer_px":        int(buffer_px),
        "img_upscaling":    int(img_upscaling),
        "canvas_shape":     list(he_canvas_dummy.shape[:2]),
        "he_reg_size":      [int(ref_reg_h), int(ref_reg_w)],
        "canvas_placement": [int(canvas_pr), int(canvas_pc)],
        "affine_matrix":    M_stored.tolist(),
    }

    if use_ihc:
        transform_meta["mode"]          = "ihc"
        transform_meta["ihc_pixel_um"]  = float(ihc_pixel_um)
        transform_meta["ihc_channel"]   = ihc_channel
        transform_meta["ihc_invert"]    = ihc_invert
        transform_meta["ihc_channels"]  = ihc_ch_names
    else:
        transform_meta["mode"]          = "he"
        transform_meta["he_pixel_um"]   = float(he_pixel_um)

    sdata["maldi_adata"].uns["he_transform"] = transform_meta

    sdata["maldi_adata"].uns["maldi_path"] = imzml_path

    _log(
        f"  Transform stored: mode={'ihc' if use_ihc else 'he'}  "
        f"rotation={best_rot}  "
        f"ref_reg_size={[ref_reg_h, ref_reg_w]}  "
        f"canvas_placement={[canvas_pr, canvas_pc]}"
    )

    _log("Done.")
    return sdata