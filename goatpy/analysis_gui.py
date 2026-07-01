from __future__ import annotations

import warnings
import threading
import uuid
from typing import Optional, Callable

import numpy as np
import pandas as pd

# ── Qt ───────────────────────────────────────────────────────────────────────
from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox,
    QPushButton, QGroupBox, QSizePolicy, QScrollArea, QTabWidget,
    QCheckBox, QSpinBox, QDoubleSpinBox, QApplication, QProgressBar,
    QDialog, QDialogButtonBox, QFileDialog, QListWidget, QListWidgetItem,
    QPlainTextEdit, QTableWidget, QTableWidgetItem, QColorDialog,
    QSlider,
)
from qtpy.QtCore import Qt, Signal, QTimer, QThread, QObject
from qtpy.QtGui import QCursor, QColor, QPixmap

# ── matplotlib ───────────────────────────────────────────────────────────────
import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.backend_bases import MouseButton
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D

# ── napari ───────────────────────────────────────────────────────────────────
import napari
from napari.utils.notifications import show_info

# ── spatialdata ──────────────────────────────────────────────────────────────
from spatialdata import SpatialData


# ════════════════════════════════════════════════════════════════════════════
# Colour palette
# ════════════════════════════════════════════════════════════════════════════

PALETTE = {
    "bg":          "#1e1e2e",
    "surface":     "#2a2a3e",
    "border":      "#3d3d5c",
    "accent":      "#7c6af7",
    "accent2":     "#f5a623",
    "text":        "#cdd6f4",
    "text_dim":    "#6c7086",
    "spectrum":    "#7c6af7",
    "raw_spec":    "#4a5580",       # dimmer colour for the dense raw spectrum
    "peak_marker": "#f38ba8",
    "highlight":   "#fab387",
    "success":     "#a6e3a1",
}

_GOATPY_REFS: dict = {}

CONTINUOUS_CMAPS = [
    "inferno", "magma", "plasma", "viridis",
    "hot", "RdBu_r", "coolwarm", "turbo", "gray",
]
CATEGORICAL_CMAPS = [
    "tab10", "tab20", "Set1", "Set2", "Set3", "Paired", "hsv", "Pastel1",
]


BASE_STYLE = f"""
    QWidget {{
        background-color: {PALETTE['bg']};
        color: {PALETTE['text']};
        font-family: 'Segoe UI', 'SF Pro Display', sans-serif;
        font-size: 11px;
    }}
    QGroupBox {{
        background-color: {PALETTE['surface']};
        border: 1px solid {PALETTE['border']};
        border-radius: 6px;
        margin-top: 8px;
        padding-top: 8px;
        font-weight: bold;
    }}
    QGroupBox::title {{
        subcontrol-origin: margin;
        left: 8px;
        padding: 0 4px;
        color: {PALETTE['accent']};
    }}
    QComboBox {{
        background-color: {PALETTE['surface']};
        border: 1px solid {PALETTE['border']};
        border-radius: 4px;
        padding: 4px 8px;
        color: {PALETTE['text']};
    }}
    QComboBox::drop-down {{ border: none; }}
    QComboBox QAbstractItemView {{
        background-color: {PALETTE['surface']};
        color: {PALETTE['text']};
        selection-background-color: {PALETTE['accent']};
    }}
    QPushButton {{
        background-color: {PALETTE['accent']};
        color: white;
        border: none;
        border-radius: 4px;
        padding: 6px 12px;
        font-weight: bold;
    }}
    QPushButton:hover {{ background-color: #9b8df8; }}
    QPushButton:pressed {{ background-color: #5d4ed6; }}
    QPushButton:disabled {{
        background-color: {PALETTE['border']};
        color: {PALETTE['text_dim']};
    }}
    QLabel {{ color: {PALETTE['text']}; }}
    QTabWidget::pane {{
        border: 1px solid {PALETTE['border']};
        background-color: {PALETTE['surface']};
        border-radius: 4px;
    }}
    QTabBar::tab {{
        background-color: {PALETTE['bg']};
        color: {PALETTE['text_dim']};
        padding: 6px 14px;
        border: 1px solid {PALETTE['border']};
        border-bottom: none;
        border-top-left-radius: 4px;
        border-top-right-radius: 4px;
    }}
    QTabBar::tab:selected {{
        background-color: {PALETTE['surface']};
        color: {PALETTE['text']};
    }}
    QCheckBox {{ color: {PALETTE['text']}; spacing: 6px; }}
    QScrollArea {{ border: none; background-color: transparent; }}
    QSpinBox, QDoubleSpinBox {{
        background-color: {PALETTE['surface']};
        border: 1px solid {PALETTE['border']};
        border-radius: 4px;
        padding: 3px 6px;
        color: {PALETTE['text']};
    }}
    QProgressBar {{
        background-color: {PALETTE['surface']};
        border: 1px solid {PALETTE['border']};
        border-radius: 3px;
        text-align: center;
        color: {PALETTE['text']};
    }}
    QProgressBar::chunk {{
        background-color: {PALETTE['accent']};
        border-radius: 3px;
    }}
"""


# ════════════════════════════════════════════════════════════════════════════
# SpatialData → Napari loader  (FIXED — uses napari-spatialdata properly)
# ════════════════════════════════════════════════════════════════════════════

def _resolve_element_cs(sdata, element_type: str, name: str, preferred_cs: str = "aligned") -> str:
    """Pick the best coordinate system for an element (prefer aligned, else global)."""
    from spatialdata.transformations import get_transformation

    element = getattr(sdata, element_type)[name]
    try:
        transforms = get_transformation(element, get_all=True)
        cs_names = list(transforms.keys()) if isinstance(transforms, dict) else []
    except Exception:
        cs_names = []

    if preferred_cs in cs_names:
        return preferred_cs
    if "global" in cs_names:
        return "global"
    if cs_names:
        return cs_names[0]
    return preferred_cs


def _add_spatialdata_layers(viewer, sdata, target_cs="aligned"):
    """
    Load every SpatialData element into the viewer via napari-spatialdata.
    Each element is added in its own best coordinate system so shapes like
    'pixels' and 'Annotations' are all available even when they differ.
    """
    from napari_spatialdata import Interactive

    interactive = Interactive(sdata, headless=True)

    for element_type in ("images", "labels", "shapes", "points"):
        container = getattr(sdata, element_type, {})
        for name in container:
            cs = _resolve_element_cs(sdata, element_type, name, target_cs)
            try:
                interactive.add_element(
                    element=name,
                    element_coordinate_system=cs,
                    view_element_system=False,
                )
                print(f"[goatpy GUI] Added '{name}' ({element_type}) in '{cs}'")
            except Exception as e:
                print(f"[goatpy GUI] Could not add '{name}' ({element_type}): {e}")

    try:
        interactive.switch_coordinate_system(target_cs)
    except Exception as e:
        print(f"[goatpy GUI] Could not switch coordinate system: {e}")

    return interactive



# ════════════════════════════════════════════════════════════════════════════
# m/z resolution helpers
# ════════════════════════════════════════════════════════════════════════════

def _resolve_mz_array(adata) -> np.ndarray:
    """
    Return float64 m/z array for every adata column, handling:
      1. adata.var["mz_original"]  (set by annotate_glycans)
      2. "mz-933.4" prefixed var_names
      3. plain numeric var_names
      4. column index fallback
    """
    n = adata.n_vars
    if "mz_original" in adata.var.columns:
        try:
            arr = pd.to_numeric(adata.var["mz_original"], errors="coerce").values
            if not np.any(np.isnan(arr)):
                return arr.astype(np.float64)
        except Exception:
            pass

    mzs = np.full(n, np.nan, dtype=np.float64)
    for i, vn in enumerate(adata.var_names):
        s = str(vn).strip()
        if s.lower().startswith("mz-"):
            s = s[3:]
        try:
            mzs[i] = float(s)
        except ValueError:
            pass
    nan_mask = np.isnan(mzs)
    if nan_mask.any():
        mzs[nan_mask] = np.where(nan_mask)[0].astype(np.float64)
    return mzs

def _load_glycan_reference_table() -> pd.DataFrame:
    """
    Load the curated theoretical-mass reference table bundled with goatpy
    (goatpy/data/glycan_list.csv). Columns are normalised to 'mz' (theoretical
    m/z [M+Na]) and 'label' (composition / glycan name).
    """
    df = None

    # Preferred: package data access (works when goatpy is installed)
    try:
        from importlib.resources import files
        path = files("goatpy").joinpath("data", "glycan_list.csv")
        df = pd.read_csv(path)
    except Exception:
        df = None

    # Fallback: locate relative to this source file (dev / editable installs)
    if df is None:
        try:
            import pathlib
            here = pathlib.Path(__file__).resolve()
            for parent in here.parents:
                candidate = parent / "data" / "glycan_list.csv"
                if candidate.exists():
                    df = pd.read_csv(candidate)
                    break
        except Exception:
            df = None

    if df is None:
        print("[goatpy GUI] Could not locate glycan_list.csv — "
              "theoretical-mass lookup will be unavailable.")
        return pd.DataFrame(columns=["mz", "label"])

    df = df.rename(columns={
        "Theoretical m/z [M+Na]": "mz",
        "Composition": "label",
    })
    df["mz"] = pd.to_numeric(df["mz"], errors="coerce")
    df["label"] = df["label"].astype(str).str.strip()
    return df.dropna(subset=["mz"]).reset_index(drop=True)

def _resolve_var_display_labels(adata) -> list[str]:
    mzs = _resolve_mz_array(adata)
    labels = []
    for i, vn in enumerate(adata.var_names):
        s = str(vn).strip()
        labels.append(f"{mzs[i]:.4f}" if s.lower().startswith("mz-") else s)
    return labels


def _looks_numeric(s: str) -> bool:
    s2 = s.strip()
    if s2.lower().startswith("mz-"):
        s2 = s2[3:]
    try:
        float(s2)
        return True
    except ValueError:
        return False


# ════════════════════════════════════════════════════════════════════════════
# Napari layer helper — render glycan ion image ON the H&E
# ════════════════════════════════════════════════════════════════════════════

GLYCAN_LAYER_NAME = "glycan_ion_map"
GOATPY_VIZ_KEY = "_goatpy_viz"


def _find_shapes_layer(viewer, shapes_name: str):
    """Match a napari Shapes layer by element name (handles napari-spatialdata suffixes)."""
    from napari.layers import Shapes

    for lyr in viewer.layers:
        if not isinstance(lyr, Shapes):
            continue
        n = lyr.name
        if (n == shapes_name
                or n.startswith(shapes_name + " [")
                or n.startswith(shapes_name + ":")
                or shapes_name in n):
            return lyr
    return None


def _is_obs_categorical(adata, col: str) -> bool:
    series = adata.obs[col]
    if series.dtype.name == "category" or series.dtype == bool:
        return True
    n_unique = series.nunique(dropna=True)
    if pd.api.types.is_integer_dtype(series) and n_unique <= 64:
        return True
    if pd.api.types.is_numeric_dtype(series):
        return n_unique <= min(16, max(1, adata.n_obs // 20))
    return n_unique <= min(50, max(1, adata.n_obs // 20))


def _is_series_categorical(series) -> bool:
    if series.dtype.name == "category" or series.dtype == bool:
        return True
    n_unique = pd.Series(series).nunique(dropna=True)
    if pd.api.types.is_integer_dtype(series) and n_unique <= 64:
        return True
    if pd.api.types.is_numeric_dtype(series):
        return n_unique <= min(16, max(1, len(series) // 20))
    return n_unique <= min(50, max(1, len(series) // 20))


def _shape_to_dataframe(shapes):
    if shapes is None:
        return None
    if hasattr(shapes, "columns"):
        return shapes
    if hasattr(shapes, "to_dataframe"):
        return shapes.to_dataframe()
    if hasattr(shapes, "compute"):
        return shapes.compute()
    return None



def _normalize_shapes_rgba(colours):
    arr = np.asarray(colours, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] < 3:
        return np.zeros((len(colours), 4), dtype=np.float32)
    if arr.max() > 1.1:
        arr = arr / 255.0
    if arr.shape[1] == 3:
        alpha = np.ones((arr.shape[0], 1), dtype=arr.dtype)
        arr = np.concatenate([arr, alpha], axis=1)
    return arr

def _rgb_to_hex(rgb) -> str:
    """Convert [r,g,b] (0-255 or 0-1) to '#rrggbb'."""
    arr = np.asarray(rgb, dtype=np.float64).ravel()[:3]
    if arr.max() <= 1.0:
        arr = arr * 255.0
    arr = np.clip(arr, 0, 255).astype(int)
    return "#{:02x}{:02x}{:02x}".format(arr[0], arr[1], arr[2])


def _hex_to_rgb(hex_str: str) -> list[int]:
    """Convert '#rrggbb' (or 'rrggbb') to [r, g, b] ints (0-255)."""
    s = str(hex_str).lstrip("#")
    if len(s) != 6:
        return [128, 128, 128]
    try:
        return [int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16)]
    except ValueError:
        return [128, 128, 128]


def _safe_is_series_categorical(series) -> bool:
    """Like _is_series_categorical but tolerant of unhashable / malformed columns."""
    try:
        # Reject columns whose values aren't hashable (e.g. lists/arrays of RGB)
        sample = series.iloc[0] if len(series) else None
        if isinstance(sample, (list, tuple, np.ndarray)):
            return False
        return _is_series_categorical(series)
    except TypeError:
        return False
    except Exception:
        return False


def _apply_direct_shapes_colors(layer, colours):
    rgba = _normalize_shapes_rgba(colours)
    layer.face_color = rgba
    layer.face_color_mode = "direct"
    layer.edge_color = "transparent"
    layer.edge_width = 0.0
    layer.refresh_colors(update_color_mapping=True)
    layer.refresh()


def _categorical_cycle_colors(colormap: str, n: int) -> np.ndarray:
    """Build an RGBA color cycle from a matplotlib categorical colormap."""
    cmap = plt.get_cmap(colormap)
    if hasattr(cmap, "colors") and cmap.colors:
        base = np.array(cmap.colors)
        if len(base) < n:
            base = np.tile(base, (int(np.ceil(n / len(base))), 1))[:n]
        else:
            base = base[:n]
    else:
        if n <= 0:
            return np.zeros((0, 4), dtype=float)
        base = np.array([cmap(i / max(n - 1, 1)) for i in range(n)])
    return base


def _apply_shapes_colormap(layer, colormap: str, categorical: bool, n_categories: int = 0):
    """Apply colormap / color-cycle to an already-mapped shapes layer and refresh."""
    if categorical:
        n = max(n_categories, 1)
        layer.face_color_cycle = _categorical_cycle_colors(colormap, n)
        layer.face_color_mode = "cycle"
    else:
        layer.face_colormap = colormap
        layer.face_color_mode = "colormap"
    layer.refresh_colors(update_color_mapping=True)
    layer.refresh()


def _draw_spatial_legend(
    canvas: MplCanvas,
    title: str,
    *,
    categorical: bool,
    colormap: str,
    categories: Optional[list] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
):
    """Draw a matplotlib legend (categorical) or colorbar (continuous) in the sidebar."""
    canvas.fig.clear()
    ax = canvas.fig.add_subplot(111)
    canvas._style_ax(ax)
    ax.set_axis_off()

    if categorical and categories:
        colors = _categorical_cycle_colors(colormap, len(categories))
        handles = [
            mpatches.Patch(
                facecolor=tuple(np.asarray(colors[i]).ravel()),
                edgecolor=PALETTE["border"],
                label=str(cat),
            )
            for i, cat in enumerate(categories)
        ]
        ax.legend(
            handles=handles, title=title, loc="center",
            fontsize=7, title_fontsize=8, framealpha=0.35,
            facecolor=PALETTE["surface"], edgecolor=PALETTE["border"],
            labelcolor=PALETTE["text"],
        )
    elif vmin is not None and vmax is not None:
        sm = plt.cm.ScalarMappable(
            cmap=colormap, norm=plt.Normalize(vmin=vmin, vmax=vmax),
        )
        sm.set_array([])
        cb = canvas.fig.colorbar(
            sm, ax=ax, fraction=0.85, pad=0.02,
            orientation="horizontal",
        )
        cb.ax.xaxis.set_tick_params(color=PALETTE["text_dim"], labelsize=7)
        cb.outline.set_edgecolor(PALETTE["border"])
        plt.setp(cb.ax.xaxis.get_ticklabels(), color=PALETTE["text"])
        cb.set_label(title, color=PALETTE["text"], fontsize=8)
    else:
        ax.text(0.5, 0.5, "No legend", ha="center", va="center",
                transform=ax.transAxes, color=PALETTE["text_dim"], fontsize=8)

    canvas.fig.tight_layout(pad=0.3)
    canvas.draw()


def _render_values_on_shapes(
    viewer,
    values,
    colormap: str,
    categorical: bool,
    label: str,
    shapes_name: str = "pixels",
) -> Optional[dict]:
    """
    Colour a shapes layer by per-instance values.
    Returns render metadata for legend updates, or None on failure.
    """
    layer = _find_shapes_layer(viewer, shapes_name)
    if layer is None:
        from napari.layers import Shapes
        available = [l.name for l in viewer.layers if isinstance(l, Shapes)]
        show_info(
            f"No Shapes layer matching '{shapes_name}' found. "
            f"Available Shapes layers: {available}"
        )
        return None

    values = np.asarray(values).ravel()
    if len(layer.data) != len(values):
        show_info(
            f"Shape count ({len(layer.data)}) ≠ value count ({len(values)}). "
            "Check obs ordering / region linkage."
        )
        return None

    if categorical:
        props = np.asarray(values, dtype=object).astype(str)
        categories = list(np.unique(props))
        n_categories = len(categories)
        vmin = vmax = None
    else:
        props = pd.to_numeric(values, errors="coerce").astype(np.float32)
        categories = None
        n_categories = 0
        valid = props[np.isfinite(props)]
        if len(valid):
            vmin = float(np.percentile(valid, 1))
            vmax = float(np.percentile(valid, 99))
        else:
            vmin, vmax = 0.0, 1.0

    # Single internal property key avoids stale napari color-cycle / colormap caches.
    layer.properties = {GOATPY_VIZ_KEY: props}
    layer.opacity = 0.75

    # Set colormapping parameters first, before assigning the face_color property
    if categorical:
        layer.face_color_mode = "cycle"
        layer.face_color_cycle = _categorical_cycle_colors(colormap, n_categories)
        layer.face_contrast_limits = None
    else:
        layer.face_color_mode = "colormap"
        layer.face_colormap = colormap
        layer.face_contrast_limits = (vmin, vmax)

    # Now assign the property name for coloring
    layer.face_color = GOATPY_VIZ_KEY
    
    # Ensure edge color doesn't obscure the face coloring
    layer.edge_color = "transparent"
    layer.edge_width = 0.0
    
    layer.refresh_colors(update_color_mapping=True)
    layer.refresh()

    show_info(f"Layer updated: {label}")
    return {
        "categorical": categorical,
        "colormap": colormap,
        "categories": categories,
        "vmin": vmin,
        "vmax": vmax,
        "shapes_name": shapes_name,
        "label": label,
    }


# ════════════════════════════════════════════════════════════════════════════
# Ion-image helper — compute per-pixel intensities for an arbitrary m/z and
# render them on the existing "pixels" Shapes layer (guaranteed alignment
# with the H&E, since it reuses the same registered shapes/transform).
# ════════════════════════════════════════════════════════════════════════════

def _compute_unregistered_ion_values(sdata, imzml_path: str, target_mz: float,
                                      tol_da: float, table_name: str = "maldi_adata"):
    """
    For an arbitrary m/z (+/- tol_da), compute a per-pixel intensity value
    ordered to match adata.obs / the "pixels" shapes layer.

    Mirrors the coordinate convention used in glyco_spatialdata():
        full_x, full_y = p.coordinates[:, :2] - 1   (0-based)
        x = full_x - full_x.min(); y = full_y - full_y.min()
    Each spectrum index i then corresponds to obs row with the same (x, y).
    """
    from pyimzml.ImzMLParser import ImzMLParser

    p = ImzMLParser(imzml_path)
    coords = np.array(p.coordinates)[:, :2].astype(np.int64) - 1  # 0-based (full_x, full_y)
    x = coords[:, 0] - coords[:, 0].min()
    y = coords[:, 1] - coords[:, 1].min()

    # Per-spectrum summed intensity within [target_mz - tol, target_mz + tol]
    n = len(p.coordinates)
    per_spectrum = np.zeros(n, dtype=np.float64)
    lo, hi = target_mz - tol_da, target_mz + tol_da
    for i in range(n):
        mzs, ints = p.getspectrum(i)
        mzs = np.asarray(mzs)
        ints = np.asarray(ints, dtype=np.float64)
        mask = (mzs >= lo) & (mzs <= hi)
        if mask.any():
            per_spectrum[i] = ints[mask].sum()

    # Map (x, y) -> intensity
    xy_to_val: dict[tuple[int, int], float] = {}
    for i in range(n):
        xy_to_val[(int(x[i]), int(y[i]))] = per_spectrum[i]

    adata = sdata.tables[table_name]
    obs_x = adata.obs["x"].to_numpy(dtype=np.int64)
    obs_y = adata.obs["y"].to_numpy(dtype=np.int64)

    values = np.array(
        [xy_to_val.get((int(xi), int(yi)), 0.0) for xi, yi in zip(obs_x, obs_y)],
        dtype=np.float32,
    )
    return values


def _display_unregistered_ion_image_on_shapes(
    viewer,
    sdata,
    imzml_path: str,
    target_mz: float,
    tol_da: float,
    label: str,
    table_name: str = "maldi_adata",
    shapes_name: str = "pixels",
    colormap: str = "inferno",
):
    """
    Compute per-pixel intensities for an arbitrary m/z (+/- tol) and render
    them on the existing 'pixels' Shapes layer — same mechanism used for
    curated glycan ion maps, so alignment with the H&E is guaranteed.
    """
    try:
        values = _compute_unregistered_ion_values(
            sdata, imzml_path, target_mz, tol_da, table_name=table_name
        )
    except Exception as e:
        show_info(f"Could not compute ion image: {e}")
        return None

    state = _render_values_on_shapes(
        viewer, values, colormap, False,
        f"{label} (\u00b1{tol_da:.4f})", shapes_name,
    )
    if state is not None:
        show_info(f"Ion image displayed for {label} (m/z {target_mz:.4f} \u00b1 {tol_da:.4f})")
    return state


def _render_glycan_on_viewer(
    viewer,
    sdata,
    peak_mz,
    label,
    table_name="maldi_adata",
    shapes_name="pixels",
    colormap: str = "inferno",
):
    adata = sdata.tables[table_name]
    var_mzs = _resolve_mz_array(adata)
    idx = int(np.argmin(np.abs(var_mzs - peak_mz)))

    if abs(var_mzs[idx] - peak_mz) > 0.5:
        show_info(f"Peak {peak_mz:.2f} not found")
        return

    values = np.asarray(adata.X[:, idx]).astype(np.float32).ravel()
    return _render_values_on_shapes(
        viewer, values, colormap, False,
        f"{label} ({peak_mz:.2f})", shapes_name,
    )



class _SpectrumLoader(QObject):
    """
    Loads the mean spectrum from the raw imzML file on a worker thread,
    then emits finished(mz_array, intensity_array).
    """
    finished = Signal(object, object)   # np.ndarray, np.ndarray
    progress = Signal(int)              # 0-100

    def __init__(self, imzml_path: str, n_sample: int = 500):
        super().__init__()
        self.imzml_path = imzml_path
        self.n_sample = n_sample        # max spectra to average (for speed)

    def run(self):
        try:
            from pyimzml.ImzMLParser import ImzMLParser
            p = ImzMLParser(self.imzml_path)
            n_total = len(p.coordinates)
            step = max(1, n_total // self.n_sample)
            indices = list(range(0, n_total, step))

            # First pass: discover global m/z range
            all_mzs = []
            for i, idx in enumerate(indices):
                mzs, _ = p.getspectrum(idx)
                all_mzs.append(mzs)
                if i % max(1, len(indices) // 20) == 0:
                    self.progress.emit(int(i / len(indices) * 50))

            lo = min(a[0] for a in all_mzs if len(a))
            hi = max(a[-1] for a in all_mzs if len(a))
            # Build uniform 0.05 Da grid
            bin_edges = np.arange(lo, hi + 0.05, 0.05)
            acc = np.zeros(len(bin_edges) - 1, dtype=np.float64)
            counts = np.zeros(len(bin_edges) - 1, dtype=np.int32)

            for i, (idx, mzs) in enumerate(zip(indices, all_mzs)):
                _, ints = p.getspectrum(idx)
                if len(mzs) == 0:
                    continue
                bin_idx = np.searchsorted(bin_edges, mzs, side="right") - 1
                valid = (bin_idx >= 0) & (bin_idx < len(acc))
                np.add.at(acc, bin_idx[valid], np.asarray(ints, dtype=np.float64)[valid])
                np.add.at(counts, bin_idx[valid], 1)
                if i % max(1, len(indices) // 20) == 0:
                    self.progress.emit(50 + int(i / len(indices) * 50))

            with np.errstate(divide="ignore", invalid="ignore"):
                mean_spec = np.where(counts > 0, acc / counts, 0.0)

            bin_centres = (bin_edges[:-1] + bin_edges[1:]) / 2.0
            self.progress.emit(100)
            self.finished.emit(bin_centres, mean_spec)

        except Exception as e:
            print(f"[SpectrumLoader] Error: {e}")
            self.finished.emit(np.array([]), np.array([]))


# ════════════════════════════════════════════════════════════════════════════
# MplCanvas
# ════════════════════════════════════════════════════════════════════════════

class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=8, height=3, dpi=90):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.fig.patch.set_facecolor(PALETTE["surface"])
        super().__init__(self.fig)
        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumHeight(80)   # was 180 — too tall for 13" MacBooks

    def _style_ax(self, ax):
        ax.set_facecolor(PALETTE["bg"])
        for spine in ax.spines.values():
            spine.set_color(PALETTE["border"])
        ax.tick_params(colors=PALETTE["text_dim"], labelsize=8)
        ax.xaxis.label.set_color(PALETTE["text"])
        ax.yaxis.label.set_color(PALETTE["text"])
        ax.title.set_color(PALETTE["text"])
        return ax


# ════════════════════════════════════════════════════════════════════════════
# 1. SPECTRUM WIDGET  (bottom dock)
#    • Loads full raw spectrum from imzML in background
#    • Scroll = pan, Ctrl+scroll = zoom
#    • Click near a red peak line → selects that glycan
# ════════════════════════════════════════════════════════════════════════════

class SpectrumWidget(QWidget):
    """Interactive spectrum panel — full imzML background + clickable peaks."""

    # Emitted when user clicks a peak; consumed by AnalysisSidebar + viewer
    peak_clicked = Signal(float, str)   # mz, display_label

    # Emitted when user requests "Display spatially" for an unregistered peak
    unregistered_peak_display = Signal(float, float)   # mz, tolerance (Da)

    def __init__(
        self,
        sdata: SpatialData,
        peaks: list[float],
        glycan_df: Optional[pd.DataFrame] = None,
        table_name: str = "maldi_adata",
        applied_tolerance: float = 0.1,
        parent=None,
    ):
        super().__init__(parent)
        self.sdata = sdata
        self.peaks = sorted(peaks)
        self.glycan_df = glycan_df
        self.table_name = table_name

        # ── Tolerance originally applied when peaks were extracted into the
        #    raw SpatialData object (glyco_spatialdata's `tol` argument) ──
        self.applied_tolerance: float = applied_tolerance
        try:
            uns_tol = self.sdata.tables[self.table_name].uns.get("maldi_tolerance")
            if uns_tol is not None:
                self.applied_tolerance = float(uns_tol)
        except Exception:
            pass

        self.highlighted_mz: Optional[float] = None
        self.annotated_highlight_mzs: list[float] = []
        self.annotated_highlight_tol: float = 0.5
        self.highlighted_label: str = ""
        self._tol = 0.15

        # ── Unregistered-peak picking mode ──────────────────────────────
        self._unreg_mode: bool = False
        self._unreg_mz: Optional[float] = None
        self._unreg_tol: float = 0.25

        self._updating_scrollbar: bool = False

        # ── m/z list (curated peaks + any added via "Add peak to list") ──
        self.peak_list: list[float] = list(self.peaks)

        # Raw full spectrum (from imzML)
        self._raw_mz: Optional[np.ndarray] = None
        self._raw_int: Optional[np.ndarray] = None

        # sdata mean spectrum (fallback / overlay reference)
        self._sdata_mz: Optional[np.ndarray] = None
        self._sdata_int: Optional[np.ndarray] = None

        # View window
        self._view_lo: float = 500.0
        self._view_hi: float = 3000.0

        # Peak label lookup mz → display string
        self._peak_labels: dict[float, str] = {}
        self._build_peak_labels()

        self._build_ui()
        QTimer.singleShot(100, self._load_sdata_spectrum)
        QTimer.singleShot(200, self._start_imzml_load)

    # ── Peak label lookup ─────────────────────────────────────────────────

    def _build_peak_labels(self):
        """Map each curated m/z to its best glycan name."""
        try:
            adata = self.sdata.tables[self.table_name]
            mz_arr = _resolve_mz_array(adata)
            disp = _resolve_var_display_labels(adata)
            for mz, lbl in zip(mz_arr, disp):
                self._peak_labels[float(mz)] = lbl
        except Exception:
            pass

        if self.glycan_df is not None:
            for _, row in self.glycan_df.iterrows():
                mz = float(row["mz"])
                lbl = str(row["label"])
                if lbl and lbl not in ("nan", ""):
                    self._peak_labels[mz] = lbl

    def _label_for_peak(self, mz: float) -> str:
        best = min(self._peak_labels.keys(), key=lambda m: abs(m - mz), default=None)
        if best is not None and abs(best - mz) < 0.5:
            return self._peak_labels[best]
        return f"{mz:.4f}"

    # ── UI ────────────────────────────────────────────────────────────────

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(2)

        # ── Controls bar ─────────────────────────────────────────────────
        ctrl = QHBoxLayout()

        lbl = QLabel("Spectrum source:")
        lbl.setFixedWidth(110)
        ctrl.addWidget(lbl)
        self.source_combo = QComboBox()
        self.source_combo.addItems(["Raw imzML (full)", "sdata mean"])
        self.source_combo.setFixedWidth(160)
        self.source_combo.currentTextChanged.connect(self._redraw)
        ctrl.addWidget(self.source_combo)

        ctrl.addSpacing(16)
        ctrl.addWidget(QLabel("Show peaks:"))
        self.show_peaks_cb = QCheckBox()
        self.show_peaks_cb.setChecked(True)
        self.show_peaks_cb.stateChanged.connect(self._redraw)
        ctrl.addWidget(self.show_peaks_cb)

        ctrl.addSpacing(16)
        ctrl.addWidget(QLabel("Zoom to:"))
        self.mz_lo = QDoubleSpinBox()
        self.mz_lo.setRange(50, 10000)
        self.mz_lo.setValue(500)
        self.mz_lo.setSingleStep(50)
        self.mz_lo.setFixedWidth(78)
        ctrl.addWidget(self.mz_lo)
        ctrl.addWidget(QLabel("–"))
        self.mz_hi = QDoubleSpinBox()
        self.mz_hi.setRange(50, 10000)
        self.mz_hi.setValue(3000)
        self.mz_hi.setSingleStep(50)
        self.mz_hi.setFixedWidth(78)
        ctrl.addWidget(self.mz_hi)
        zoom_btn = QPushButton("Go")
        zoom_btn.setFixedWidth(40)
        zoom_btn.clicked.connect(self._apply_zoom)
        ctrl.addWidget(zoom_btn)

        reset_btn = QPushButton("Reset")
        reset_btn.setFixedWidth(55)
        reset_btn.clicked.connect(self._reset_zoom)
        ctrl.addWidget(reset_btn)

        ctrl.addSpacing(16)
        self.show_applied_tol_cb = QCheckBox("Show applied tolerance")
        self.show_applied_tol_cb.setChecked(False)
        self.show_applied_tol_cb.stateChanged.connect(self._redraw)
        ctrl.addWidget(self.show_applied_tol_cb)

        self.applied_tol_spin = QDoubleSpinBox()
        self.applied_tol_spin.setRange(0.001, 50.0)
        self.applied_tol_spin.setDecimals(3)
        self.applied_tol_spin.setSingleStep(0.01)
        self.applied_tol_spin.setValue(self.applied_tolerance)
        self.applied_tol_spin.setFixedWidth(78)
        self.applied_tol_spin.valueChanged.connect(self._on_applied_tol_changed)
        ctrl.addWidget(self.applied_tol_spin)

        ctrl.addStretch()
        self.status_lbl = QLabel("Loading…")
        self.status_lbl.setStyleSheet(f"color: {PALETTE['text_dim']};")
        ctrl.addWidget(self.status_lbl)

        layout.addLayout(ctrl)

        # ── Unregistered peak controls ────────────────────────────────────
        unreg = QHBoxLayout()

        self.unreg_btn = QPushButton("Check unregistered peak")
        self.unreg_btn.setCheckable(True)
        self.unreg_btn.toggled.connect(self._on_unreg_toggled)
        unreg.addWidget(self.unreg_btn)

        unreg.addSpacing(12)
        self.unreg_tol_lbl = QLabel("± Tolerance (Da):")
        unreg.addWidget(self.unreg_tol_lbl)
        self.unreg_tol_spin = QDoubleSpinBox()
        self.unreg_tol_spin.setRange(0.001, 50.0)
        self.unreg_tol_spin.setDecimals(3)
        self.unreg_tol_spin.setSingleStep(0.01)
        self.unreg_tol_spin.setValue(self._unreg_tol)
        self.unreg_tol_spin.setFixedWidth(90)
        self.unreg_tol_spin.valueChanged.connect(self._on_unreg_tol_changed)
        unreg.addWidget(self.unreg_tol_spin)

        unreg.addSpacing(12)
        self.unreg_selected_lbl = QLabel("No peak selected")
        self.unreg_selected_lbl.setStyleSheet(f"color: {PALETTE['text_dim']};")
        unreg.addWidget(self.unreg_selected_lbl)

        unreg.addStretch()

        self.display_spatially_btn = QPushButton("Display spatially")
        self.display_spatially_btn.setEnabled(False)
        self.display_spatially_btn.clicked.connect(self._on_display_spatially)
        unreg.addWidget(self.display_spatially_btn)

        self.add_peak_btn = QPushButton("Add peak to list")
        self.add_peak_btn.setEnabled(False)
        self.add_peak_btn.clicked.connect(self._on_add_peak_to_list)
        unreg.addWidget(self.add_peak_btn)

        self.export_peaks_btn = QPushButton("Export list")
        self.export_peaks_btn.clicked.connect(self._on_export_peak_list)
        unreg.addWidget(self.export_peaks_btn)

        layout.addLayout(unreg)

        # Hide unregistered-peak controls until the mode is enabled
        self.unreg_tol_lbl.setVisible(False)
        self.unreg_tol_spin.setVisible(False)
        self.unreg_selected_lbl.setVisible(False)
        self.display_spatially_btn.setVisible(False)
        self.add_peak_btn.setVisible(False)

        # ── Progress bar (shown while imzML loads) ────────────────────────
        self.progress_bar = QProgressBar()
        self.progress_bar.setFixedHeight(4)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.hide()
        layout.addWidget(self.progress_bar)

        # # ── Hint label ────────────────────────────────────────────────────
        # hint = QLabel(
        #     "Scroll to pan  ·  Ctrl+scroll to zoom  ·  Click a red peak line to select glycan  ·  "
        #     "'Check unregistered peak' lets you click anywhere on the spectrum"
        # )
        # hint.setStyleSheet(f"color: {PALETTE['text_dim']}; font-size: 9px;")
        # layout.addWidget(hint)

        # ── Canvas ────────────────────────────────────────────────────────────────
        try:
            _app = QApplication.instance()
            _screen_h = _app.primaryScreen().availableGeometry().height() if _app else 900
        except Exception:
            _screen_h = 900
        _canvas_h_in = min(2.6, max(1.4, (_screen_h * 0.22) / 90))  # dpi=90

        self.canvas = MplCanvas(self, width=10, height=_canvas_h_in, dpi=90)
        layout.addWidget(self.canvas)

        # ── Horizontal scrollbar for panning across the spectrum ──────────
        self._scroll_resolution = 10000
        self.spectrum_scrollbar = QSlider(Qt.Horizontal)
        self.spectrum_scrollbar.setRange(0, self._scroll_resolution)
        self.spectrum_scrollbar.setValue(0)
        self.spectrum_scrollbar.setFixedHeight(14)
        self.spectrum_scrollbar.valueChanged.connect(self._on_scrollbar_changed)
        layout.addWidget(self.spectrum_scrollbar)

        # ── Interactions ──────────────────────────────────────────────────
        self.canvas.mpl_connect("scroll_event",         self._on_scroll)
        self.canvas.mpl_connect("button_press_event",   self._on_click)

    # ── Data loading ──────────────────────────────────────────────────────

    def _load_sdata_spectrum(self):
        try:
            adata = self.sdata.tables[self.table_name]
            mz_arr = _resolve_mz_array(adata)
            X = np.asarray(adata.X, dtype=np.float32)
            intensities = X.mean(axis=0).astype(np.float64)
            self._sdata_mz = mz_arr
            self._sdata_int = intensities
            self._view_lo = float(mz_arr.min())
            self._view_hi = float(mz_arr.max())
            self.mz_lo.setValue(self._view_lo)
            self.mz_hi.setValue(self._view_hi)
            self.status_lbl.setText(
                f"{adata.n_obs:,} pixels · {adata.n_vars:,} peaks  |  "
                "Loading full spectrum…"
            )
        except Exception as e:
            self.status_lbl.setText(f"sdata load error: {e}")
        self._redraw()

    def _start_imzml_load(self):
        try:
            path = self.sdata.tables[self.table_name].uns.get("maldi_path")
        except Exception:
            path = None

        if not path:
            self.status_lbl.setText(
                self.status_lbl.text().replace("Loading full spectrum…", "No imzML path found")
            )
            return

        self.progress_bar.show()
        self.progress_bar.setValue(0)

        self._loader = _SpectrumLoader(path, n_sample=800)
        self._thread = QThread()
        self._loader.moveToThread(self._thread)
        self._thread.started.connect(self._loader.run)
        self._loader.finished.connect(self._on_imzml_loaded)
        self._loader.progress.connect(self.progress_bar.setValue)
        self._loader.finished.connect(self._thread.quit)
        self._thread.start()

    def _on_imzml_loaded(self, mz_arr, int_arr):
        self.progress_bar.hide()
        if len(mz_arr) == 0:
            self.status_lbl.setText("imzML load failed — using sdata mean")
            return

        self._raw_mz = mz_arr
        self._raw_int = int_arr

        # Switch to raw source automatically
        self.source_combo.setCurrentText("Raw imzML (full)")

        # Update view window to full range
        self._view_lo = float(mz_arr.min())
        self._view_hi = float(mz_arr.max())
        self.mz_lo.setValue(self._view_lo)
        self.mz_hi.setValue(self._view_hi)

        try:
            adata = self.sdata.tables[self.table_name]
            self.status_lbl.setText(
                f"{adata.n_obs:,} pixels · {adata.n_vars:,} peaks  |  "
                f"Full spectrum: {len(mz_arr):,} bins"
            )
        except Exception:
            pass

        self._redraw()

    # ── Unregistered peak mode ──────────────────────────────────────────────

    def _on_unreg_toggled(self, checked: bool):
        self._unreg_mode = checked

        self.unreg_tol_lbl.setVisible(checked)
        self.unreg_tol_spin.setVisible(checked)
        self.unreg_selected_lbl.setVisible(checked)
        self.display_spatially_btn.setVisible(checked)
        self.add_peak_btn.setVisible(checked)

        if checked:
            self.unreg_btn.setText("Exit unregistered-peak mode")
            self.unreg_selected_lbl.setText("Click a peak on the spectrum to select it")
            self.display_spatially_btn.setEnabled(False)
            self.add_peak_btn.setEnabled(False)
        else:
            self.unreg_btn.setText("Check unregistered peak")
            self._unreg_mz = None
            self.display_spatially_btn.setEnabled(False)
            self.add_peak_btn.setEnabled(False)

        self._redraw()

    def _on_applied_tol_changed(self, value: float):
        self.applied_tolerance = float(value)
        self._redraw()

    def _on_unreg_tol_changed(self, value: float):
        self._unreg_tol = float(value)
        if self._unreg_mz is not None:
            self.unreg_selected_lbl.setText(
                f"Selected m/z: {self._unreg_mz:.4f}  (± {self._unreg_tol:.3f} Da)"
            )
        self._redraw()

    def _on_display_spatially(self):
        if self._unreg_mz is None:
            return
        self.unregistered_peak_display.emit(self._unreg_mz, self._unreg_tol)

    def _on_add_peak_to_list(self):
        if self._unreg_mz is None:
            return
        mz = float(self._unreg_mz)
        # Avoid near-duplicate entries (within 1e-4 Da)
        if any(abs(mz - existing) < 1e-4 for existing in self.peak_list):
            show_info(f"m/z {mz:.4f} is already in the list.")
            return
        self.peak_list.append(mz)
        self.peak_list.sort()
        show_info(f"Added m/z {mz:.4f} to list (n={len(self.peak_list)}).")

    def _on_export_peak_list(self):
        from qtpy.QtWidgets import QFileDialog

        if not self.peak_list:
            show_info("Peak list is empty — nothing to export.")
            return

        path, _ = QFileDialog.getSaveFileName(
            self, "Export m/z list", "peak_list.csv", "CSV files (*.csv)"
        )
        if not path:
            return

        try:
            df = pd.DataFrame({"m/z": sorted(self.peak_list)})
            df.to_csv(path, index=False)
            show_info(f"Exported {len(self.peak_list)} m/z values to {path}")
        except Exception as e:
            show_info(f"Export failed: {e}")

    # ── Interactions ──────────────────────────────────────────────────────

    def _on_scroll(self, event):
        """
        Plain scroll  → pan (shift view left/right).
        Ctrl+scroll   → zoom in/out around cursor position.
        """
        if self._current_mz() is None:
            return

        span = self._view_hi - self._view_lo
        if span <= 0:
            return

        ctrl_held = (event.key == "control") or (
            QApplication.keyboardModifiers() & Qt.ControlModifier
        )

        if ctrl_held:
            # Zoom: shrink/expand around cursor
            factor = 0.85 if event.button == "up" else 1.0 / 0.85
            cursor_mz = event.xdata if event.xdata is not None else (self._view_lo + self._view_hi) / 2
            new_lo = cursor_mz - (cursor_mz - self._view_lo) * factor
            new_hi = cursor_mz + (self._view_hi - cursor_mz) * factor
        else:
            # Pan: move 15% of span per scroll tick
            shift = span * 0.15 * (-1 if event.button == "up" else 1)
            new_lo = self._view_lo + shift
            new_hi = self._view_hi + shift

        # Clamp to data extent
        mz = self._current_mz()
        data_lo, data_hi = float(mz.min()), float(mz.max())
        width = new_hi - new_lo
        new_lo = max(data_lo, new_lo)
        new_hi = min(data_hi, new_hi)
        if new_hi - new_lo < 5:
            new_lo = new_hi - 5

        self._view_lo = new_lo
        self._view_hi = new_hi
        self.mz_lo.setValue(new_lo)
        self.mz_hi.setValue(new_hi)
        self._redraw()

    def _on_click(self, event):
        """Left-click: find nearest curated peak within a tolerance and select it,
        or — in unregistered-peak mode — select the clicked m/z directly."""
        if event.button != MouseButton.LEFT:
            return
        if event.xdata is None:
            return

        click_mz = float(event.xdata)

        if self._unreg_mode:
            mz = self._current_mz()
            if mz is not None:
                data_lo, data_hi = float(mz.min()), float(mz.max())
                click_mz = max(data_lo, min(data_hi, click_mz))
            self._unreg_mz = click_mz
            self.unreg_selected_lbl.setText(
                f"Selected m/z: {click_mz:.4f}  (± {self._unreg_tol:.3f} Da)"
            )
            self.display_spatially_btn.setEnabled(True)
            self.add_peak_btn.setEnabled(True)
            self._redraw()
            return

        if not self.show_peaks_cb.isChecked():
            return

        span = self._view_hi - self._view_lo
        snap_tol = span * 0.015          # 1.5% of visible window

        # Find nearest curated peak within snap_tol
        visible_peaks = [p for p in self.peaks if self._view_lo <= p <= self._view_hi]
        if not visible_peaks:
            return

        nearest = min(visible_peaks, key=lambda p: abs(p - click_mz))
        if abs(nearest - click_mz) > snap_tol:
            return

        label = self._label_for_peak(nearest)
        self.highlight_glycan(nearest, label)
        self.peak_clicked.emit(nearest, label)

    # ── View helpers ──────────────────────────────────────────────────────

    def _current_mz(self) -> Optional[np.ndarray]:
        src = self.source_combo.currentText()
        if src == "Raw imzML (full)" and self._raw_mz is not None:
            return self._raw_mz
        return self._sdata_mz

    def _on_scrollbar_changed(self, value: int):
        """Pan the view in response to the horizontal scrollbar."""
        if self._updating_scrollbar:
            return

        mz = self._current_mz()
        if mz is None:
            return

        data_lo, data_hi = float(mz.min()), float(mz.max())
        span = self._view_hi - self._view_lo
        if span <= 0:
            return

        max_lo = max(data_lo, data_hi - span)
        frac = value / self._scroll_resolution
        new_lo = data_lo + frac * (max_lo - data_lo)
        new_hi = new_lo + span

        self._view_lo = new_lo
        self._view_hi = new_hi
        self.mz_lo.setValue(new_lo)
        self.mz_hi.setValue(new_hi)
        self._redraw(update_scrollbar=False)

    def _sync_scrollbar(self):
        """Update scrollbar position/handle size to reflect the current view."""
        mz = self._current_mz()
        if mz is None:
            return

        data_lo, data_hi = float(mz.min()), float(mz.max())
        data_span = data_hi - data_lo
        if data_span <= 0:
            return

        view_span = self._view_hi - self._view_lo
        max_lo = max(data_lo, data_hi - view_span)

        # Handle size proportional to visible fraction of the data range
        frac_visible = min(1.0, view_span / data_span)
        page_step = max(1, int(self._scroll_resolution * frac_visible))

        if max_lo > data_lo:
            frac = (self._view_lo - data_lo) / (max_lo - data_lo)
        else:
            frac = 0.0
        value = int(round(frac * self._scroll_resolution))

        self._updating_scrollbar = True
        try:
            self.spectrum_scrollbar.setPageStep(page_step)
            self.spectrum_scrollbar.setValue(max(0, min(self._scroll_resolution, value)))
            self.spectrum_scrollbar.setEnabled(frac_visible < 1.0)
        finally:
            self._updating_scrollbar = False

    def _apply_zoom(self):
        self._view_lo = self.mz_lo.value()
        self._view_hi = self.mz_hi.value()
        self._redraw()

    def _reset_zoom(self):
        mz = self._current_mz()
        if mz is not None:
            self._view_lo = float(mz.min())
            self._view_hi = float(mz.max())
            self.mz_lo.setValue(self._view_lo)
            self.mz_hi.setValue(self._view_hi)
        self._redraw()

    # ── Drawing ──────────────────────────────────────────────────────────

    def _redraw(self, update_scrollbar: bool = True):
        src = self.source_combo.currentText()
        if src == "Raw imzML (full)" and self._raw_mz is not None:
            mz, intensity = self._raw_mz, self._raw_int
        elif self._sdata_mz is not None:
            mz, intensity = self._sdata_mz, self._sdata_int
        else:
            return

        lo, hi = self._view_lo, self._view_hi
        mask = (mz >= lo) & (mz <= hi)
        mz_v = mz[mask]
        int_v = intensity[mask]
        if len(mz_v) == 0:
            return

        self.canvas.fig.clear()
        ax = self.canvas.fig.add_subplot(111)
        self.canvas._style_ax(ax)

        # ── Background spectrum ───────────────────────────────────────────
        spec_col = PALETTE["raw_spec"] if src == "Raw imzML (full)" else PALETTE["spectrum"]
        ax.plot(mz_v, int_v, color=spec_col, linewidth=0.6, alpha=0.8, zorder=2)
        ax.fill_between(mz_v, int_v, color=spec_col, alpha=0.08, zorder=1)
        max_int = float(int_v.max()) if len(int_v) else 1.0

        # ── Curated peaks (red dashed lines) ─────────────────────────────
        if self.show_peaks_cb.isChecked():
            for pk in self.peaks:
                if lo <= pk <= hi:
                    ax.axvline(
                        pk, color=PALETTE["peak_marker"],
                        linewidth=1.0, linestyle="--", alpha=0.7, zorder=3,
                        picker=5,
                    )
        
        # ── Annotated peaks (green) ─────────────────────────────────────
        if self.annotated_highlight_mzs:
            tol = self.annotated_highlight_tol
            for pk in self.annotated_highlight_mzs:
                if (pk - tol) <= hi and (pk + tol) >= lo:
                    ax.axvspan(pk - tol, pk + tol,
                               color=PALETTE["success"], alpha=0.25, zorder=3.5)
                    ax.axvline(pk, color=PALETTE["success"],
                               linewidth=1.3, alpha=0.9, zorder=4)

        # ── Applied tolerance (red outline: baseline -> peak -> baseline) ──
        if self.show_applied_tol_cb.isChecked():
            tol = self.applied_tolerance
            for pk in self.peaks:
                if (pk - tol) <= hi and (pk + tol) >= lo:
                    # Height of the spectrum at this peak's m/z (interpolated)
                    pk_height = float(np.interp(pk, mz_v, int_v)) if len(mz_v) else 0.0

                    x_left, x_right = pk - tol, pk + tol
                    # Smooth rise/fall using a half-cosine profile so the line
                    # leaves/returns to baseline tangentially.
                    n = 30
                    t_up = np.linspace(0, 1, n)
                    x_up = x_left + (pk - x_left) * t_up
                    y_up = pk_height * (1 - np.cos(np.pi * t_up)) / 2

                    t_down = np.linspace(0, 1, n)
                    x_down = pk + (x_right - pk) * t_down
                    y_down = pk_height * (1 + np.cos(np.pi * t_down)) / 2

                    curve_x = np.concatenate([x_up, x_down[1:]])
                    curve_y = np.concatenate([y_up, y_down[1:]])

                    ax.plot(curve_x, curve_y, color="red", linewidth=1.2,
                            alpha=0.85, zorder=6, clip_on=True)

        # ── Highlighted / selected glycan ─────────────────────────────────
        if self.highlighted_mz is not None:
            hmz = self.highlighted_mz
            if lo <= hmz <= hi:
                ax.axvspan(hmz - self._tol, hmz + self._tol,
                           color=PALETTE["highlight"], alpha=0.22, zorder=4)
                ax.axvline(hmz, color=PALETTE["highlight"],
                           linewidth=1.8, linestyle="-", alpha=0.95, zorder=5)
                ax.text(
                    hmz, max_int * 1.01, self.highlighted_label,
                    color=PALETTE["highlight"], fontsize=7.5,
                    ha="center", va="bottom", rotation=90, clip_on=True,
                )

        # ── Unregistered peak selection (cyan band) ────────────────────────
        if self._unreg_mode and self._unreg_mz is not None:
            umz = self._unreg_mz
            if lo <= umz <= hi:
                ax.axvspan(umz - self._unreg_tol, umz + self._unreg_tol,
                           color="#89dceb", alpha=0.25, zorder=4)
                ax.axvline(umz, color="#89dceb",
                           linewidth=1.8, linestyle="-", alpha=0.95, zorder=5)
                ax.text(
                    umz, max_int * 1.01, f"{umz:.4f}",
                    color="#89dceb", fontsize=7.5,
                    ha="center", va="bottom", rotation=90, clip_on=True,
                )

        # ── Legend ────────────────────────────────────────────────────────
        handles = [
            Line2D([0], [0], color=spec_col, linewidth=1.5,
                   label="Full spectrum" if src == "Raw imzML (full)" else "sdata mean"),
        ]
        if self.show_peaks_cb.isChecked():
            handles.append(
                Line2D([0], [0], color=PALETTE["peak_marker"], linewidth=1,
                       linestyle="--", label=f"Curated peaks (n={len(self.peaks)})")
            )
        if self.annotated_highlight_mzs:
            handles.append(
                mpatches.Patch(
                    color=PALETTE["success"], alpha=0.5,
                    label=f"Annotated (n={len(self.annotated_highlight_mzs)}, "
                          f"±{self.annotated_highlight_tol:.3f} Da)",
                )
            )
        if self.highlighted_mz is not None:
            handles.append(
                mpatches.Patch(color=PALETTE["highlight"], alpha=0.5,
                               label=f"Selected: {self.highlighted_label}")
            )
        if self.show_applied_tol_cb.isChecked():
            handles.append(
                Line2D([0], [0], color="red", linewidth=1.2,
                       label=f"Applied tolerance (\u00b1{self.applied_tolerance:.3f} Da)")
            )
        if self._unreg_mode and self._unreg_mz is not None:
            handles.append(
                mpatches.Patch(color="#89dceb", alpha=0.5,
                               label=f"Unregistered: {self._unreg_mz:.4f} ± {self._unreg_tol:.3f}")
            )
        ax.legend(handles=handles, fontsize=7.5, framealpha=0.3,
                  loc="upper right",
                  facecolor=PALETTE["surface"], labelcolor=PALETTE["text"])

        ax.set_xlabel("m/z (Da)", fontsize=9)
        ax.set_ylabel("Intensity", fontsize=9)
        ax.set_xlim(lo, hi)
        ax.set_ylim(bottom=0)
        ax.xaxis.set_minor_locator(mticker.AutoMinorLocator(5))
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        self.canvas.fig.tight_layout(pad=0.4)
        self.canvas.draw()

        if update_scrollbar:
            self._sync_scrollbar()

    # ── Public API ────────────────────────────────────────────────────────

    def highlight_glycan(self, mz: float, label: str, tol: float = 0.15):
        """Highlight a glycan in the spectrum and scroll it into view."""
        self.highlighted_mz = mz
        self.highlighted_label = label
        self._tol = tol

        # Centre the view on this peak, keeping current span or defaulting to 400 Da
        span = max(self._view_hi - self._view_lo, 50)
        half = span / 2
        mz_full = self._current_mz()
        if mz_full is not None:
            data_lo, data_hi = float(mz_full.min()), float(mz_full.max())
            new_lo = max(data_lo, mz - half)
            new_hi = min(data_hi, mz + half)
            self._view_lo = new_lo
            self._view_hi = new_hi
            self.mz_lo.setValue(new_lo)
            self.mz_hi.setValue(new_hi)

        self._redraw()

    def clear_highlight(self):
        self.highlighted_mz = None
        self.highlighted_label = ""
        self._redraw()


    def set_annotated_highlights(self, mzs: list[float], tol: float):
        """Highlight all successfully-annotated peaks in green, ± tol Da."""
        self.annotated_highlight_mzs = list(mzs)
        self.annotated_highlight_tol = float(tol)
        self._redraw()

    def clear_annotated_highlights(self):
        self.annotated_highlight_mzs = []
        self._redraw()

    def update_peak_labels(self, label_map: dict[float, str]):
        """Refresh peak->name lookup after names are edited in the Nomenclature tab."""
        for mz, lbl in label_map.items():
            if lbl:
                self._peak_labels[float(mz)] = lbl
        self._redraw()




# ════════════════════════════════════════════════════════════════════════════
# Glycan selection dialog (heatmap)
# ════════════════════════════════════════════════════════════════════════════

class GlycanSelectionDialog(QDialog):
    """Popup to choose glycans for heatmap plotting via select, type, or upload."""

    def __init__(
        self,
        glycan_names: list[str],
        peaks: list[float],
        label_to_mz: dict[str, float],
        parent=None,
    ):
        super().__init__(parent)
        self.glycan_names = glycan_names
        self.peaks = peaks
        self.label_to_mz = label_to_mz
        self.selected_indices: list[int] = []
        self._upload_path: Optional[str] = None

        self.setWindowTitle("Select Glycans for Heatmap")
        self.setMinimumWidth(360)
        self.setStyleSheet(BASE_STYLE)

        layout = QVBoxLayout(self)

        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)

        # ── Tab 1: checklist ────────────────────────────────────────────
        select_w = QWidget()
        select_layout = QVBoxLayout(select_w)
        select_hint = QLabel("Check one or more glycans:")
        select_hint.setStyleSheet(f"color: {PALETTE['text_dim']}; font-size: 9px;")
        select_layout.addWidget(select_hint)
        self.glycan_list = QListWidget()
        for name in glycan_names:
            item = QListWidgetItem(name)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Unchecked)
            self.glycan_list.addItem(item)
        select_layout.addWidget(self.glycan_list)
        self.tabs.addTab(select_w, "Select")

        # ── Tab 2: comma-separated text ─────────────────────────────────
        type_w = QWidget()
        type_layout = QVBoxLayout(type_w)
        type_hint = QLabel("Enter glycan names or m/z values, separated by commas:")
        type_hint.setStyleSheet(f"color: {PALETTE['text_dim']}; font-size: 9px;")
        type_hint.setWordWrap(True)
        type_layout.addWidget(type_hint)
        self.type_edit = QPlainTextEdit()
        self.type_edit.setPlaceholderText("e.g. HexNAc, 1685.65, Fuc-HexNAc  (933.40)")
        self.type_edit.setMinimumHeight(100)
        type_layout.addWidget(self.type_edit)
        self.tabs.addTab(type_w, "Type")

        # ── Tab 3: file upload ──────────────────────────────────────────
        upload_w = QWidget()
        upload_layout = QVBoxLayout(upload_w)
        upload_hint = QLabel(
            "Upload a .txt or .csv file with one glycan per line, or comma-separated."
        )
        upload_hint.setStyleSheet(f"color: {PALETTE['text_dim']}; font-size: 9px;")
        upload_hint.setWordWrap(True)
        upload_layout.addWidget(upload_hint)
        upload_row = QHBoxLayout()
        self.upload_lbl = QLabel("No file selected")
        self.upload_lbl.setStyleSheet(f"color: {PALETTE['text_dim']};")
        self.upload_lbl.setWordWrap(True)
        upload_row.addWidget(self.upload_lbl, stretch=1)
        browse_btn = QPushButton("Browse…")
        browse_btn.clicked.connect(self._browse_file)
        upload_row.addWidget(browse_btn)
        upload_layout.addLayout(upload_row)
        self.tabs.addTab(upload_w, "Upload")

        self.summary_lbl = QLabel("")
        self.summary_lbl.setStyleSheet(f"color: {PALETTE['text_dim']}; font-size: 9px;")
        layout.addWidget(self.summary_lbl)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        ok_btn = buttons.button(QDialogButtonBox.Ok)
        if ok_btn is not None:
            ok_btn.setText("Enter")
            ok_btn.setDefault(True)
            ok_btn.setAutoDefault(True)

    def _browse_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select glycan list", "",
            "Text files (*.txt *.csv);;All files (*)",
        )
        if path:
            self._upload_path = path
            self.upload_lbl.setText(path)
            self.upload_lbl.setStyleSheet(f"color: {PALETTE['text']};")

    def _tokens_from_active_tab(self) -> list[str]:
        tab = self.tabs.currentIndex()
        if tab == 0:
            return [
                self.glycan_list.item(i).text()
                for i in range(self.glycan_list.count())
                if self.glycan_list.item(i).checkState() == Qt.Checked
            ]
        if tab == 1:
            raw = self.type_edit.toPlainText()
            return [t.strip() for t in raw.replace("\n", ",").split(",") if t.strip()]
        if tab == 2 and self._upload_path:
            try:
                with open(self._upload_path, encoding="utf-8") as fh:
                    content = fh.read()
                tokens = []
                for line in content.splitlines():
                    tokens.extend(t.strip() for t in line.split(",") if t.strip())
                return tokens
            except Exception as e:
                show_info(f"Could not read file: {e}")
        return []

    def _resolve_tokens(self, tokens: list[str]) -> list[int]:
        indices: list[int] = []
        seen: set[int] = set()

        for token in tokens:
            idx = self._match_token(token)
            if idx is not None and idx not in seen:
                indices.append(idx)
                seen.add(idx)

        return indices

    def _match_token(self, token: str) -> Optional[int]:
        token_l = token.strip().lower()

        for i, name in enumerate(self.glycan_names):
            if name.lower() == token_l or token_l in name.lower():
                return i

        if token in self.label_to_mz:
            mz = self.label_to_mz[token]
            for i, pk in enumerate(self.peaks):
                if abs(pk - mz) < 0.5:
                    return i

        try:
            mz_val = float(token_l.replace("mz-", "").strip())
            best_i, best_d = None, float("inf")
            for i, pk in enumerate(self.peaks):
                d = abs(pk - mz_val)
                if d < best_d:
                    best_d, best_i = d, i
            if best_i is not None and best_d < 0.5:
                return best_i
        except ValueError:
            pass

        return None

    def _on_accept(self):
        tokens = self._tokens_from_active_tab()
        if not tokens:
            show_info("No glycans selected. Check items, type a list, or upload a file.")
            return

        self.selected_indices = self._resolve_tokens(tokens)
        if not self.selected_indices:
            show_info("Could not match any glycans. Check names or m/z values.")
            return

        self.summary_lbl.setText(f"{len(self.selected_indices)} glycan(s) selected.")
        self.accept()

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key_Return, Qt.Key_Enter):
            self._on_accept()
        else:
            super().keyPressEvent(event)


# ════════════════════════════════════════════════════════════════════════════
# 2. ANALYSIS SIDEBAR  (right dock)
# ════════════════════════════════════════════════════════════════════════════

class AnalysisSidebar(QWidget):
    """
    Right-dock panel.
    Glycan tab: selector + Violin/Box/Histogram (no spatial scatter — that is
    now rendered on the napari viewer directly).
    """

    glycan_selected = Signal(float, str)   # mz, display_label
    show_annotated_on_spectra = Signal(list, float)   # mzs, tolerance
    nomenclature_updated = Signal(dict)                # mz -> new label

    def __init__(
        self,
        sdata: SpatialData,
        peaks: list[float],
        viewer: napari.Viewer,
        glycan_df: Optional[pd.DataFrame] = None,
        table_name: str = "maldi_adata",
        parent=None,
    ):
        super().__init__(parent)
        self.sdata = sdata
        self.peaks = sorted(peaks)
        self.viewer = viewer
        self.glycan_df = glycan_df
        self.table_name = table_name

        self.hmap_custom_indices: Optional[list[int]] = None
        self._last_glycan_render: Optional[dict] = None
        self._last_meta_render: Optional[dict] = None

        self._build_glycan_lookup()
        self.glycan_reference_df = _load_glycan_reference_table()
        self._build_theoretical_mz_lookup()
        self._build_ui()

    # ── Data helpers ──────────────────────────────────────────────────────

    def _on_cmap_changed(self, cmap_name: str):
        """Re-apply colormap after a glycan render, and refresh the sidebar legend."""
        if not cmap_name or self._last_glycan_render is None:
            return
        layer = _find_shapes_layer(self.viewer, self._last_glycan_render["shapes_name"])
        if layer is None:
            return
        try:
            _apply_shapes_colormap(layer, cmap_name, categorical=False)
            self._last_glycan_render["colormap"] = cmap_name
            self._draw_glycan_legend()
        except Exception as e:
            show_info(f"Could not apply colormap: {e}")

    def _draw_glycan_legend(self):
        state = self._last_glycan_render
        if state is None:
            return
        _draw_spatial_legend(
            self.glycan_legend_canvas, state["label"],
            categorical=False, colormap=state["colormap"],
            vmin=state["vmin"], vmax=state["vmax"],
        )

    def _draw_meta_legend(self):
        state = self._last_meta_render
        if state is None:
            return
        _draw_spatial_legend(
            self.meta_legend_canvas, state["label"],
            categorical=state["categorical"], colormap=state["colormap"],
            categories=state.get("categories"),
            vmin=state.get("vmin"), vmax=state.get("vmax"),
        )

    def _build_glycan_lookup(self):
        self.mz_to_label: dict[float, str] = {}
        self.label_to_mz: dict[str, float] = {}
        self.glycan_names: list[str] = []

        try:
            adata = self.sdata.tables[self.table_name]
            mz_arr = _resolve_mz_array(adata)
            disp = _resolve_var_display_labels(adata)
            for mz, lbl in zip(mz_arr, disp):
                self.mz_to_label[mz] = lbl
                self.label_to_mz[lbl] = mz
        except Exception:
            pass

        if self.glycan_df is not None:
            for _, row in self.glycan_df.iterrows():
                mz = float(row["mz"])
                lbl = str(row["label"])
                if lbl and lbl not in ("nan", ""):
                    self.mz_to_label[mz] = lbl
                    self.label_to_mz[lbl] = mz

        names = []
        for pk in self.peaks:
            best_label = f"{pk:.4f}"
            best_dist = float("inf")
            if self.glycan_df is not None:
                for _, row in self.glycan_df.iterrows():
                    d = abs(float(row["mz"]) - pk)
                    if d < best_dist and d < 0.5:
                        lbl = str(row["label"])
                        if lbl and lbl not in ("nan", ""):
                            best_dist = d
                            best_label = f"{lbl}  ({pk:.2f})"
            if best_dist == float("inf"):
                nearest = min(self.mz_to_label.keys(), key=lambda m: abs(m - pk), default=None)
                if nearest is not None and abs(nearest - pk) < 0.5:
                    lbl = self.mz_to_label[nearest]
                    if not _looks_numeric(lbl):
                        best_label = f"{lbl}  ({pk:.2f})"
            names.append(best_label)
        self.glycan_names = names

        self.nomenclature_labels: dict[float, str] = {
            pk: self._clean_glycan_name(pk) for pk in self.peaks
        }

    def _build_theoretical_mz_lookup(self):
        self.peak_theoretical_mz: dict[float, float] = {}
        ref = getattr(self, "glycan_reference_df", None)
        if ref is None or ref.empty:
            return

        for pk in self.peaks:
            name = self.nomenclature_labels.get(pk, "") if hasattr(self, "nomenclature_labels") else ""
            if not name:
                continue

            matches = ref[ref["label"].str.lower() == name.strip().lower()]
            if matches.empty:
                continue

            # If multiple rows share the same name, pick the theoretical
            # mass that produces the smallest absolute delta to the observed peak.
            best_theo = matches.loc[
                (matches["mz"] - pk).abs().idxmin(), "mz"
            ]
            self.peak_theoretical_mz[pk] = float(best_theo)


    def _on_nomenclature_row_clicked(self, row: int, col: int):
        if row < 0 or row >= len(self.peaks):
            return
        pk = self.peaks[row]
        name = self.nomenclature_labels.get(pk, "") or f"{pk:.4f}"
        # Reuse the existing glycan_selected signal — already wired to
        # spectrum_widget.highlight_glycan in launch_goatpy_gui
        self.glycan_selected.emit(pk, name)

    def _clean_glycan_name(self, pk: float) -> str:
        """Return just the glycan name for a peak (blank if unannotated)."""
        nearest = min(self.mz_to_label.keys(), key=lambda m: abs(m - pk), default=None)
        if nearest is not None and abs(nearest - pk) < 0.5:
            lbl = self.mz_to_label[nearest]
            if not _looks_numeric(lbl):
                return lbl
        return ""



    def _adata(self):
        return self.sdata.tables[self.table_name]

    def _get_peak_index(self, peak_mz: float) -> Optional[int]:
        adata = self._adata()
        try:
            var_mzs = _resolve_mz_array(adata)
            idx = int(np.argmin(np.abs(var_mzs - peak_mz)))
            if abs(var_mzs[idx] - peak_mz) < 0.5:
                return idx
        except Exception:
            pass
        return None

    # ── UI ────────────────────────────────────────────────────────────────

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        title = QLabel("goatpy  Analysis")
        title.setStyleSheet(
            f"font-size: 14px; font-weight: bold; color: {PALETTE['accent']}; padding: 4px 0;"
        )
        layout.addWidget(title)

        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)
        self.tabs.addTab(self._build_glycan_tab(),    "Glycan")
        self.tabs.addTab(self._build_metadata_tab(), "Metadata")
        self.tabs.addTab(self._build_umap_tab(),     "UMAP")
        self.tabs.addTab(self._build_heatmap_tab(),  "Heatmap")
        self.tabs.addTab(self._build_annotations_tab(), "Annotations")
        self.tabs.addTab(self._build_nomenclature_tab(), "Nomenclature")   # ← new
        self.tabs.addTab(self._build_stats_tab(),    "Stats")

    # ── Glycan tab ────────────────────────────────────────────────────────

    def _build_glycan_tab(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(6)

        # Selector
        sel_grp = QGroupBox("Select Glycan / m/z")
        sel_layout = QVBoxLayout(sel_grp)

        self.glycan_search = QComboBox()
        self.glycan_search.setEditable(True)
        self.glycan_search.setInsertPolicy(QComboBox.NoInsert)
        self.glycan_search.addItems(self.glycan_names)
        self.glycan_search.setCurrentIndex(0)
        sel_layout.addWidget(self.glycan_search)

        show_btn = QPushButton("Show on H&E Viewer")
        show_btn.clicked.connect(self._on_glycan_show)

        # In _build_glycan_tab, after the "show_btn" line, add:

        cmap_row = QHBoxLayout()
        cmap_row.addWidget(QLabel("Ion map colormap:"))
        self.ion_cmap_combo = QComboBox()
        self.ion_cmap_combo.addItems(CONTINUOUS_CMAPS)
        self.ion_cmap_combo.setCurrentText("inferno")
        self.ion_cmap_combo.currentTextChanged.connect(self._on_cmap_changed)
        cmap_row.addWidget(self.ion_cmap_combo)
        sel_layout.addLayout(cmap_row)

        sel_layout.addWidget(show_btn)

        note = QLabel("Ion map is added as a napari layer over the H&E image.")
        note.setStyleSheet(f"color: {PALETTE['text_dim']}; font-size: 9px;")
        note.setWordWrap(True)
        sel_layout.addWidget(note)

        layout.addWidget(sel_grp)

        legend_grp = QGroupBox("Spatial legend")
        legend_layout = QVBoxLayout(legend_grp)
        self.glycan_legend_canvas = MplCanvas(w, width=4.5, height=1.6, dpi=90)
        self.glycan_legend_canvas.setMinimumHeight(70)
        legend_layout.addWidget(self.glycan_legend_canvas)
        layout.addWidget(legend_grp)

        # Distribution plot
        plot_grp = QGroupBox("Distribution Plot")
        plot_layout = QVBoxLayout(plot_grp)

        row1 = QHBoxLayout()
        row1.addWidget(QLabel("Type:"))
        self.glycan_plot_type = QComboBox()
        self.glycan_plot_type.addItems(["Violin by cluster", "Box by cluster", "Histogram"])
        row1.addWidget(self.glycan_plot_type)
        plot_layout.addLayout(row1)

        row2 = QHBoxLayout()
        row2.addWidget(QLabel("Group:"))
        self.cluster_col_combo = QComboBox()
        self._populate_obs_categoricals(self.cluster_col_combo)
        row2.addWidget(self.cluster_col_combo)
        plot_layout.addLayout(row2)

        dist_btn = QPushButton("Plot Distribution")
        dist_btn.clicked.connect(self._on_dist_plot)
        plot_layout.addWidget(dist_btn)

        layout.addWidget(plot_grp)

        self.glycan_canvas = MplCanvas(w, width=4.5, height=3.8, dpi=90)
        layout.addWidget(self.glycan_canvas)

        return w

    def _populate_obs_categoricals(self, combo: QComboBox):
        try:
            obs = self._adata().obs
            cats = [c for c in obs.columns
                    if obs[c].dtype.name == "category" or
                    c in ("GPCA_clusters", "leiden", "batch", "annotation")]
            combo.clear()
            combo.addItems(cats if cats else ["(none)"])
        except Exception:
            combo.addItems(["(none)"])

    def _populate_obs_columns(self, combo: QComboBox):
        try:
            cols = list(self._adata().obs.columns)
            combo.clear()
            combo.addItems(cols if cols else ["(none)"])
        except Exception:
            combo.addItems(["(none)"])

    def _populate_shapes_elements(self, combo: QComboBox):
        try:
            names = list(self.sdata.shapes.keys())
            combo.clear()
            combo.addItems(names if names else ["pixels"])
        except Exception:
            combo.addItems(["pixels"])

    # ── Metadata tab ──────────────────────────────────────────────────────

    def _build_metadata_tab(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(6)

        sel_grp = QGroupBox("Plot metadata on shapes")
        sel_layout = QVBoxLayout(sel_grp)

        shapes_row = QHBoxLayout()
        shapes_row.addWidget(QLabel("Shapes layer:"))
        self.meta_shapes_combo = QComboBox()
        self._populate_shapes_elements(self.meta_shapes_combo)
        self.meta_shapes_combo.currentTextChanged.connect(
            lambda name: self._populate_meta_columns(self.meta_col_combo, name)
        )
        shapes_row.addWidget(self.meta_shapes_combo)
        sel_layout.addLayout(shapes_row)

        col_row = QHBoxLayout()
        col_row.addWidget(QLabel("metadata column:"))
        self.meta_col_combo = QComboBox()
        self._populate_meta_columns(self.meta_col_combo, self.meta_shapes_combo.currentText())
        self.meta_col_combo.currentTextChanged.connect(self._on_meta_col_changed)
        col_row.addWidget(self.meta_col_combo)
        sel_layout.addLayout(col_row)

        self.meta_type_lbl = QLabel("Type: —")
        self.meta_type_lbl.setStyleSheet(f"color: {PALETTE['text_dim']}; font-size: 9px;")
        sel_layout.addWidget(self.meta_type_lbl)

        cmap_row = QHBoxLayout()
        cmap_row.addWidget(QLabel("Colormap:"))
        self.meta_cmap_combo = QComboBox()
        self.meta_cmap_combo.addItems(CONTINUOUS_CMAPS)
        self.meta_cmap_combo.currentTextChanged.connect(self._on_meta_cmap_changed)
        cmap_row.addWidget(self.meta_cmap_combo)
        sel_layout.addLayout(cmap_row)

        show_btn = QPushButton("Show on H&E Viewer")
        show_btn.clicked.connect(self._on_metadata_show)
        sel_layout.addWidget(show_btn)

        note = QLabel(
            "Colour the selected shapes layer by metadata values. "
            "Categorical values use discrete colormaps; numeric values use continuous ones."
        )
        note.setStyleSheet(f"color: {PALETTE['text_dim']}; font-size: 9px;")
        note.setWordWrap(True)
        sel_layout.addWidget(note)

        layout.addWidget(sel_grp)

        legend_grp = QGroupBox("Spatial legend")
        legend_layout = QVBoxLayout(legend_grp)
        self.meta_legend_canvas = MplCanvas(w, width=4.5, height=1.6, dpi=90)
        self.meta_legend_canvas.setMinimumHeight(70)
        legend_layout.addWidget(self.meta_legend_canvas)
        layout.addWidget(legend_grp)

        layout.addStretch()
        self._on_meta_col_changed(self.meta_col_combo.currentText())
        return w

    def _on_meta_col_changed(self, col: str):
        if col == "(none)":
            self.meta_type_lbl.setText("Type: —")
            return
        shapes_name = self.meta_shapes_combo.currentText()
        series = self._get_metadata_series(shapes_name, col)
        if series is None:
            self.meta_type_lbl.setText("Type: —")
            return
        is_cat = _safe_is_series_categorical(series)
        self.meta_type_lbl.setText(
            f"Type: {'categorical' if is_cat else 'continuous'}"
        )
        cmap = self.meta_cmap_combo.currentText()
        self.meta_cmap_combo.blockSignals(True)
        self.meta_cmap_combo.clear()
        self.meta_cmap_combo.addItems(CATEGORICAL_CMAPS if is_cat else CONTINUOUS_CMAPS)
        idx = self.meta_cmap_combo.findText(cmap)
        self.meta_cmap_combo.setCurrentIndex(idx if idx >= 0 else 0)
        self.meta_cmap_combo.blockSignals(False)

    def _get_metadata_series(self, shapes_name: str, col: str):
        if shapes_name == "pixels":
            adata = self._adata()
            return adata.obs[col] if col in adata.obs.columns else None
        if shapes_name not in self.sdata.shapes:
            return None
        data = _shape_to_dataframe(self.sdata.shapes[shapes_name])
        if data is None or col not in data.columns:
            return None
        series = data[col]
        # 'colour' columns may be stored as hex strings (good) or list/array RGB
        # (legacy). Convert legacy list-form to hex so downstream categorical
        # detection and colormap logic works correctly.
        if col == "colour":
            sample = series.iloc[0] if len(series) else None
            if isinstance(sample, (list, tuple, np.ndarray)):
                series = series.apply(_rgb_to_hex)
        return series

    def _get_shape_metadata_columns(self, shapes_name: str) -> list[str]:
        if shapes_name == "pixels":
            return list(self._adata().obs.columns)
        if shapes_name not in self.sdata.shapes:
            return []
        data = _shape_to_dataframe(self.sdata.shapes[shapes_name])
        if data is None:
            return []
        return [c for c in data.columns if c != "geometry"]

    def _populate_meta_columns(self, combo: QComboBox, shapes_name: str):
        combo.blockSignals(True)
        combo.clear()
        columns = self._get_shape_metadata_columns(shapes_name)
        if not columns:
            combo.addItem("(none)")
        else:
            combo.addItems(columns)
        combo.blockSignals(False)
        if hasattr(self, "meta_cmap_combo"):
            self._on_meta_col_changed(combo.currentText())

    def _on_meta_cmap_changed(self, cmap_name: str):
        """Re-apply colormap after a metadata render, and refresh the sidebar legend."""
        if not cmap_name or self._last_meta_render is None:
            return
        layer = _find_shapes_layer(self.viewer, self._last_meta_render["shapes_name"])
        if layer is None:
            return
        try:
            n_cat = len(self._last_meta_render.get("categories") or [])
            _apply_shapes_colormap(
                layer, cmap_name,
                categorical=self._last_meta_render["categorical"],
                n_categories=n_cat,
            )
            self._last_meta_render["colormap"] = cmap_name
            self._draw_meta_legend()
        except Exception as e:
            show_info(f"Could not apply colormap: {e}")

    def _on_metadata_show(self):
        col = self.meta_col_combo.currentText()
        shapes_name = self.meta_shapes_combo.currentText()
        if col == "(none)":
            show_info("Select a valid metadata column.")
            return

        series = self._get_metadata_series(shapes_name, col)
        if series is None:
            show_info(f"Column '{col}' not found for shapes layer '{shapes_name}'.")
            return

        values = series.values
        is_cat = _safe_is_series_categorical(series)
        cmap = self.meta_cmap_combo.currentText()
        state = _render_values_on_shapes(
            self.viewer, values, cmap, is_cat,
            col, shapes_name,
        )
        if state is not None:
            self._last_meta_render = state
            self._draw_meta_legend()

    def _on_glycan_show(self):
        idx = self.glycan_search.currentIndex()
        if idx < 0 or idx >= len(self.peaks):
            return
        pk = self.peaks[idx]
        label = self.glycan_names[idx]
        self._render_on_viewer(pk, label)
        self.glycan_selected.emit(pk, label)

    def _on_dist_plot(self):
        idx = self.glycan_search.currentIndex()
        if idx < 0 or idx >= len(self.peaks):
            return
        pk = self.peaks[idx]
        label = self.glycan_names[idx]
        self._draw_distribution(pk, label)

    def _render_on_viewer(self, peak_mz: float, label: str):
        state = _render_glycan_on_viewer(
            self.viewer, self.sdata, peak_mz, label, self.table_name,
            colormap=self.ion_cmap_combo.currentText(),
        )
        if state is not None:
            self._last_glycan_render = state
            self._draw_glycan_legend()

    def _draw_distribution(self, peak_mz: float, label: str):
        col_idx = self._get_peak_index(peak_mz)
        if col_idx is None:
            show_info(f"Peak {peak_mz:.2f} not found.")
            return

        adata = self._adata()
        X = np.asarray(adata.X, dtype=np.float32)
        values = X[:, col_idx]
        short = label.split("(")[0].strip()[:28]
        cluster_col = self.cluster_col_combo.currentText()
        plot_type = self.glycan_plot_type.currentText()

        self.glycan_canvas.fig.clear()
        ax = self.glycan_canvas.fig.add_subplot(111)
        self.glycan_canvas._style_ax(ax)

        if plot_type == "Histogram":
            ax.hist(values[values > 0], bins=50,
                    color=PALETTE["accent"], edgecolor="none", alpha=0.8)
            ax.set_xlabel("Intensity", fontsize=8)
            ax.set_ylabel("# Pixels", fontsize=8)
            ax.set_title(short, fontsize=9)
        else:
            self._plot_by_cluster(
                ax, adata, values, cluster_col,
                kind="violin" if "Violin" in plot_type else "box",
                label=short,
            )

        self.glycan_canvas.fig.tight_layout(pad=0.5)
        self.glycan_canvas.draw()

    def _plot_by_cluster(self, ax, adata, values, cluster_col, kind, label):
        if cluster_col == "(none)" or cluster_col not in adata.obs.columns:
            ax.text(0.5, 0.5, "No cluster column\nfound",
                    ha="center", va="center", transform=ax.transAxes,
                    color=PALETTE["text_dim"])
            return
        clusters = adata.obs[cluster_col].astype(str).values
        unique = sorted(set(clusters))
        cmap = plt.get_cmap("tab20", len(unique))
        data_by_cluster = [values[clusters == c] for c in unique]

        if kind == "violin":
            parts = ax.violinplot(data_by_cluster, positions=range(len(unique)),
                                  showmedians=True, showextrema=False)
            for i, pc in enumerate(parts["bodies"]):
                pc.set_facecolor(cmap(i)); pc.set_alpha(0.7)
            parts["cmedians"].set_color(PALETTE["text"])
        else:
            bp = ax.boxplot(data_by_cluster, positions=range(len(unique)),
                            patch_artist=True, showfliers=False,
                            medianprops={"color": PALETTE["text"], "linewidth": 1.5},
                            whiskerprops={"color": PALETTE["text_dim"]},
                            capprops={"color": PALETTE["text_dim"]})
            for i, patch in enumerate(bp["boxes"]):
                patch.set_facecolor(cmap(i)); patch.set_alpha(0.75)

        ax.set_xticks(range(len(unique)))
        ax.set_xticklabels(unique, rotation=45, ha="right", fontsize=7)
        ax.set_ylabel("Intensity", fontsize=8)
        ax.set_title(f"{label}  by {cluster_col}", fontsize=8.5)

    # ── Unregistered peak: display ion image spatially ─────────────────────

    def display_unregistered_ion_image(self, mz: float, tol: float):
        """Compute per-pixel intensities for an arbitrary m/z (+/- tol) from
        the raw imzML and render them on the 'pixels' Shapes layer, exactly
        like the curated glycan ion maps (guarantees H&E alignment)."""
        try:
            path = self.sdata.tables[self.table_name].uns.get("maldi_path")
        except Exception:
            path = None

        if not path:
            show_info("No raw imzML path found (uns['maldi_path']) — cannot compute ion image.")
            return

        label = f"m/z {mz:.4f}"
        state = _display_unregistered_ion_image_on_shapes(
            self.viewer, self.sdata, path, mz, tol,
            label=label, table_name=self.table_name,
            colormap=self.ion_cmap_combo.currentText(),
        )
        if state is not None:
            self._last_glycan_render = state
            self._draw_glycan_legend()

    # ── Select glycan from spectrum click ─────────────────────────────────

    def select_peak_from_spectrum(self, mz: float, label: str):
        """Called when the user clicks a peak in the spectrum widget."""
        # Update combo box to match if possible
        for i, pk in enumerate(self.peaks):
            if abs(pk - mz) < 0.5:
                self.glycan_search.setCurrentIndex(i)
                break
        self._render_on_viewer(mz, label)

    # ── UMAP tab ──────────────────────────────────────────────────────────

    def _build_umap_tab(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setContentsMargins(4, 4, 4, 4)

        col_grp = QGroupBox("Colour By")
        col_layout = QVBoxLayout(col_grp)
        self.umap_color_type = QComboBox()
        self.umap_color_type.addItems(["Metadata column", "Glycan intensity"])
        self.umap_color_type.currentTextChanged.connect(self._toggle_umap_controls)
        col_layout.addWidget(self.umap_color_type)

        self.umap_meta_col = QComboBox()
        self._populate_obs_columns(self.umap_meta_col)
        self.umap_meta_col.currentTextChanged.connect(self._on_umap_meta_col_changed)
        col_layout.addWidget(self.umap_meta_col)

        self.umap_glycan_combo = QComboBox()
        self.umap_glycan_combo.addItems(self.glycan_names)
        self.umap_glycan_combo.setVisible(False)
        col_layout.addWidget(self.umap_glycan_combo)

        cmap_row = QHBoxLayout()
        cmap_row.addWidget(QLabel("Colormap:"))
        self.umap_cmap_combo = QComboBox()
        self.umap_cmap_combo.addItems(CONTINUOUS_CMAPS)
        self.umap_cmap_combo.setCurrentText("inferno")
        cmap_row.addWidget(self.umap_cmap_combo)
        col_layout.addLayout(cmap_row)

        layout.addWidget(col_grp)

        emb_grp = QGroupBox("Embedding")
        emb_layout = QHBoxLayout(emb_grp)
        self.umap_embedding_combo = QComboBox()
        self._populate_obsm(self.umap_embedding_combo)
        emb_layout.addWidget(self.umap_embedding_combo)
        layout.addWidget(emb_grp)

        plot_btn = QPushButton("Plot UMAP")
        plot_btn.clicked.connect(self._draw_umap)
        layout.addWidget(plot_btn)

        self.umap_canvas = MplCanvas(w, width=4.5, height=4.5, dpi=90)
        layout.addWidget(self.umap_canvas)

        # Initialise colormap list for the starting metadata selection.
        self._on_umap_meta_col_changed(self.umap_meta_col.currentText())
        return w

    def _toggle_umap_controls(self, text):
        is_glycan = text == "Glycan intensity"
        self.umap_glycan_combo.setVisible(is_glycan)
        self.umap_meta_col.setVisible(not is_glycan)
        self.umap_cmap_combo.setVisible(True)
        if not is_glycan:
            self._on_umap_meta_col_changed(self.umap_meta_col.currentText())
        else:
            self.umap_cmap_combo.blockSignals(True)
            self.umap_cmap_combo.clear()
            self.umap_cmap_combo.addItems(CONTINUOUS_CMAPS)
            self.umap_cmap_combo.setCurrentText("inferno")
            self.umap_cmap_combo.blockSignals(False)

    def _on_umap_meta_col_changed(self, col: str):
        if col == "(none)":
            return
        adata = self._adata()
        if col not in adata.obs.columns:
            return
        is_cat = _is_obs_categorical(adata, col)
        current = self.umap_cmap_combo.currentText()
        self.umap_cmap_combo.blockSignals(True)
        self.umap_cmap_combo.clear()
        self.umap_cmap_combo.addItems(CATEGORICAL_CMAPS if is_cat else CONTINUOUS_CMAPS)
        if current in (CATEGORICAL_CMAPS if is_cat else CONTINUOUS_CMAPS):
            self.umap_cmap_combo.setCurrentText(current)
        self.umap_cmap_combo.blockSignals(False)

    def _populate_obsm(self, combo: QComboBox):
        try:
            keys = list(self._adata().obsm.keys())
            combo.addItems(keys if keys else ["(none)"])
        except Exception:
            combo.addItems(["(none)"])

    def _draw_umap(self):
        adata = self._adata()
        emb_key = self.umap_embedding_combo.currentText()
        if emb_key == "(none)" or emb_key not in adata.obsm:
            show_info("No embedding found. Run graphpca_spatialdata() first.")
            return
        emb = adata.obsm[emb_key]
        if emb.shape[1] == 2:
            coords = emb
        else:
            try:
                from umap import UMAP
                coords = UMAP(n_components=2, random_state=42).fit_transform(emb)
            except ImportError:
                coords = emb[:, :2]

        color_type = self.umap_color_type.currentText()
        self.umap_canvas.fig.clear()
        ax = self.umap_canvas.fig.add_subplot(111)
        self.umap_canvas._style_ax(ax)

        if color_type == "Metadata column":
            col = self.umap_meta_col.currentText()
            if col != "(none)" and col in adata.obs.columns:
                cmap_name = self.umap_cmap_combo.currentText()
                labels = adata.obs[col]
                if _is_obs_categorical(adata, col):
                    labels = labels.astype(str).values
                    unique = sorted(set(labels))
                    cmap = plt.get_cmap(cmap_name, len(unique))
                    for i, cl in enumerate(unique):
                        mask = labels == cl
                        ax.scatter(coords[mask, 0], coords[mask, 1],
                                   s=1, alpha=0.6, color=cmap(i), label=cl, rasterized=True)
                    ax.legend(markerscale=4, fontsize=6.5, framealpha=0.3,
                              facecolor=PALETTE["surface"], labelcolor=PALETTE["text"],
                              bbox_to_anchor=(1, 1), loc="upper left")
                else:
                    values = pd.to_numeric(labels, errors="coerce").astype(np.float32)
                    valid = values[np.isfinite(values)]
                    vmin = float(np.percentile(valid, 1)) if len(valid) else 0.0
                    vmax = float(np.percentile(valid, 99)) if len(valid) else 1.0
                    sc = ax.scatter(coords[:, 0], coords[:, 1], c=values, cmap=cmap_name,
                                    s=1, alpha=0.6, vmin=vmin, vmax=vmax, rasterized=True)
                    self.umap_canvas.fig.colorbar(sc, ax=ax, shrink=0.8, pad=0.01)
                ax.set_title(f"UMAP — {col}", fontsize=8.5)
            else:
                ax.scatter(coords[:, 0], coords[:, 1], s=1, alpha=0.4,
                           color=PALETTE["accent"], rasterized=True)
        else:
            idx_g = self.umap_glycan_combo.currentIndex()
            if idx_g < len(self.peaks):
                pk = self.peaks[idx_g]
                col_idx = self._get_peak_index(pk)
                if col_idx is not None:
                    v = np.asarray(adata.X, dtype=np.float32)[:, col_idx]
                    sc = ax.scatter(coords[:, 0], coords[:, 1], c=v, cmap="inferno",
                                    s=1, alpha=0.6,
                                    vmin=np.percentile(v, 1), vmax=np.percentile(v, 99),
                                    rasterized=True)
                    self.umap_canvas.fig.colorbar(sc, ax=ax, shrink=0.8, pad=0.01)
                    ax.set_title(
                        f"UMAP — {self.glycan_names[idx_g].split('(')[0][:20]}", fontsize=8.5
                    )
        ax.set_xlabel("UMAP 1", fontsize=8)
        ax.set_ylabel("UMAP 2", fontsize=8)
        self.umap_canvas.fig.tight_layout(pad=0.5)
        self.umap_canvas.draw()

    # ── Heatmap tab ───────────────────────────────────────────────────────

    def _build_heatmap_tab(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setContentsMargins(4, 4, 4, 4)

        ctrl_grp = QGroupBox("Options")
        ctrl_layout = QVBoxLayout(ctrl_grp)

        r1 = QHBoxLayout()
        r1.addWidget(QLabel("Group by:"))
        self.hmap_group_col = QComboBox()
        self._populate_obs_categoricals(self.hmap_group_col)
        r1.addWidget(self.hmap_group_col)
        ctrl_layout.addLayout(r1)

        r2 = QHBoxLayout()
        r2.addWidget(QLabel("Top N peaks:"))
        self.hmap_topn = QSpinBox()
        self.hmap_topn.setRange(5, 200)
        self.hmap_topn.setValue(30)
        r2.addWidget(self.hmap_topn)
        ctrl_layout.addLayout(r2)

        r3 = QHBoxLayout()
        r3.addWidget(QLabel("Normalise:"))
        self.hmap_norm_combo = QComboBox()
        self.hmap_norm_combo.addItems(["z-score", "min-max", "none"])
        r3.addWidget(self.hmap_norm_combo)
        ctrl_layout.addLayout(r3)

        glycan_row = QHBoxLayout()
        self.hmap_glycan_lbl = QLabel("Glycans: top N (default)")
        self.hmap_glycan_lbl.setStyleSheet(f"color: {PALETTE['text_dim']}; font-size: 9px;")
        self.hmap_glycan_lbl.setWordWrap(True)
        glycan_row.addWidget(self.hmap_glycan_lbl, stretch=1)
        glycan_btn = QPushButton("Select Glycans…")
        glycan_btn.clicked.connect(self._open_glycan_selection)
        glycan_row.addWidget(glycan_btn)
        ctrl_layout.addLayout(glycan_row)

        clear_btn = QPushButton("Clear Selection")
        clear_btn.clicked.connect(self._clear_glycan_selection)
        ctrl_layout.addWidget(clear_btn)

        layout.addWidget(ctrl_grp)
        plot_btn = QPushButton("Plot Heatmap")
        plot_btn.clicked.connect(self._draw_heatmap)
        layout.addWidget(plot_btn)
        self.hmap_canvas = MplCanvas(w, width=4.5, height=5.5, dpi=90)
        layout.addWidget(self.hmap_canvas)
        return w


    # ── Nomenclature tab ────────────────────────────────────────────────

    def _build_nomenclature_tab(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(6)

        info = QLabel("Curated m/z values and their assigned glycan names.")
        info.setStyleSheet(f"color: {PALETTE['text_dim']}; font-size: 9px;")
        info.setWordWrap(True)
        layout.addWidget(info)

        self.nomenclature_table = QTableWidget()
        self.nomenclature_table.setColumnCount(3)
        self.nomenclature_table.setHorizontalHeaderLabels(["m/z (observed)", "Glycan", "Δ m/z"])
        self.nomenclature_table.horizontalHeader().setStretchLastSection(True)
        self.nomenclature_table.setColumnWidth(0, 100)
        self.nomenclature_table.setColumnWidth(2, 80)
        layout.addWidget(self.nomenclature_table)
        self._populate_nomenclature_table()
        self.nomenclature_table.cellClicked.connect(self._on_nomenclature_row_clicked)

        edit_row = QHBoxLayout()
        self.edit_names_btn = QPushButton("Edit Names")
        self.edit_names_btn.setCheckable(True)
        self.edit_names_btn.toggled.connect(self._on_toggle_edit_names)
        edit_row.addWidget(self.edit_names_btn)

        self.save_names_btn = QPushButton("Save Changes")
        self.save_names_btn.clicked.connect(self._on_save_nomenclature)
        edit_row.addWidget(self.save_names_btn)
        layout.addLayout(edit_row)

        spectra_grp = QGroupBox("Show on Spectra")
        spectra_layout = QHBoxLayout(spectra_grp)
        spectra_layout.addWidget(QLabel("Tolerance (± Da):"))
        self.nomenclature_tol_spin = QDoubleSpinBox()
        self.nomenclature_tol_spin.setRange(0.001, 50.0)
        self.nomenclature_tol_spin.setDecimals(3)
        self.nomenclature_tol_spin.setSingleStep(0.01)
        self.nomenclature_tol_spin.setValue(0.15)
        self.nomenclature_tol_spin.setFixedWidth(80)
        spectra_layout.addWidget(self.nomenclature_tol_spin)
        spectra_layout.addStretch()

        show_spectra_btn = QPushButton("Show on Spectra")
        show_spectra_btn.clicked.connect(self._on_show_nomenclature_on_spectra)
        spectra_layout.addWidget(show_spectra_btn)

        clear_spectra_btn = QPushButton("Clear")
        clear_spectra_btn.clicked.connect(
            lambda: self.show_annotated_on_spectra.emit([], 0.0)
        )
        spectra_layout.addWidget(clear_spectra_btn)
        layout.addWidget(spectra_grp)

        note = QLabel(
            "Successfully annotated peaks (non-blank glycan name) are highlighted "
            "in green on the Spectrum panel, within the chosen tolerance window."
        )
        note.setStyleSheet(f"color: {PALETTE['text_dim']}; font-size: 9px;")
        note.setWordWrap(True)
        layout.addWidget(note)

        layout.addStretch()
        return w

    def _populate_nomenclature_table(self):
        self.nomenclature_table.setRowCount(len(self.peaks))
        for row, pk in enumerate(self.peaks):
            mz_item = QTableWidgetItem(f"{pk:.4f}")
            mz_item.setFlags(mz_item.flags() & ~Qt.ItemIsEditable)
            self.nomenclature_table.setItem(row, 0, mz_item)

            name = self.nomenclature_labels.get(pk, "")
            name_item = QTableWidgetItem(name)
            name_item.setFlags(name_item.flags() & ~Qt.ItemIsEditable)
            self.nomenclature_table.setItem(row, 1, name_item)

            theo = self.peak_theoretical_mz.get(pk)
            if theo is not None:
                delta = theo - pk          # theoretical − observed
                sign = "+" if delta >= 0 else ""
                delta_text = f"{sign}{delta:.4f}"
                abs_d = abs(delta)
                if abs_d < 0.05:
                    colour = PALETTE["success"]
                elif abs_d < 0.15:
                    colour = PALETTE["accent2"]
                else:
                    colour = PALETTE["peak_marker"]
            else:
                delta_text = "—"
                colour = PALETTE["text_dim"]

            delta_item = QTableWidgetItem(delta_text)
            delta_item.setFlags(delta_item.flags() & ~Qt.ItemIsEditable)
            delta_item.setForeground(QColor(colour))
            self.nomenclature_table.setItem(row, 2, delta_item)

    def _on_toggle_edit_names(self, checked: bool):
        for row in range(self.nomenclature_table.rowCount()):
            item = self.nomenclature_table.item(row, 1)
            if item is None:
                continue
            if checked:
                item.setFlags(item.flags() | Qt.ItemIsEditable)
            else:
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)
        self.edit_names_btn.setText(
            "Editing… (click to lock)" if checked else "Edit Names"
        )

    def _on_save_nomenclature(self):
        updated: dict[float, str] = {}
        for row, pk in enumerate(self.peaks):
            item = self.nomenclature_table.item(row, 1)
            updated[pk] = item.text().strip() if item is not None else ""

        self.nomenclature_labels = updated

        # Keep the rest of the sidebar (Glycan dropdown, UMAP, etc.) in sync
        for pk, name in updated.items():
            if name:
                self.mz_to_label[pk] = name
                self.label_to_mz[name] = pk

        new_names = []
        for pk in self.peaks:
            name = updated.get(pk, "")
            new_names.append(f"{name}  ({pk:.2f})" if name else f"{pk:.4f}")
        self.glycan_names = new_names
        self._refresh_glycan_name_combos()
        self._build_theoretical_mz_lookup()
        # If a glycan_df was supplied at launch, keep it in sync for this session
        if self.glycan_df is not None and "mz" in self.glycan_df.columns:
            for pk, name in updated.items():
                if not name:
                    continue
                mask = (self.glycan_df["mz"].astype(float) - pk).abs() < 1e-6
                if mask.any():
                    self.glycan_df.loc[mask, "label"] = name

        self.edit_names_btn.setChecked(False)
        self.nomenclature_updated.emit({k: v for k, v in updated.items() if v})
        show_info("Glycan names saved.")

    def _refresh_glycan_name_combos(self):
        for combo in (getattr(self, "glycan_search", None),
                      getattr(self, "umap_glycan_combo", None)):
            if combo is None:
                continue
            current = combo.currentIndex()
            combo.blockSignals(True)
            combo.clear()
            combo.addItems(self.glycan_names)
            if 0 <= current < combo.count():
                combo.setCurrentIndex(current)
            combo.blockSignals(False)

    def _on_show_nomenclature_on_spectra(self):
        tol = self.nomenclature_tol_spin.value()
        theoretical_mzs = []
        unmatched = 0

        for pk in self.peaks:
            name = self.nomenclature_labels.get(pk, "")
            if not name:
                continue
            theo = self.peak_theoretical_mz.get(pk)
            if theo is not None:
                theoretical_mzs.append(theo)
            else:
                unmatched += 1

        if not theoretical_mzs:
            show_info(
                "No annotated glycans matched a theoretical mass in "
                "glycan_list.csv — nothing to show."
            )
            return

        if unmatched:
            show_info(
                f"{unmatched} annotated glycan(s) had no match in "
                "glycan_list.csv and were skipped."
            )

        self.show_annotated_on_spectra.emit(theoretical_mzs, tol)



    def _open_glycan_selection(self):
        dlg = GlycanSelectionDialog(
            self.glycan_names, self.peaks, self.label_to_mz, parent=self,
        )
        if dlg.exec_() == QDialog.Accepted:
            self.hmap_custom_indices = dlg.selected_indices
            n = len(self.hmap_custom_indices)
            preview = ", ".join(
                self.glycan_names[i].split("(")[0].strip()[:16]
                for i in self.hmap_custom_indices[:4]
            )
            suffix = f" … +{n - 4} more" if n > 4 else ""
            self.hmap_glycan_lbl.setText(f"Glycans: {n} selected ({preview}{suffix})")
            self.hmap_glycan_lbl.setStyleSheet(f"color: {PALETTE['text']}; font-size: 9px;")

    def _clear_glycan_selection(self):
        self.hmap_custom_indices = None
        self.hmap_glycan_lbl.setText("Glycans: top N (default)")
        self.hmap_glycan_lbl.setStyleSheet(f"color: {PALETTE['text_dim']}; font-size: 9px;")

    def _draw_heatmap(self):
        adata = self._adata()
        group_col = self.hmap_group_col.currentText()
        top_n = self.hmap_topn.value()
        norm = self.hmap_norm_combo.currentText()

        if group_col == "(none)" or group_col not in adata.obs.columns:
            show_info("Select a valid grouping column first.")
            return

        X = np.asarray(adata.X, dtype=np.float32)
        labels = adata.obs[group_col].astype(str).values
        groups = sorted(set(labels))
        means = np.array([X[labels == g].mean(axis=0) for g in groups])

        if self.hmap_custom_indices:
            col_idx = self._resolve_heatmap_column_indices(adata)
            if not col_idx:
                return
            top_idx = np.array(col_idx, dtype=int)
        else:
            top_idx = np.argsort(means.var(axis=0))[::-1][:top_n]
        means_top = means[:, top_idx]

        if norm == "z-score":
            from scipy.stats import zscore
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                means_top = np.nan_to_num(zscore(means_top, axis=0))
        elif norm == "min-max":
            mn = means_top.min(axis=0, keepdims=True)
            mx = means_top.max(axis=0, keepdims=True)
            with np.errstate(divide="ignore", invalid="ignore"):
                means_top = np.where(mx > mn, (means_top - mn) / (mx - mn), 0.0)

        disp = _resolve_var_display_labels(adata)
        col_labels = [disp[i][:12] for i in top_idx]

        self.hmap_canvas.fig.clear()
        ax = self.hmap_canvas.fig.add_subplot(111)
        self.hmap_canvas._style_ax(ax)
        im = ax.imshow(means_top, aspect="auto", cmap="RdBu_r", interpolation="nearest")
        self.hmap_canvas.fig.colorbar(im, ax=ax, shrink=0.8, pad=0.01,
                                      label=norm if norm != "none" else "intensity")
        ax.set_yticks(range(len(groups)))
        ax.set_yticklabels(groups, fontsize=7)
        ax.set_xticks(range(len(col_labels)))
        ax.set_xticklabels(col_labels, rotation=90, fontsize=5.5, ha="center")
        ax.set_title(f"Mean intensity by {group_col}", fontsize=8.5)
        self.hmap_canvas.fig.tight_layout(pad=0.5)
        self.hmap_canvas.draw()

    def _resolve_heatmap_column_indices(self, adata) -> list[int]:
        """Map selected peak indices to adata column indices."""
        col_indices: list[int] = []
        for peak_i in self.hmap_custom_indices or []:
            if peak_i < 0 or peak_i >= len(self.peaks):
                continue
            col_idx = self._get_peak_index(self.peaks[peak_i])
            if col_idx is not None:
                col_indices.append(col_idx)
        if not col_indices:
            show_info("Selected glycans could not be matched to data columns.")
        return col_indices

    # ── Annotations tab ──────────────────────────────────────────────────

    def _build_annotations_tab(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(6)

        info_lbl = QLabel("Edit annotation labels and colors")
        info_lbl.setStyleSheet(f"color: {PALETTE['text_dim']}; font-size: 9px;")
        layout.addWidget(info_lbl)

        # Create table with columns: ID (hidden), Classification, Colour
        self.annotations_table = QTableWidget()
        self.annotations_table.setColumnCount(3)
        self.annotations_table.setHorizontalHeaderLabels(["ID", "Classification", "Colour"])
        self.annotations_table.horizontalHeader().setStretchLastSection(False)
        self.annotations_table.setColumnHidden(0, True)
        self.annotations_table.setColumnWidth(1, 150)
        self.annotations_table.setColumnWidth(2, 80)
        layout.addWidget(self.annotations_table)


        # ── Outer container: New Annotation ────────────────────────────────
        new_annotation_label = QLabel("Add New Annotation")
        new_annotation_label.setStyleSheet(
            f"color: {PALETTE['text']}; font-weight: bold; font-size: 13px; margin-top: 6px;"
        )

        new_annotation_group = QWidget()
        new_annotation_layout = QVBoxLayout(new_annotation_group)
        new_annotation_layout.setContentsMargins(0, 0, 0, 0)
        new_annotation_layout.setSpacing(4)


        # ── Step 1: Create new annotation ─────────────────────────────────
        step1_group = QGroupBox("1. Assign new annotation label and colour")
        step1_group.setStyleSheet(
            f"QGroupBox {{ background-color: {PALETTE['surface']}; border: 2px solid {PALETTE['accent']}; "
            f"border-radius: 6px; margin-top: 8px; padding-top: 8px; }}"
            f"QGroupBox::title {{ subcontrol-origin: margin; left: 8px; padding: 0 4px; "
            f"color: {PALETTE['accent']}; font-weight: bold; }}"
        )
        step1_layout = QVBoxLayout(step1_group)
        step1_layout.setContentsMargins(4,4,4,4)
        step1_layout.setSpacing(2)
        step1_group.adjustSize()
        step1_group.setSizePolicy(
            QSizePolicy.Preferred,
            QSizePolicy.Maximum
        )

        add_row_layout = QHBoxLayout()
        add_row_layout.addWidget(QLabel("Label:"))
        self.new_annotation_name = QPlainTextEdit()
        self.new_annotation_name.setPlaceholderText("e.g. Tumor region")
        self.new_annotation_name.setMaximumHeight(28)
        add_row_layout.addWidget(self.new_annotation_name, stretch=1)

        self._new_annotation_colour = [255, 165, 0]  # default orange
        new_colour_btn = QPushButton()
        new_colour_btn.setMaximumWidth(40)
        new_colour_btn.setStyleSheet(
            f"background-color: rgb(255,165,0); border: 1px solid {PALETTE['border']}; border-radius: 3px;"
        )
        new_colour_btn.clicked.connect(self._on_pick_new_annotation_colour)
        self._new_annotation_colour_btn = new_colour_btn
        add_row_layout.addWidget(new_colour_btn)
        step1_layout.addLayout(add_row_layout)

        add_btn = QPushButton("Construct Annotation")
        add_btn.setStyleSheet(
            f"QPushButton {{ background-color: {PALETTE['accent']}; color: {PALETTE['bg']}; "
            f"font-weight: bold; padding: 6px; border-radius: 4px; }}"
            f"QPushButton:hover {{ background-color: {PALETTE['highlight']}; }}"
        )
        add_btn.clicked.connect(self._on_add_annotation)
        step1_layout.addWidget(add_btn)

        new_annotation_layout.addWidget(step1_group)

        # ── Step 2: Draw annotation ───────────────────────────────────────
        step2_group = QGroupBox("2. Draw your Annotation (using the rectangle or polygon tool)")
        step2_group.setStyleSheet(
            f"QGroupBox {{ background-color: {PALETTE['surface']}; border: 2px solid {PALETTE['success']}; "
            f"border-radius: 6px; margin-top: 8px; padding-top: 8px; }}"
            f"QGroupBox::title {{ subcontrol-origin: margin; left: 8px; padding: 0 4px; "
            f"color: {PALETTE['success']}; font-weight: bold; }}"
        )
        step2_layout = QVBoxLayout(step2_group)
        step2_layout.setContentsMargins(8, 12, 8, 8)
        step2_layout.setSpacing(4)

        step2_hint = QLabel(
            "After clicking 'Construct Annotation' above, draw the region on the 'Annotations' "
            "layer in napari using the shape tools, then click 'Finish Annotation' below."
        )
        step2_hint.setStyleSheet(f"color: {PALETTE['text_dim']}; font-size: 10px;")
        step2_hint.setWordWrap(True)
        step2_layout.addWidget(step2_hint)

        register_btn = QPushButton("Finish Annotation")
        register_btn.setStyleSheet(
            f"QPushButton {{ background-color: {PALETTE['success']}; color: {PALETTE['bg']}; "
            f"font-weight: bold; padding: 6px; border-radius: 4px; }}"
            f"QPushButton:hover {{ background-color: {PALETTE['highlight']}; }}"
        )
        register_btn.clicked.connect(self._on_register_annotation)
        step2_layout.addWidget(register_btn)

        # Store references for state management
        self._step1_group = step1_group
        self._step2_group = step2_group
        self._step2_group.setEnabled(False)  # Disabled by default
        new_annotation_layout.addWidget(step2_group)

        layout.addWidget(new_annotation_label)
        layout.addWidget(new_annotation_group)


        # ── Save and reload buttons ───────────────────────────────────────
        button_layout = QHBoxLayout()
        save_btn = QPushButton("Save Changes")
        save_btn.clicked.connect(self._on_annotations_save)
        reload_btn = QPushButton("Reload")
        reload_btn.clicked.connect(self._on_annotations_reload)
        button_layout.addWidget(save_btn)
        button_layout.addWidget(reload_btn)
        layout.addLayout(button_layout)


        # Load annotations on tab init
        QTimer.singleShot(100, self._on_annotations_reload)
        return w

    def _on_annotations_reload(self):
        """Load annotations from sdata.shapes['Annotations']."""
        try:
            if "Annotations" not in self.sdata.shapes:
                show_info("No Annotations layer found in SpatialData.")
                self.annotations_table.setRowCount(0)
                # Reset step 2 to disabled
                if hasattr(self, "_step2_group"):
                    self._step2_group.setEnabled(False)
                return

            annotations = self.sdata.shapes["Annotations"]
            data = _shape_to_dataframe(annotations)
            if data is None:
                show_info("Annotations layer is not accessible as tabular data.")
                self.annotations_table.setRowCount(0)
                if hasattr(self, "_step2_group"):
                    self._step2_group.setEnabled(False)
                return

            if "classification" in data.columns and "colour" in data.columns:
                classifications = data["classification"].astype(str).tolist()
                raw_colours = data["colour"].tolist()
                # Accept either hex strings or legacy [r,g,b] lists.
                colours = []
                for c in raw_colours:
                    if c is None:
                        # Use default gray if colour is None
                        colours.append([128, 128, 128])
                    elif isinstance(c, str):
                        colours.append(_hex_to_rgb(c))
                    elif isinstance(c, (list, tuple)):
                        # Ensure it's a proper RGB list
                        colours.append([int(c[0]) if len(c) > 0 else 128,
                                       int(c[1]) if len(c) > 1 else 128,
                                       int(c[2]) if len(c) > 2 else 128])
                    else:
                        colours.append([128, 128, 128])
                
                ids = data["id"].astype(str).tolist() if "id" in data.columns else [str(i) for i in range(len(classifications))]
            else:
                attrs = getattr(annotations, "attrs", None)
                if not isinstance(attrs, dict) or "classification" not in attrs or "colour" not in attrs:
                    show_info("Annotations missing 'classification' or 'colour' metadata.")
                    self.annotations_table.setRowCount(0)
                    if hasattr(self, "_step2_group"):
                        self._step2_group.setEnabled(False)
                    return
                classifications = attrs.get("classification", [])
                raw_colours = attrs.get("colour", [])
                colours = []
                for c in raw_colours:
                    if c is None:
                        colours.append([128, 128, 128])
                    elif isinstance(c, str):
                        colours.append(_hex_to_rgb(c))
                    elif isinstance(c, (list, tuple)):
                        colours.append([int(c[0]) if len(c) > 0 else 128,
                                       int(c[1]) if len(c) > 1 else 128,
                                       int(c[2]) if len(c) > 2 else 128])
                    else:
                        colours.append([128, 128, 128])
                ids = attrs.get("id", [str(i) for i in range(len(classifications))])

            self.annotations_table.setRowCount(len(classifications))
            self._annotations_data = {
                "ids": ids,
                "classifications": list(classifications),
                "colours": colours,
            }

            for row, (ann_id, label, colour) in enumerate(
                zip(ids, classifications, colours)
            ):
                # ID column (hidden)
                id_item = QTableWidgetItem(str(ann_id))
                id_item.setFlags(id_item.flags() & ~Qt.ItemIsEditable)
                self.annotations_table.setItem(row, 0, id_item)

                # Classification column (editable)
                label_item = QTableWidgetItem(str(label))
                self.annotations_table.setItem(row, 1, label_item)

                # Colour column with color picker button
                color_widget = QWidget()
                color_layout = QHBoxLayout(color_widget)
                color_layout.setContentsMargins(2, 2, 2, 2)
                color_btn = QPushButton()
                color_btn.setMaximumWidth(60)
                rgb = colour[:3] if len(colour) >= 3 else [128, 128, 128]
                color_btn.setStyleSheet(
                    f"background-color: rgb({int(rgb[0])}, {int(rgb[1])}, {int(rgb[2])}); "
                    f"border: 1px solid {PALETTE['border']}; border-radius: 3px;"
                )
                color_btn.clicked.connect(lambda checked, r=row: self._on_color_picker(r))
                color_layout.addWidget(color_btn)
                color_layout.addStretch()
                self.annotations_table.setCellWidget(row, 2, color_widget)

            # Reset step 2 to disabled when reloading
            if hasattr(self, "_step2_group"):
                self._step2_group.setEnabled(False)

        except Exception as e:
            show_info(f"Error loading annotations: {e}")

    def _on_color_picker(self, row: int):
        """Open a color picker for a specific annotation."""
        if not hasattr(self, "_annotations_data"):
            return
        current_colour = self._annotations_data["colours"][row]
        rgb = current_colour[:3] if len(current_colour) >= 3 else [0, 0, 0]

        color = QColorDialog.getColor(
            QColor(int(rgb[0]), int(rgb[1]), int(rgb[2])),
            self,
            "Choose annotation colour",
        )
        if color.isValid():
            r, g, b, _ = color.getRgb()
            self._annotations_data["colours"][row] = [r, g, b]

            # Update button color
            color_widget = self.annotations_table.cellWidget(row, 2)
            if color_widget is not None:
                color_btn = color_widget.findChild(QPushButton)
                if color_btn is not None:
                    color_btn.setStyleSheet(
                        f"background-color: rgb({r}, {g}, {b}); "
                        f"border: 1px solid {PALETTE['border']};"
                    )

    def _on_annotations_save(self):
        """Save annotation changes (classification + colour) back to sdata.

        Colours are stored as [r,g,b] integer lists and classification is
        preserved as a categorical column where possible.
        """
        try:
            self._collect_pending_geometries()
            if "Annotations" not in self.sdata.shapes:
                show_info("No Annotations layer to save.")
                return

            if not hasattr(self, "_annotations_data"):
                show_info("No annotation data loaded.")
                return

            annotations = self.sdata.shapes["Annotations"]
            layer = _find_shapes_layer(self.viewer, "Annotations")
            if hasattr(self, "_pending_annotation_base_df") and self._pending_annotation_base_df is not None:
                data = self._pending_annotation_base_df.copy()
            else:
                data = _shape_to_dataframe(annotations)

            n_rows = self.annotations_table.rowCount()
            updated_ids = [
                self._annotations_data["ids"][row]
                if row < len(self._annotations_data.get("ids", []))
                else str(uuid.uuid4())
                for row in range(n_rows)
            ]

            updated_classifications = []
            updated_colours_rgb = []
            for row in range(n_rows):
                item = self.annotations_table.item(row, 1)
                if item is not None:
                    updated_classifications.append(item.text())
                elif row < len(self._annotations_data["classifications"]):
                    updated_classifications.append(self._annotations_data["classifications"][row])
                else:
                    updated_classifications.append("new_annotation")

                if row < len(self._annotations_data["colours"]):
                    rgb = self._annotations_data["colours"][row]
                else:
                    rgb = [128, 128, 128]
                updated_colours_rgb.append([int(rgb[0]), int(rgb[1]), int(rgb[2])])

            if data is not None:
                if len(data) != n_rows:
                    if len(data) < n_rows:
                        n_new = n_rows - len(data)
                        new_geoms = []
                        if hasattr(self, "_pending_annotation_base_df"):
                            new_geoms = self._pending_annotation_base_df["geometry"].iloc[len(data):len(data) + n_new].tolist()
                        if len(new_geoms) < n_new or any(g is None for g in new_geoms):
                            self._collect_pending_geometries()
                            if hasattr(self, "_pending_annotation_base_df"):
                                new_geoms = self._pending_annotation_base_df["geometry"].iloc[len(data):len(data) + n_new].tolist()
                        if len(new_geoms) < n_new:
                            show_info(
                                "Some new annotation rows have no drawn shape yet. "
                                "Draw the new shapes in napari, then save again."
                            )
                            new_geoms = new_geoms + [None] * (n_new - len(new_geoms))
                        import geopandas as gpd
                        extra = gpd.GeoDataFrame(
                            {"geometry": new_geoms}, crs=getattr(data, "crs", None)
                        )
                        data = pd.concat([data, extra], ignore_index=True)
                    else:
                        data = data.iloc[:n_rows].reset_index(drop=True)

                data = data.copy()
                categories = list(dict.fromkeys(updated_classifications))
                if "classification" in data.columns and pd.api.types.is_categorical_dtype(data["classification"]):
                    for label in data["classification"].cat.categories:
                        if label not in categories:
                            categories.append(label)

                data["classification"] = pd.Categorical(
                    updated_classifications,
                    categories=categories,
                )
                data["colour"] = [list(rgb) for rgb in updated_colours_rgb]
                data["id"] = updated_ids
                if "objectType" in data.columns:
                    obj_types = list(data["objectType"].astype(str).tolist())
                    obj_types = obj_types[:len(data)] + ["annotation"] * max(0, n_rows - len(obj_types))
                else:
                    obj_types = ["annotation"] * n_rows
                data["objectType"] = obj_types

                try:
                    self.sdata.shapes["Annotations"] = data
                except Exception:
                    try:
                        annotations["classification"] = pd.Categorical(
                            updated_classifications,
                            categories=categories,
                        )
                        annotations["colour"] = [list(rgb) for rgb in updated_colours_rgb]
                        annotations["id"] = updated_ids
                        annotations["objectType"] = obj_types
                    except Exception as e:
                        show_info(f"Could not write back annotations table: {e}")
                        return
            else:
                if not hasattr(annotations, "attrs"):
                    annotations.attrs = {}
                annotations.attrs["classification"] = updated_classifications
                annotations.attrs["colour"] = [list(rgb) for rgb in updated_colours_rgb]
                annotations.attrs["id"] = updated_ids
                annotations.attrs["objectType"] = ["annotation"] * n_rows

            self._annotations_data["ids"] = updated_ids
            self._annotations_data["classifications"] = updated_classifications
            self._annotations_data["colours"] = updated_colours_rgb

            from napari.layers import Shapes
            for layer in self.viewer.layers:
                if isinstance(layer, Shapes) and layer.name == "Annotations":
                    if updated_colours_rgb:
                        _apply_direct_shapes_colors(layer, updated_colours_rgb)
                    break

            show_info("Annotations saved successfully.")
            self._pending_annotation_base_df = None
            self._pending_annotation_draw_start = None
            self._pending_annotation_original_count = None
            self._pending_annotations_registered = False

            if hasattr(self, "meta_shapes_combo") and self.meta_shapes_combo.currentText() == "Annotations":
                self._populate_meta_columns(self.meta_col_combo, "Annotations")

        except Exception as e:
            show_info(f"Error saving annotations: {e}")
    

    def _on_pick_new_annotation_colour(self):
        rgb = self._new_annotation_colour
        color = QColorDialog.getColor(
            QColor(int(rgb[0]), int(rgb[1]), int(rgb[2])),
            self,
            "Choose colour for new annotation",
        )
        if color.isValid():
            r, g, b, _ = color.getRgb()
            self._new_annotation_colour = [r, g, b]
            self._new_annotation_colour_btn.setStyleSheet(
                f"background-color: rgb({r}, {g}, {b}); "
                f"border: 1px solid {PALETTE['border']};"
            )

    def _on_add_annotation(self):
        """
        Append a pending annotation row with label and colour, and reserve a
        placeholder in the stored annotations GeoDataFrame. The actual geometry
        is registered later from the live napari layer.
        """
        if not hasattr(self, "_annotations_data"):
            self._annotations_data = {"ids": [], "classifications": [], "colours": []}

        label = self.new_annotation_name.toPlainText().strip()
        if not label:
            show_info("Enter a label for the new annotation.")
            return

        rgb = list(self._new_annotation_colour)

        layer = _find_shapes_layer(self.viewer, "Annotations")
        layer_count = len(layer.data) if layer is not None else 0
        if getattr(self, "_pending_annotation_draw_start", None) is None:
            self._pending_annotation_draw_start = layer_count

        annotations = self.sdata.shapes.get("Annotations", None)
        data = _shape_to_dataframe(annotations)
        if data is None:
            try:
                import geopandas as gpd
                data = gpd.GeoDataFrame(
                    {
                        "id": pd.Series(dtype=str),
                        "objectType": pd.Series(dtype=str),
                        "classification": pd.Series(dtype=str),
                        "colour": pd.Series(dtype=object),
                    },
                    geometry=pd.Series(dtype="geometry"),
                )
            except Exception:
                data = pd.DataFrame(
                    columns=["id", "objectType", "classification", "colour", "geometry"]
                )

        if getattr(self, "_pending_annotation_original_count", None) is None:
            self._pending_annotation_original_count = len(data)

        columns = list(data.columns)
        for required in ["id", "objectType", "classification", "colour", "geometry"]:
            if required not in columns:
                columns.append(required)

        new_row = {col: None for col in columns}
        new_id = str(uuid.uuid4())
        new_row.update({
            "id": new_id,
            "objectType": "annotation",
            "classification": label,
            "colour": [int(rgb[0]), int(rgb[1]), int(rgb[2])],
            "geometry": None,
        })
        new_row_df = pd.DataFrame([new_row], columns=columns)
        if hasattr(data, "geometry") and "geometry" in data.columns:
            try:
                import geopandas as gpd
                new_row_df = gpd.GeoDataFrame(new_row_df, geometry="geometry", crs=getattr(data, "crs", None))
            except Exception:
                pass

        self._pending_annotation_base_df = pd.concat([data, new_row_df], ignore_index=True)

        new_row_index = self.annotations_table.rowCount()
        self.annotations_table.insertRow(new_row_index)

        self._annotations_data["ids"].append(new_id)
        self._annotations_data["classifications"].append(label)
        self._annotations_data["colours"].append(rgb)

        id_item = QTableWidgetItem(new_id)
        id_item.setFlags(id_item.flags() & ~Qt.ItemIsEditable)
        self.annotations_table.setItem(new_row_index, 0, id_item)

        label_item = QTableWidgetItem(label)
        self.annotations_table.setItem(new_row_index, 1, label_item)

        color_widget = QWidget()
        color_layout = QHBoxLayout(color_widget)
        color_layout.setContentsMargins(2, 2, 2, 2)
        color_btn = QPushButton()
        color_btn.setMaximumWidth(60)
        color_btn.setStyleSheet(
            f"background-color: rgb({rgb[0]}, {rgb[1]}, {rgb[2]}); "
            f"border: 1px solid {PALETTE['border']}; border-radius: 3px;"
        )
        color_btn.clicked.connect(lambda checked, r=new_row_index: self._on_color_picker(r))
        color_layout.addWidget(color_btn)
        color_layout.addStretch()
        self.annotations_table.setCellWidget(new_row_index, 2, color_widget)

        self.new_annotation_name.clear()
        
        # Enable Step 2 for drawing
        if hasattr(self, "_step2_group"):
            self._step2_group.setEnabled(True)
        
        show_info(
            f"Added '{label}'. Now draw the corresponding shape on the "
            "'Annotations' layer in napari using the shape tools."
        )

    def save_to_sdata(
        self,
        layers: list[Layer] | None = None,
        spatial_element_name: str | None = None,
        table_name: str | None = None,
        table_columns: list[str] | None = None,
        overwrite: bool = False,
    ) -> None:
        """
        Add the current selected napari layer(s) to the SpatialData object.

        If the layer is newly added and not yet linked with a spatialdata object it will be automatically
        linked if only 1 spatialdata object is being visualized in the viewer.

        Notes
        -----
        Usage:

            - you can invoke this function by pressing Shift+E;
            - the selected layer (needs to be exactly one) will be saved;
            - if more than one SpatialData object is being shown with napari, before saving the layer you need to link
              it to a layer with a SpatialData object. This can be done by selecting both layers and pressing Shift+L.
            - Currently images and labels are not supported.
            - Currently updating existing elements is not supported.
        """
        selected_layers = layers if layers else self.viewer.layers.selection
        if len(selected_layers) != 1:
            raise ValueError("Only one layer can be saved at a time.")
        selected = list(selected_layers)[0]
        if "sdata" not in selected.metadata:
            sdatas = [(layer, layer.metadata["sdata"]) for layer in self.viewer.layers if "sdata" in layer.metadata]
            if len(sdatas) < 1:
                raise ValueError(
                    "No SpatialData layers found in the viewer. Layer cannot be linked to SpatialData object."
                )
            if len(sdatas) > 1 and not all(sdatas[0][1] is sdata[1] for sdata in sdatas[1:]):
                raise ValueError(
                    "Multiple different spatialdata object found in the viewer. Please link the layer to "
                    "one of them by selecting both the layer to save and the layer containing the SpatialData object "
                    "and then pressing Shift+L. Then select the layer to save and press Shift+E again."
                )
            # link the layer to the only sdata object
            self._inherit_metadata(self.viewer)
        assert selected.metadata["sdata"]

        # now we can save the layer since it is linked to a SpatialData object
        if isinstance(selected, Points):
            parsed, cs = self._save_points_to_sdata(selected, spatial_element_name, overwrite)
        elif isinstance(selected, Shapes):
            parsed, cs = self._save_shapes_to_sdata(selected, spatial_element_name, overwrite)
            if table_name:
                self._save_table_to_sdata(selected, table_name, spatial_element_name, table_columns, overwrite)
        elif isinstance(selected, Image | Labels):
            raise NotImplementedError
        else:
            raise ValueError(f"Layer of type {type(selected)} cannot be saved.")

        self.layer_names.add(selected.name)
        self._layer_event_caches[selected.name] = []
        self._update_metadata(selected, parsed)
        selected.events.data.connect(self._update_cache_indices)
        selected.events.name.connect(self._validate_name)
        self.layer_saved.emit(cs)
        show_info("Layer saved")
    

    def _on_register_annotation(self):
        if not hasattr(self, "_pending_annotation_base_df"):
            show_info("No pending annotations to register. Add an annotation first.")
            return

        layer = _find_shapes_layer(self.viewer, "Annotations")
        if layer is None:
            show_info("Annotations layer not found in napari.")
            return

        start = getattr(self, "_pending_annotation_draw_start", None)
        if start is None:
            show_info("No reference draw state found. Add an annotation first.")
            return

        total_new = len(self._pending_annotation_base_df) - getattr(self, "_pending_annotation_original_count", 0)
        if total_new <= 0:
            show_info("No pending new annotations to register.")
            return

        from shapely.geometry import Polygon

        geoms = []
        for coords in layer.data[start:start + total_new]:
            try:
                coords_arr = np.asarray(coords, dtype=float)
                if coords_arr.ndim == 2 and coords_arr.shape[1] == 2:
                    geoms.append(Polygon(coords_arr[:, ::-1]))
                else:
                    geoms.append(None)
            except Exception:
                geoms.append(None)

        if len(geoms) != total_new:
            show_info(
                "Could not register all new geometries. Draw the new region(s) "
                "on the Annotations layer, then try Register Annotation again."
            )
            return

        if "geometry" not in self._pending_annotation_base_df.columns:
            self._pending_annotation_base_df["geometry"] = None

        self._pending_annotation_base_df.loc[
            self._pending_annotation_original_count:, "geometry"
        ] = geoms
        self._pending_annotations_registered = True

        sdata_viewer = self._get_sdata_viewer()
        if sdata_viewer is not None:
            try:
                # Select the layer (mirrors what the user does manually before Shift+E)
                self.viewer.layers.selection.active = layer
                self.viewer.layers.selection = {layer}

                # Ensure the layer is linked to the SpatialData object (Shift+L equivalent)
                if "sdata" not in layer.metadata:
                    sdata_viewer._inherit_metadata(self.viewer)

                # Same call Shift+E triggers
                sdata_viewer.save_to_sdata([layer])
                show_info("New annotation geometry registered and saved to SpatialData.")
            except Exception as e:
                show_info(f"Geometry registered, but save_to_sdata failed: {e}")
        else:
            show_info(
                "New annotation geometry registered. Could not find the napari-spatialdata "
                "viewer widget; press Shift+E manually to save."
            )

    def _get_sdata_viewer(self):
        """Return the napari_spatialdata SpatialDataViewer instance, if available."""
        interactive = _GOATPY_REFS.get("interactive")
        if interactive is None:
            return None
        # napari-spatialdata stores the SpatialDataViewer (QObject) on the dock widget
        for attr in ("_sdata_widget", "_widget"):
            widget = getattr(interactive, attr, None)
            if widget is not None:
                viewer_model = getattr(widget, "_viewer_model", None) or getattr(widget, "viewer_model", None)
                if viewer_model is not None:
                    return viewer_model
        return None

    def _collect_pending_geometries(self):
        """Pull newly-drawn annotation geometries from the live 'Annotations' layer."""
        if not hasattr(self, "_pending_annotation_base_df"):
            return
        layer = _find_shapes_layer(self.viewer, "Annotations")
        if layer is None:
            return
        start = getattr(self, "_pending_annotation_draw_start", None)
        if start is None:
            return
        total_new = len(self._pending_annotation_base_df) - getattr(self, "_pending_annotation_original_count", 0)
        if total_new <= 0:
            return

        from shapely.geometry import Polygon

        geoms = []
        for coords in layer.data[start:start + total_new]:
            try:
                coords_arr = np.asarray(coords, dtype=float)
                if coords_arr.ndim == 2 and coords_arr.shape[1] == 2:
                    geoms.append(Polygon(coords_arr[:, ::-1]))
                else:
                    geoms.append(None)
            except Exception:
                geoms.append(None)

        if len(geoms) != total_new:
            return

        if "geometry" not in self._pending_annotation_base_df.columns:
            self._pending_annotation_base_df["geometry"] = None

        self._pending_annotation_base_df.loc[
            self._pending_annotation_original_count:, "geometry"
        ] = geoms
        self._pending_annotations_registered = True


    # ── Stats tab ─────────────────────────────────────────────────────────

    def _build_stats_tab(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setContentsMargins(4, 4, 4, 4)
        plot_btn = QPushButton("Refresh Statistics")
        plot_btn.clicked.connect(self._draw_stats)
        layout.addWidget(plot_btn)
        self.stats_canvas = MplCanvas(w, width=4.5, height=5.5, dpi=90)
        layout.addWidget(self.stats_canvas)
        QTimer.singleShot(400, self._draw_stats)
        return w

    def _draw_stats(self):
        adata = self._adata()
        X = np.asarray(adata.X, dtype=np.float32)
        self.stats_canvas.fig.clear()
        axes = self.stats_canvas.fig.subplots(2, 2)
        for ax in axes.flat:
            self.stats_canvas._style_ax(ax)

        axes[0, 0].hist(X.sum(axis=1), bins=50,
                        color=PALETTE["accent"], edgecolor="none", alpha=0.8)
        axes[0, 0].set_title("TIC distribution", fontsize=8)

        axes[0, 1].hist((X > 0).sum(axis=1), bins=40,
                        color=PALETTE["accent2"], edgecolor="none", alpha=0.8)
        axes[0, 1].set_title("Peaks / pixel", fontsize=8)

        axes[1, 0].hist((X > 0).mean(axis=0) * 100, bins=40,
                        color=PALETTE["success"], edgecolor="none", alpha=0.8)
        axes[1, 0].set_title("Peak frequency (%)", fontsize=8)

        cluster_drawn = False
        for col in ("GPCA_clusters", "leiden", "batch", "annotation"):
            if col in adata.obs.columns:
                cats = adata.obs[col].astype(str)
                counts = cats.value_counts().sort_index()
                cmap = plt.get_cmap("tab20", len(counts))
                axes[1, 1].bar(range(len(counts)), counts.values,
                               color=[cmap(i) for i in range(len(counts))],
                               edgecolor="none", alpha=0.85)
                axes[1, 1].set_xticks(range(len(counts)))
                axes[1, 1].set_xticklabels(counts.index, rotation=45, ha="right", fontsize=6)
                axes[1, 1].set_title(f"Cluster sizes ({col})", fontsize=8)
                cluster_drawn = True
                break
        if not cluster_drawn:
            axes[1, 1].text(0.5, 0.5, "No cluster\ncolumn found",
                            ha="center", va="center", transform=axes[1, 1].transAxes,
                            color=PALETTE["text_dim"])

        for ax in axes.flat:
            ax.tick_params(labelsize=6.5)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        self.stats_canvas.fig.suptitle(
            f"{adata.n_obs:,} pixels × {adata.n_vars:,} peaks",
            fontsize=8.5, color=PALETTE["text"], y=1.01,
        )
        self.stats_canvas.fig.tight_layout(pad=0.5)
        self.stats_canvas.draw()



# ════════════════════════════════════════════════════════════════════════════
# Post-launch window setup
# ════════════════════════════════════════════════════════════════════════════


def _configure_viewer_window(viewer, screen_height: int = None):
    from qtpy.QtCore import QTimer

    def _setup():
        win = viewer.window._qt_window
        
        _install_info_bar(viewer)

        # ── 1. Detect real available screen size ──────────────────────────
        app = QApplication.instance()
        if screen_height is not None:
            avail_h = screen_height
        elif app is not None:
            screen = app.primaryScreen()
            avail_h = screen.availableGeometry().height()
        else:
            avail_h = 900  # safe fallback

        h = int(avail_h * 0.88)   # leave room for macOS menu bar + dock
        w = int(h * 16 / 9)
        win.resize(w, h)

        # ── 2. Hide napari-spatialdata docks we don't need ────────────────
        from qtpy.QtWidgets import QDockWidget
        _hidden_docks = {"View (napari-spatialdata)", "slider", "colorbar"}
        for qdock in win.findChildren(QDockWidget):
            if qdock.windowTitle() in _hidden_docks:
                qdock.hide()

        # ── 3. Fix vertical splitter so spectrum dock is visible ──────────
        from qtpy.QtWidgets import QSplitter
        from qtpy.QtCore import Qt
        for sp in win.findChildren(QSplitter):
            if sp.orientation() == Qt.Vertical:
                sizes = sp.sizes()
                total = sum(sizes)
                if total > 100 and len(sizes) >= 2:
                    # On small screens (< 900px), give spectrum less room
                    frac = 0.22 if avail_h < 900 else 0.28
                    bottom = int(total * frac)
                    top = total - bottom
                    new_sizes = [top] + [bottom] * (len(sizes) - 1)
                    sp.setSizes(new_sizes)
                    break

    QTimer.singleShot(500, _setup)




# ════════════════════════════════════════════════════════════════════════════
# Tips dialog
# ════════════════════════════════════════════════════════════════════════════

class TipsDialog(QDialog):
    """Popup explaining tooltips/usage for each goatpy widget."""

    TIPS = {
        "Spectrum Widget": [
            ("Scroll",            "Pan left/right across the m/z axis"),
            ("Ctrl + Scroll",     "Zoom in/out around the cursor position"),
            ("Click red line",    "Select that curated glycan peak — updates the H&E viewer and Analysis sidebar"),
            ("Zoom to: Go",       "Jump to a specific m/z range"),
            ("Reset",             "Restore the full m/z range view"),
            ("Show applied tol.", "Overlay a ± window around each curated peak showing the extraction tolerance"),
            ("Check unreg. peak", "Enter free-click mode: click anywhere on the spectrum to pick an arbitrary m/z"),
            ("Display spatially", "Render the selected unregistered m/z as an ion map on the H&E viewer"),
            ("Add peak to list",  "Append the selected m/z to an in-session list for later export"),
            ("Export list",       "Save the accumulated m/z list to a CSV file"),
        ],
        "Glycan Tab": [
            ("Select Glycan",     "Choose a curated glycan/peak from the dropdown (supports free-text search)"),
            ("Show on H&E Viewer","Colour the 'pixels' Shapes layer by intensity for the selected glycan"),
            ("Ion map colormap",  "Change the continuous colormap applied to the ion map"),
            ("Violin / Box",      "Plot intensity distribution of the selected glycan grouped by a metadata column"),
            ("Histogram",         "Plot a histogram of per-pixel intensities for the selected glycan"),
        ],
        "Metadata Tab": [
            ("Shapes layer",      "Choose which SpatialData shapes element to colour"),
            ("Metadata column",   "Column from obs (for 'pixels') or the shapes GeoDataFrame to visualise"),
            ("Type indicator",    "Shows whether the column is detected as categorical or continuous"),
            ("Colormap",          "Categorical columns use discrete palettes; continuous use sequential ones"),
            ("Show on H&E Viewer","Apply the selected column's values as colours on the chosen shapes layer"),
        ],
        "UMAP Tab": [
            ("Colour by",         "Colour UMAP points by a metadata column or a glycan's per-pixel intensity"),
            ("Embedding",         "Select which obsm key to use (e.g. X_umap, X_pca)"),
            ("Plot UMAP",         "Render the 2-D embedding with the chosen colouring"),
        ],
        "Heatmap Tab": [
            ("Group by",          "Categorical obs column used to average intensities per group"),
            ("Top N peaks",       "Number of highest-variance peaks to display (default 30)"),
            ("Normalise",         "z-score, min-max, or raw mean intensity"),
            ("Select Glycans…",   "Pick a custom subset of glycans by name, m/z, or file upload"),
            ("Clear Selection",   "Revert to automatic top-N peak selection"),
        ],
        "Annotations Tab": [
            ("Table",             "Edit classification labels and colours for existing annotations"),
            ("Colour picker",     "Click the colour swatch in a row to change that annotation's colour"),
            ("Construct Annot.",  "Add a new annotation row with a label and colour — then draw the region in napari"),
            ("Finish Annotation", "Register the shape you just drew and link it to the pending annotation row"),
            ("Save Changes",      "Write all label/colour edits back to sdata.shapes['Annotations']"),
            ("Reload",            "Refresh the table from the current sdata.shapes['Annotations'] state"),
        ],
        "Stats Tab": [
            ("TIC distribution",  "Histogram of total ion current (sum of all peaks) per pixel"),
            ("Peaks / pixel",     "Number of non-zero peaks detected per pixel"),
            ("Peak frequency",    "Percentage of pixels in which each peak is non-zero"),
            ("Cluster sizes",     "Bar chart of cell counts per cluster (uses first available cluster column)"),
            ("Refresh",           "Recompute all four panels from the current adata state"),
        ],
        "Napari Viewer": [
            ("Layer list",        "Toggle layer visibility with the eye icon; drag to reorder"),
            ("Layer controls",    "Opacity, colormap, and contrast controls for the selected layer"),
            ("Scroll (canvas)",   "Zoom the spatial view"),
            ("Click + drag",      "Pan the spatial view"),
            ("Shapes tools",      "Rectangle / polygon tools for drawing annotation regions"),
            ("Window menu",       "Show/hide Spectrum, Analysis, Layer Controls, and Layer List panels"),
        ],
    }

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("goatpy — Information & Tips")
        self.setMinimumWidth(640)
        self.setMinimumHeight(500)
        self.setStyleSheet(BASE_STYLE)
        self._build_ui()

    def _build_ui(self):
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        # ── Add logo (optional) ───────────────────────────────────
        try:
            from importlib.resources import files

            logo_path = files("goatpy.assets").joinpath("logo.png")
            pixmap = QPixmap(str(logo_path))

            if not pixmap.isNull():
                logo = QLabel()
                logo.setPixmap(
                    pixmap.scaledToWidth(
                        180,
                        Qt.SmoothTransformation,
                    )
                )
                logo.setAlignment(Qt.AlignCenter)
                layout.addWidget(logo)

        except Exception:
            # No logo available – continue with the normal dialog
            pass

        # ── Existing header ───────────────────────────────────────
        header = QLabel("Information & Tips")
        header.setStyleSheet(
            f"font-size: 16px; font-weight: bold; "
            f"color: {PALETTE['accent']}; padding-bottom: 4px;"
        )
        header.setAlignment(Qt.AlignCenter)
        layout.addWidget(header)

        sub = QLabel("Click a section header to expand its tips.")
        sub.setStyleSheet(f"color: {PALETTE['text_dim']}; font-size: 10px;")
        layout.addWidget(sub)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border: none; }")
        container = QWidget()
        container_layout = QVBoxLayout(container)
        container_layout.setContentsMargins(0, 0, 0, 0)
        container_layout.setSpacing(4)

        for section, entries in self.TIPS.items():
            container_layout.addWidget(self._make_section(section, entries))

        container_layout.addStretch()
        scroll.setWidget(container)
        layout.addWidget(scroll)

        close_btn = QPushButton("Close")
        close_btn.setFixedWidth(100)
        close_btn.clicked.connect(self.accept)
        row = QHBoxLayout()
        row.addStretch()
        row.addWidget(close_btn)
        layout.addLayout(row)

    def _make_section(self, title: str, entries: list[tuple[str, str]]) -> QWidget:
        """Collapsible section: clicking the header toggles the body."""
        wrapper = QWidget()
        wrapper_layout = QVBoxLayout(wrapper)
        wrapper_layout.setContentsMargins(0, 0, 0, 0)
        wrapper_layout.setSpacing(0)

        # Header button (acts as toggle)
        header_btn = QPushButton(f"▶  {title}")
        header_btn.setCheckable(True)
        header_btn.setChecked(False)
        header_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {PALETTE['surface']};
                color: {PALETTE['text']};
                border: 1px solid {PALETTE['border']};
                border-radius: 4px;
                padding: 7px 12px;
                font-weight: bold;
                font-size: 11px;
                text-align: left;
            }}
            QPushButton:checked {{
                background-color: {PALETTE['border']};
                color: {PALETTE['accent']};
                border-color: {PALETTE['accent']};
            }}
            QPushButton:hover {{
                background-color: {PALETTE['border']};
            }}
        """)

        # Body table
        body = QWidget()
        body.setVisible(False)
        body_layout = QVBoxLayout(body)
        body_layout.setContentsMargins(8, 4, 8, 8)
        body_layout.setSpacing(0)

        table = QTableWidget(len(entries), 2)
        table.setHorizontalHeaderLabels(["Action / Control", "Description"])
        table.horizontalHeader().setStretchLastSection(True)
        table.horizontalHeader().setStyleSheet(
            f"QHeaderView::section {{ background-color: {PALETTE['bg']}; "
            f"color: {PALETTE['accent']}; font-weight: bold; border: none; padding: 4px; }}"
        )
        table.verticalHeader().setVisible(False)
        table.setEditTriggers(QTableWidget.NoEditTriggers)
        table.setSelectionMode(QTableWidget.NoSelection)
        table.setShowGrid(False)
        table.setAlternatingRowColors(True)
        table.setStyleSheet(f"""
            QTableWidget {{
                background-color: {PALETTE['bg']};
                alternate-background-color: {PALETTE['surface']};
                border: none;
                gridline-color: {PALETTE['border']};
            }}
            QTableWidget::item {{ padding: 5px 8px; color: {PALETTE['text']}; border: none; }}
        """)
        table.setColumnWidth(0, 180)
        table.setMinimumHeight(len(entries) * 30 + 30)

        for row_i, (action, desc) in enumerate(entries):
            action_item = QTableWidgetItem(action)
            action_item.setForeground(QColor(PALETTE["highlight"]))
            desc_item = QTableWidgetItem(desc)
            table.setItem(row_i, 0, action_item)
            table.setItem(row_i, 1, desc_item)

        body_layout.addWidget(table)
        wrapper_layout.addWidget(header_btn)
        wrapper_layout.addWidget(body)

        def _toggle(checked, btn=header_btn, b=body):
            b.setVisible(checked)
            btn.setText(("▼  " if checked else "▶  ") + title)

        header_btn.toggled.connect(_toggle)
        return wrapper


# ════════════════════════════════════════════════════════════════════════════
# Info bar (replaces napari status bar)
# ════════════════════════════════════════════════════════════════════════════

from qtpy.QtCore import QObject, QEvent

class _StatusBarTextBlocker(QObject):
    """
    Event filter installed on the QStatusBar that intercepts any attempt
    by napari to write keybinding hint text into the bar's message area.
    """
    def eventFilter(self, obj, event):
        # Block StatusTip and WhatsThis events that carry hint text
        if event.type() in (QEvent.StatusTip, QEvent.WhatsThis):
            return True
        return super().eventFilter(obj, event)


def _install_info_bar(viewer) -> None:
    from qtpy.QtWidgets import QStatusBar, QLabel, QPushButton
    from qtpy.QtCore import QObject, QEvent, QTimer

    win = viewer.window._qt_window
    status: QStatusBar = win.statusBar()

    # ── 1. Nuke the hint label and block all future text ──────────────────

    # Hide any existing child labels
    for widget in status.findChildren(QLabel):
        widget.hide()
        widget.setFixedWidth(0)

    # Override showMessage and clearMessage at the Python level
    def _noop(*args, **kwargs):
        pass
    status.showMessage = _noop

    # Install event filter to catch lower-level status tip events
    blocker = _StatusBarTextBlocker(status)
    status.installEventFilter(blocker)
    win.installEventFilter(blocker)   # also block at window level

    # Napari writes the hint text via a QTimer after tool changes —
    # repeatedly clear the message for the first few seconds after launch
    def _clear():
        # Call the real C++ clearMessage, bypassing our noop override
        QStatusBar.clearMessage(status)
        # Also hide any labels that reappeared
        for w in status.findChildren(QLabel):
            if w.isVisible() and w.text():
                w.hide()
                w.setFixedWidth(0)

    # Fire several times to catch delayed napari hint refreshes
    for delay_ms in (800, 1200, 1800, 2500, 3500):
        QTimer.singleShot(delay_ms, _clear)

    # Keep a reference so the blocker isn't garbage collected
    _GOATPY_REFS["_status_blocker"] = blocker

    # ── 2. Insert Tips button on the left ─────────────────────────────────
    tips_btn = QPushButton("ℹ  Information / Tips")
    tips_btn.setFixedHeight(20)
    tips_btn.setStyleSheet(f"""
        QPushButton {{
            background-color: transparent;
            color: {PALETTE['accent']};
            border: 1px solid {PALETTE['accent']};
            border-radius: 3px;
            padding: 0 10px;
            font-size: 10px;
            font-weight: bold;
        }}
        QPushButton:hover {{ background-color: {PALETTE['border']}; }}
        QPushButton:pressed {{ background-color: {PALETTE['accent']}; color: white; }}
    """)
    tips_btn.clicked.connect(lambda: TipsDialog(win).exec_())

    status.insertPermanentWidget(0, tips_btn)
    status.show()







# ════════════════════════════════════════════════════════════════════════════
# 3. MAIN LAUNCH FUNCTION
# ════════════════════════════════════════════════════════════════════════════

def launch_goatpy_gui(
    sdata: SpatialData,
    peaks: Optional[list[float]] = None,
    glycan_csv: Optional[str] = None,
    table_name: str = "maldi_adata",
    viewer: Optional[napari.Viewer] = None,
    applied_tolerance: float = 0.1,
    screen_height: int = 1080
) -> napari.Viewer:

    if peaks is None:
        try:
            peaks = list(_resolve_mz_array(sdata.tables[table_name]))
        except Exception:
            peaks = []

    glycan_df = None
    if glycan_csv:
        try:
            raw = pd.read_csv(glycan_csv)
            raw.columns = [c.strip() for c in raw.columns]
            glycan_df = raw
        except Exception as e:
            print(f"[goatpy GUI] glycan CSV error: {e}")

    # ── Load layers via napari-spatialdata (handles aligned CS correctly) ──
    interactive = _add_spatialdata_layers(
        viewer=None,          # not used anymore — Interactive owns its viewer
        sdata=sdata,
        target_cs="aligned",
    )

    # Get the actual napari Viewer from Interactive
    viewer = interactive._viewer   # napari_spatialdata stores it here

    viewer.title = "goatpy — Spatial Glycomics Analysis"

    # ── Keep Interactive alive (don't let it be GC'd) ─────────────────────
    _GOATPY_REFS["interactive"] = interactive

    # In launch_goatpy_gui, replace the two add_dock_widget calls:

    # ── Spectrum widget ────────────────────────────────────────────────────────
    spectrum_widget = SpectrumWidget(
        sdata=sdata, peaks=peaks, glycan_df=glycan_df, table_name=table_name,
        applied_tolerance=applied_tolerance,
    )
    spectrum_dock = viewer.window.add_dock_widget(
        spectrum_widget,
        area="bottom",
        name="Spectrum",        # this name appears in the Window menu
    )

    # ── Sidebar widget ─────────────────────────────────────────────────────────
    sidebar = AnalysisSidebar(
        sdata=sdata, peaks=peaks, viewer=viewer,
        glycan_df=glycan_df, table_name=table_name,
    )
    analysis_dock = viewer.window.add_dock_widget(
        sidebar,
        area="right",
        name="Analysis",        # this name appears in the Window menu
    )

    # ── Make them appear in Window menu with toggle actions ────────────────────
    def _register_dock_in_window_menu(viewer, dock, name: str):
        """
        Ensures the dock widget appears in napari's Window menu
        as a checkable toggle action, matching how napari registers
        its own built-in panels (Layer Controls, Layer List, etc.).
        """
        from qtpy.QtWidgets import QDockWidget, QMenuBar, QMenu
        from qtpy.QtGui import QAction

        # napari wraps our widget in a QDockWidget — find it
        qt_dock = None
        if isinstance(dock, QDockWidget):
            qt_dock = dock
        else:
            # add_dock_widget may return the inner widget; walk up to the QDockWidget
            parent = getattr(dock, "parent", lambda: None)()
            while parent is not None:
                if isinstance(parent, QDockWidget):
                    qt_dock = parent
                    break
                parent = getattr(parent, "parent", lambda: None)()

        if qt_dock is None:
            return

        qt_dock.setObjectName(name)          # stable ID for Qt's save/restore

        # Find the Window menu in napari's menubar
        win = viewer.window._qt_window
        menubar: QMenuBar = win.menuBar()
        window_menu: QMenu = None
        for action in menubar.actions():
            if action.text().lower().strip("&") in ("window", "&window"):
                window_menu = action.menu()
                break

        if window_menu is None:
            return

        # Avoid duplicates — napari sometimes auto-adds it already
        existing_titles = {a.text() for a in window_menu.actions()}
        if name in existing_titles:
            return

        toggle_action = qt_dock.toggleViewAction()
        toggle_action.setText(name)
        window_menu.addAction(toggle_action)

    QTimer.singleShot(600, lambda: _register_dock_in_window_menu(viewer, spectrum_dock, "Spectrum"))
    QTimer.singleShot(600, lambda: _register_dock_in_window_menu(viewer, analysis_dock, "Analysis"))


    
    # ── Wiring ─────────────────────────────────────────────────────────────
    sidebar.glycan_selected.connect(
        lambda mz, lbl: spectrum_widget.highlight_glycan(mz, lbl)
    )
    sidebar.show_annotated_on_spectra.connect(spectrum_widget.set_annotated_highlights)
    sidebar.nomenclature_updated.connect(spectrum_widget.update_peak_labels)

    spectrum_widget.peak_clicked.connect(sidebar.select_peak_from_spectrum)
    spectrum_widget.unregistered_peak_display.connect(sidebar.display_unregistered_ion_image)

    print(
        f"[goatpy GUI] Ready\n"
        f"{len(sdata.tables[table_name])} pixels · {len(peaks)} peaks"
    )

    _configure_viewer_window(viewer, screen_height=screen_height)

    # In launch_goatpy_gui, after _configure_viewer_window(viewer):
    _GOATPY_REFS["info_bar"] = None   # filled by _install_info_bar inside _setup

    # Optional helper to push a message from anywhere in the app:
    def post_info(text: str):
        bar = _GOATPY_REFS.get("info_bar")
        if bar is not None:
            bar.set_message(text)

    return viewer


