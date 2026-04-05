import spatialdata
from spatialdata import SpatialData
import napari
from napari.utils.notifications import show_info
import numpy as np
from spatialdata import SpatialData
from spatialdata.models import PointsModel
import pandas as pd
from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QListWidget, QMessageBox, QComboBox, QGroupBox
)
from qtpy.QtCore import Qt

from spatialdata.transformations import (
    align_elements_using_landmarks,
    get_transformation_between_landmarks,
)

from spatialdata.transformations import (
    Identity,
    BaseTransformation,
    Sequence,
    get_transformation,
    set_transformation,
)


def postpone_transformation(
        sdata: SpatialData,
        transformation: BaseTransformation,
        source_coordinate_system: str,
        target_coordinate_system: str,
    ):
        for element_type, element_name, element in sdata._gen_elements():
            old_transformations = get_transformation(element, get_all=True)
            if source_coordinate_system in old_transformations:
                old_transformation = old_transformations[source_coordinate_system]
                sequence = Sequence([old_transformation, transformation])
                set_transformation(element, sequence, target_coordinate_system)


def _get_scale_levels(image) -> dict:
    """
    Inspect a SpatialData image element and return available scale levels.

    Returns a dict like: {'scale0': (3, 76899, 109479), 'scale1': (3, 19224, 27369), ...}
    """
    levels = {}
    try:
        if hasattr(image, 'ds'):
            # It's a DataTree — try scale0, scale1, ...
            for i in range(10):
                key = f"scale{i}"
                try:
                    node = image[key]
                    ds = node.ds
                    if ds and len(ds.data_vars) > 0:
                        var = list(ds.data_vars)[0]
                        levels[key] = tuple(ds[var].shape)
                except (KeyError, AttributeError):
                    break
        elif hasattr(image, 'values'):
            levels['scale0'] = tuple(image.shape)
        else:
            levels['scale0'] = tuple(np.array(image).shape)
    except Exception:
        levels['scale0'] = ('unknown',)
    return levels


def _extract_image_at_scale(image, scale_level: str = 'scale0') -> np.ndarray:
    """Extract numpy array from a SpatialData image at a specific scale level."""
    try:
        if hasattr(image, 'ds') and hasattr(image, scale_level):
            node = image[scale_level]
            ds = node.ds
            var = list(ds.data_vars)[0]
            data = ds[var].values
        elif hasattr(image, 'ds'):
            ds = image.ds
            if len(ds.data_vars) > 0:
                var = list(ds.data_vars)[0]
                data = ds[var].values
            else:
                raise ValueError("DataTree has no data variables")
        elif hasattr(image, 'to_dataset'):
            ds = image.to_dataset()
            var = list(ds.data_vars)[0]
            data = ds[var].values
        elif hasattr(image, 'values'):
            data = image.values
        elif hasattr(image, 'data'):
            data = image.data.compute() if hasattr(image.data, 'compute') else image.data
        else:
            data = np.array(image)
        return data
    except Exception as e:
        print(f"Image type: {type(image)}, attributes: {dir(image)}")
        raise ValueError(f"Could not extract image data at {scale_level}: {e}")


def _scale_factor_for_level(image, scale_level: str) -> float:
    """
    Compute the spatial scale factor between scale_level and scale0 (full resolution).
    For example, if scale0 is (3, 76899, 109479) and scale1 is (3, 19224, 27369),
    the factor is ~4.0 meaning landmark coords must be multiplied by ~4 to get full-res coords.
    """
    levels = _get_scale_levels(image)
    if scale_level == 'scale0' or 'scale0' not in levels or scale_level not in levels:
        return 1.0
    full_shape = levels['scale0']
    this_shape = levels[scale_level]
    # Use spatial dims (last two for cyx layout)
    try:
        fy = full_shape[-2] / this_shape[-2]
        fx = full_shape[-1] / this_shape[-1]
        return (fy + fx) / 2.0  # average, usually identical
    except (IndexError, ZeroDivisionError):
        return 1.0


def align_image_using_landmarks(
    maldi_sd: SpatialData,
    he_sd: SpatialData,
    maldi_landmark_key: str = "maldi_landmarks",
    he_landmark_key: str = "he_landmarks",
    maldi_image_key: str = "optical_image",
    he_image_key: str = "he_image",
) -> SpatialData:
    """
    Align two SpatialData objects using landmarks.

    Landmarks saved by the GUI are always stored at full (scale0) resolution,
    regardless of which scale level was used for display. The scale correction
    is applied automatically inside the GUI's save_landmarks() method.

    Parameters
    ----------
    maldi_sd : SpatialData
        The reference SpatialData object.
    he_sd : SpatialData
        The moving SpatialData object to be aligned to the reference (H&E).
    maldi_landmark_key : str
        Key in maldi_sd.points for the reference landmarks.
    he_landmark_key : str
        Key in he_sd.points for the moving landmarks.
    maldi_image_key : str
        Key for the reference image in maldi_sd.images.
    he_image_key : str
        Key for the H&E image in he_sd.images.

    Returns
    -------
    maldi_sd : SpatialData
        The reference SpatialData with the H&E image merged in under the 'aligned'
        coordinate system.
    """
    affine = get_transformation_between_landmarks(
        references_coords=maldi_sd[maldi_landmark_key],
        moving_coords=he_sd[he_landmark_key]
    )

    affine = align_elements_using_landmarks(
        references_coords=maldi_sd[maldi_landmark_key],
        moving_coords=he_sd[he_landmark_key],
        reference_element=maldi_sd[maldi_image_key],
        moving_element=he_sd[he_image_key],
        reference_coordinate_system="global",
        moving_coordinate_system="global",
        new_coordinate_system="aligned",
    )

    postpone_transformation(
        sdata=he_sd,
        transformation=affine,
        source_coordinate_system="global",
        target_coordinate_system="aligned",
    )

    for _, _, element in maldi_sd._gen_elements():
        transforms = get_transformation(element, get_all=True)
        if "global" not in transforms:
            set_transformation(element, Identity(), "global")

    postpone_transformation(
        sdata=maldi_sd,
        transformation=Identity(),
        source_coordinate_system="global",
        target_coordinate_system="aligned",
    )

    #he_image = he_sd.images[he_image_key]
    #maldi_sd.images[he_image_key] = he_image

    for attr in ["images", "labels", "points", "shapes", "tables"]:
        source = getattr(he_sd, attr)
        target = getattr(maldi_sd, attr)
        for key, val in source.items():
            target[key] = val

    return maldi_sd


class LandmarkAlignmentWidget(QWidget):
    """Widget for selecting corresponding landmarks in two images."""

    def __init__(self, maldi_sd: SpatialData, he_sd: SpatialData,
                 maldi_viewer: napari.Viewer = None,
                 he_viewer: napari.Viewer = None,
                 combined_viewer: napari.Viewer = None,
                 maldi_image_key: str = "optical_image",
                 he_image_key: str = "he_image",
                 split_view: bool = False,
                 maldi_scale_level: str = 'scale0',
                 he_scale_level: str = 'scale0'):
        super().__init__()
        self.maldi_sd = maldi_sd
        self.he_sd = he_sd
        self.maldi_image_key = maldi_image_key
        self.he_image_key = he_image_key
        self.split_view = split_view

        # Scale levels used for display
        self.maldi_scale_level = maldi_scale_level
        self.he_scale_level = he_scale_level

        # Pre-compute scale factors so landmark coords can be mapped back to full res
        self.maldi_scale_factor = _scale_factor_for_level(
            maldi_sd.images[maldi_image_key], maldi_scale_level
        )
        self.he_scale_factor = _scale_factor_for_level(
            he_sd.images[he_image_key], he_scale_level
        )

        print(
            f"[LandmarkGUI] MALDI display level={maldi_scale_level}, "
            f"scale_factor={self.maldi_scale_factor:.3f}  |  "
            f"H&E display level={he_scale_level}, "
            f"scale_factor={self.he_scale_factor:.3f}"
        )

        # Store viewers
        if split_view:
            self.maldi_viewer = maldi_viewer
            self.he_viewer = he_viewer
            self.viewer = None
        else:
            self.viewer = combined_viewer
            self.maldi_viewer = None
            self.he_viewer = None

        self.maldi_landmarks = []  # stored at display resolution
        self.he_landmarks = []     # stored at display resolution
        self.active_dataset = "maldi"

        self.init_ui()
        self.setup_layers()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def init_ui(self):
        layout = QVBoxLayout()

        title = QLabel("<h2>Landmark Alignment Tool</h2>")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        # Scale level info box
        info_box = QGroupBox("Active Scale Levels")
        info_layout = QVBoxLayout()
        maldi_levels = _get_scale_levels(self.maldi_sd.images[self.maldi_image_key])
        he_levels = _get_scale_levels(self.he_sd.images[self.he_image_key])

        self.maldi_scale_label = QLabel(
            f"MALDI: <b>{self.maldi_scale_level}</b>  {maldi_levels.get(self.maldi_scale_level, '?')}"
        )
        self.he_scale_label = QLabel(
            f"H&E:   <b>{self.he_scale_level}</b>  {he_levels.get(self.he_scale_level, '?')}"
        )
        info_layout.addWidget(self.maldi_scale_label)
        info_layout.addWidget(self.he_scale_label)
        info_layout.addWidget(QLabel(
            "<i>Landmark coordinates will be automatically scaled to full resolution before saving.</i>"
        ))
        info_box.setLayout(info_layout)
        layout.addWidget(info_box)

        # Instructions
        instructions = QLabel(
            "1. Select which image to add landmarks to\n"
            "2. Click on the image to add landmarks\n"
            "3. Add corresponding points in both images\n"
            "4. Minimum 3 pairs recommended (5+ for best accuracy)\n"
            "5. Click 'Save Landmarks' when done"
        )
        instructions.setStyleSheet("padding: 10px; background-color: #f0f0f0; border-radius: 5px;")
        layout.addWidget(instructions)

        # Active dataset selector
        dataset_layout = QHBoxLayout()
        self.maldi_btn = QPushButton("Add MALDI Landmarks")
        self.maldi_btn.setCheckable(True)
        self.maldi_btn.setChecked(True)
        self.maldi_btn.clicked.connect(lambda: self.set_active_dataset("maldi"))
        self.maldi_btn.setStyleSheet("QPushButton:checked { background-color: #4CAF50; color: white; }")

        self.he_btn = QPushButton("Add H&E Landmarks")
        self.he_btn.setCheckable(True)
        self.he_btn.clicked.connect(lambda: self.set_active_dataset("he"))
        self.he_btn.setStyleSheet("QPushButton:checked { background-color: #2196F3; color: white; }")

        dataset_layout.addWidget(self.maldi_btn)
        dataset_layout.addWidget(self.he_btn)
        layout.addLayout(dataset_layout)

        # Landmark lists
        lists_layout = QHBoxLayout()

        maldi_list_layout = QVBoxLayout()
        maldi_label = QLabel("<b>MALDI Landmarks</b> (display coords)")
        maldi_label.setAlignment(Qt.AlignCenter)
        self.maldi_list = QListWidget()
        maldi_list_layout.addWidget(maldi_label)
        maldi_list_layout.addWidget(self.maldi_list)

        he_list_layout = QVBoxLayout()
        he_label = QLabel("<b>H&E Landmarks</b> (display coords)")
        he_label.setAlignment(Qt.AlignCenter)
        self.he_list = QListWidget()
        he_list_layout.addWidget(he_label)
        he_list_layout.addWidget(self.he_list)

        lists_layout.addLayout(maldi_list_layout)
        lists_layout.addLayout(he_list_layout)
        layout.addLayout(lists_layout)

        # Action buttons
        button_layout = QHBoxLayout()
        self.undo_btn = QPushButton("Undo Last")
        self.undo_btn.clicked.connect(self.undo_last)

        self.clear_btn = QPushButton("Clear All")
        self.clear_btn.clicked.connect(self.clear_all)

        self.save_btn = QPushButton("Save Landmarks")
        self.save_btn.clicked.connect(self.save_landmarks)
        self.save_btn.setStyleSheet("background-color: #FF9800; color: white; font-weight: bold;")

        button_layout.addWidget(self.undo_btn)
        button_layout.addWidget(self.clear_btn)
        button_layout.addWidget(self.save_btn)
        layout.addLayout(button_layout)

        self.status_label = QLabel("Status: Ready. Click to add landmarks.")
        self.status_label.setStyleSheet("padding: 5px; background-color: #e0e0e0;")
        layout.addWidget(self.status_label)

        self.setLayout(layout)

    # ------------------------------------------------------------------
    # Layer setup
    # ------------------------------------------------------------------

    def setup_layers(self):
        if self.split_view:
            self._setup_split_view()
        else:
            self._setup_combined_view()

    def _setup_combined_view(self):
        maldi_data = _extract_image_at_scale(
            self.maldi_sd.images[self.maldi_image_key], self.maldi_scale_level
        )
        self.viewer.add_image(maldi_data, name=f"MALDI ({self.maldi_scale_level})", colormap="viridis")

        he_data = _extract_image_at_scale(
            self.he_sd.images[self.he_image_key], self.he_scale_level
        )
        if he_data.ndim == 3 and he_data.shape[0] in [3, 4]:
            he_data = np.transpose(he_data, (1, 2, 0))
        self.viewer.add_image(he_data, name=f"H&E ({self.he_scale_level})",
                              rgb=he_data.ndim == 3)

        self.maldi_points_layer = self.viewer.add_points(
            np.empty((0, 2)), name="MALDI Landmarks",
            face_color="green", size=20, border_color="white", border_width=2
        )
        self.he_points_layer = self.viewer.add_points(
            np.empty((0, 2)), name="H&E Landmarks",
            face_color="blue", size=20, border_color="white", border_width=2
        )

        self.maldi_points_layer.mouse_drag_callbacks.append(self.on_click_combined)
        self.he_points_layer.mouse_drag_callbacks.append(self.on_click_combined)

        self.viewer.grid.enabled = True
        self.viewer.grid.shape = (1, 2)
        self.viewer.grid.stride = 1

    def _setup_split_view(self):
        maldi_data = _extract_image_at_scale(
            self.maldi_sd.images[self.maldi_image_key], self.maldi_scale_level
        )
        self.maldi_viewer.add_image(maldi_data, name=f"MALDI ({self.maldi_scale_level})",
                                    colormap="viridis")
        self.maldi_points_layer = self.maldi_viewer.add_points(
            np.empty((0, 2)), name="MALDI Landmarks",
            face_color="green", size=20, border_color="white", border_width=2
        )

        he_data = _extract_image_at_scale(
            self.he_sd.images[self.he_image_key], self.he_scale_level
        )
        if he_data.ndim == 3 and he_data.shape[0] in [3, 4]:
            he_data = np.transpose(he_data, (1, 2, 0))
        self.he_viewer.add_image(he_data, name=f"H&E ({self.he_scale_level})",
                                 rgb=he_data.ndim == 3)
        self.he_points_layer = self.he_viewer.add_points(
            np.empty((0, 2)), name="H&E Landmarks",
            face_color="blue", size=20, border_color="white", border_width=2
        )

        self.maldi_points_layer.mouse_drag_callbacks.append(
            lambda l, e: self.on_click_split(self.maldi_viewer, l, e)
        )
        self.he_points_layer.mouse_drag_callbacks.append(
            lambda l, e: self.on_click_split(self.he_viewer, l, e)
        )

    # ------------------------------------------------------------------
    # Interaction callbacks
    # ------------------------------------------------------------------

    def set_active_dataset(self, dataset: str):
        self.active_dataset = dataset
        if dataset == "maldi":
            self.maldi_btn.setChecked(True)
            self.he_btn.setChecked(False)
            self.status_label.setText("Status: Click on MALDI image to add landmarks")
        else:
            self.maldi_btn.setChecked(False)
            self.he_btn.setChecked(True)
            self.status_label.setText("Status: Click on H&E image to add landmarks")

    def on_click_combined(self, layer, event):
        if event.type == 'mouse_press' and event.button == 1:
            coords = layer.world_to_data(event.position)
            self._register_click(coords)

    def on_click_split(self, viewer, layer, event):
        if event.type == 'mouse_press' and event.button == 1:
            coords = layer.world_to_data(event.position)
            self._register_click(coords)

    def _register_click(self, coords):
        if self.active_dataset == "maldi":
            self.maldi_landmarks.append([coords[0], coords[1]])
            self.maldi_list.addItem(
                f"Point {len(self.maldi_landmarks)}: ({coords[0]:.1f}, {coords[1]:.1f})"
            )
            self.maldi_points_layer.data = np.array(self.maldi_landmarks)
            show_info(f"Added MALDI landmark {len(self.maldi_landmarks)}")
        else:
            self.he_landmarks.append([coords[0], coords[1]])
            self.he_list.addItem(
                f"Point {len(self.he_landmarks)}: ({coords[0]:.1f}, {coords[1]:.1f})"
            )
            self.he_points_layer.data = np.array(self.he_landmarks)
            show_info(f"Added H&E landmark {len(self.he_landmarks)}")
        self.update_status()

    def update_status(self):
        n_maldi = len(self.maldi_landmarks)
        n_he = len(self.he_landmarks)
        if n_maldi == n_he and n_maldi >= 3:
            self.status_label.setText(f"Status: {n_maldi} pairs — ready to save!")
            self.status_label.setStyleSheet("padding: 5px; background-color: #C8E6C9;")
        elif n_maldi == n_he:
            self.status_label.setText(f"Status: {n_maldi} pairs — need at least 3.")
            self.status_label.setStyleSheet("padding: 5px; background-color: #FFF9C4;")
        else:
            self.status_label.setText(
                f"Status: Unequal (MALDI: {n_maldi}, H&E: {n_he})"
            )
            self.status_label.setStyleSheet("padding: 5px; background-color: #FFCDD2;")

    def undo_last(self):
        if self.active_dataset == "maldi" and self.maldi_landmarks:
            self.maldi_landmarks.pop()
            self.maldi_list.takeItem(self.maldi_list.count() - 1)
            self.maldi_points_layer.data = (
                np.array(self.maldi_landmarks) if self.maldi_landmarks else np.empty((0, 2))
            )
            show_info("Removed last MALDI landmark")
        elif self.active_dataset == "he" and self.he_landmarks:
            self.he_landmarks.pop()
            self.he_list.takeItem(self.he_list.count() - 1)
            self.he_points_layer.data = (
                np.array(self.he_landmarks) if self.he_landmarks else np.empty((0, 2))
            )
            show_info("Removed last H&E landmark")
        self.update_status()

    def clear_all(self):
        reply = QMessageBox.question(
            self, 'Clear All', 'Clear all landmarks?',
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            self.maldi_landmarks = []
            self.he_landmarks = []
            self.maldi_list.clear()
            self.he_list.clear()
            self.maldi_points_layer.data = np.empty((0, 2))
            self.he_points_layer.data = np.empty((0, 2))
            self.update_status()
            show_info("All landmarks cleared")

    # ------------------------------------------------------------------
    # Save — scale coords back to full resolution
    # ------------------------------------------------------------------

    def save_landmarks(self):
        n_maldi = len(self.maldi_landmarks)
        n_he = len(self.he_landmarks)

        if n_maldi != n_he:
            QMessageBox.warning(self, 'Unequal Landmarks',
                                f'MALDI: {n_maldi}, H&E: {n_he} — must be equal.')
            return
        if n_maldi < 3:
            QMessageBox.warning(self, 'Too Few Landmarks',
                                f'Need at least 3 pairs (have {n_maldi}).')
            return

        try:
            # Scale display-resolution coords back to full (scale0) resolution
            maldi_pts = np.array(self.maldi_landmarks) * self.maldi_scale_factor
            he_pts    = np.array(self.he_landmarks)    * self.he_scale_factor

            maldi_df = pd.DataFrame({'x': maldi_pts[:, 1], 'y': maldi_pts[:, 0]})
            self.maldi_sd.points['maldi_landmarks'] = PointsModel.parse(maldi_df)

            he_df = pd.DataFrame({'x': he_pts[:, 1], 'y': he_pts[:, 0]})
            self.he_sd.points['he_landmarks'] = PointsModel.parse(he_df)

            QMessageBox.information(
                self, 'Success',
                f'Saved {n_maldi} landmark pairs at full resolution.\n\n'
                f'MALDI scale factor applied: {self.maldi_scale_factor:.3f}\n'
                f'H&E scale factor applied:   {self.he_scale_factor:.3f}\n\n'
                'You can now call align_image_using_landmarks().'
            )
            show_info(f"Saved {n_maldi} landmark pairs (scaled to full resolution)")

        except Exception as e:
            QMessageBox.critical(self, 'Error', f'Failed to save landmarks:\n{str(e)}')


# ----------------------------------------------------------------------
# Public launch function
# ----------------------------------------------------------------------

def launch_landmark_gui(
    maldi_sd: SpatialData,
    he_sd: SpatialData,
    maldi_image_key: str = "optical_image",
    he_image_key: str = "he_image",
    split_view: bool = False,
    maldi_scale_level: str = None,
    he_scale_level: str = "sc",
):
    """
    Launch the landmark alignment GUI.

    Parameters
    ----------
    maldi_sd : SpatialData
        MALDI spatial data object.
    he_sd : SpatialData
        H&E spatial data object.
    maldi_image_key : str
        Key for MALDI image in maldi_sd.images.
    he_image_key : str
        Key for H&E image in he_sd.images.
    split_view : bool
        If True, open each image in its own napari window.
    maldi_scale_level : str or None
        Which multiscale level to load for MALDI, e.g. 'scale0', 'scale1', 'scale2'.
        If None, the available levels are printed and you are prompted to choose.
    he_scale_level : str or None
        Which multiscale level to load for H&E.
        If None, the available levels are printed and you are prompted to choose.

    Returns
    -------
    (viewer, widget) or (main_viewer, maldi_viewer, he_viewer, widget)
    """
    # Discover available levels
    maldi_levels = _get_scale_levels(maldi_sd.images[maldi_image_key])
    he_levels    = _get_scale_levels(he_sd.images[he_image_key])

    print("Available MALDI scale levels:")
    for k, v in maldi_levels.items():
        print(f"  {k}: {v}")
    print("Available H&E scale levels:")
    for k, v in he_levels.items():
        print(f"  {k}: {v}")

    # Default to finest level if not specified
    if maldi_scale_level is None:
        maldi_scale_level = list(maldi_levels.keys())[0]
        print(f"\nNo maldi_scale_level specified — defaulting to '{maldi_scale_level}'.")
    if he_scale_level is None:
        he_scale_level = list(he_levels.keys())[0]
        print(f"No he_scale_level specified — defaulting to '{he_scale_level}'.")

    if maldi_scale_level not in maldi_levels:
        raise ValueError(f"maldi_scale_level '{maldi_scale_level}' not found. "
                         f"Available: {list(maldi_levels.keys())}")
    if he_scale_level not in he_levels:
        raise ValueError(f"he_scale_level '{he_scale_level}' not found. "
                         f"Available: {list(he_levels.keys())}")

    print(f"\nLaunching GUI with MALDI={maldi_scale_level}, H&E={he_scale_level}")

    if split_view:
        maldi_viewer = napari.Viewer(title="MALDI Image")
        he_viewer    = napari.Viewer(title="H&E Image")
        main_viewer  = napari.Viewer(title="Landmark Alignment Control")

        widget = LandmarkAlignmentWidget(
            maldi_sd, he_sd,
            maldi_viewer=maldi_viewer,
            he_viewer=he_viewer,
            maldi_image_key=maldi_image_key,
            he_image_key=he_image_key,
            split_view=True,
            maldi_scale_level=maldi_scale_level,
            he_scale_level=he_scale_level,
        )
        main_viewer.window.add_dock_widget(widget, area='right', name='Landmark Alignment')
        return main_viewer, maldi_viewer, he_viewer, widget

    else:
        viewer = napari.Viewer(title="Landmark Alignment - Grid View")
        widget = LandmarkAlignmentWidget(
            maldi_sd, he_sd,
            combined_viewer=viewer,
            maldi_image_key=maldi_image_key,
            he_image_key=he_image_key,
            split_view=False,
            maldi_scale_level=maldi_scale_level,
            he_scale_level=he_scale_level,
        )
        viewer.window.add_dock_widget(widget, area='right', name='Landmark Alignment')
        return viewer, widget