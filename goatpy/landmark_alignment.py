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
    QLabel, QListWidget, QMessageBox
)
from qtpy.QtCore import Qt


from spatialdata.transformations import (
    align_elements_using_landmarks,
    get_transformation_between_landmarks,
)

from spatialdata.transformations import (
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



def align_image_using_landmarks(
    maldi_sd: SpatialData,
    he_sd: SpatialData,
    maldi_landmark_key: str = "maldi_landmarks",
    he_landmark_key: str = "he_landmarks",
    maldi_image_key: str = "optical_image",
    he_image_key: str = "he_image",
) -> None:
    """
    Align two SpatialData objects using landmarks.

    Parameters
    ----------
    maldi_sd
        The reference SpatialData object (maldi_data).
    he_sd
        The moving SpatialData object to be aligned to the reference (H&E).
    """

    affine = get_transformation_between_landmarks(
        references_coords=maldi_sd[maldi_landmark_key], moving_coords= he_sd[he_landmark_key]
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

    he_image = he_sd.images[he_image_key]
    maldi_sd.images[he_image_key] = he_image

    return maldi_sd



class LandmarkAlignmentWidget(QWidget):
    """Widget for selecting corresponding landmarks in two images."""
    
    def __init__(self, maldi_sd: SpatialData, he_sd: SpatialData,
                 maldi_viewer: napari.Viewer = None, 
                 he_viewer: napari.Viewer = None,
                 combined_viewer: napari.Viewer = None,
                 maldi_image_key: str = "optical_image", 
                 he_image_key: str = "he_image",
                 split_view: bool = False):
        super().__init__()
        self.maldi_sd = maldi_sd
        self.he_sd = he_sd
        self.maldi_image_key = maldi_image_key
        self.he_image_key = he_image_key
        self.split_view = split_view
        
        # Store viewers based on mode
        if split_view:
            self.maldi_viewer = maldi_viewer
            self.he_viewer = he_viewer
            self.viewer = None
        else:
            self.viewer = combined_viewer
            self.maldi_viewer = None
            self.he_viewer = None
        
        # Store landmarks
        self.maldi_landmarks = []
        self.he_landmarks = []
        self.active_dataset = "maldi"
        
        self.init_ui()
        self.setup_layers()
    
    def _extract_image_data(self, image):
        """Extract numpy array from various image formats."""
        try:
            # Handle DataTree (xarray) - spatialdata multiscale format
            if hasattr(image, 'ds') and hasattr(image, 'scale0'):
                ds = image['scale0'].ds
                data_var = list(ds.data_vars)[0]
                data = ds[data_var].values
            elif hasattr(image, 'ds'):
                ds = image.ds
                if len(ds.data_vars) > 0:
                    data_var = list(ds.data_vars)[0]
                    data = ds[data_var].values
                else:
                    raise ValueError("DataTree has no data variables")
            elif hasattr(image, 'to_dataset'):
                ds = image.to_dataset()
                data_var = list(ds.data_vars)[0]
                data = ds[data_var].values
            elif hasattr(image, 'values'):
                data = image.values
            elif hasattr(image, 'data'):
                if hasattr(image.data, 'compute'):
                    data = image.data.compute()
                else:
                    data = image.data
            else:
                data = np.array(image)
            
            return data
        except Exception as e:
            print(f"Image type: {type(image)}")
            print(f"Image attributes: {dir(image)}")
            raise ValueError(f"Could not extract image data: {e}")
        
    def init_ui(self):
        """Initialize the user interface."""
        layout = QVBoxLayout()
        
        # Title
        title = QLabel("<h2>Landmark Alignment Tool</h2>")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        
        # Instructions (adapt based on mode)
        if self.split_view:
            instructions_text = (
                "1. Select which image to add landmarks to\n"
                "2. Click on the ACTIVE image window to add landmarks\n"
                "3. Add corresponding points in both images\n"
                "4. Minimum 3 pairs recommended, 5+ for better accuracy\n"
                "5. Click 'Save Landmarks' when done\n\n"
                "Note: Each image is in its own window for independent zoom/pan!"
            )
        else:
            instructions_text = (
                "1. Select which image to add landmarks to\n"
                "2. Click on the image to add landmarks\n"
                "3. Add corresponding points in both images\n"
                "4. Minimum 3 pairs recommended, 5+ for better accuracy\n"
                "5. Click 'Save Landmarks' when done\n\n"
                "Note: Images are side-by-side with independent zoom/pan!"
            )
        
        instructions = QLabel(instructions_text)
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
        
        # MALDI landmarks
        maldi_list_layout = QVBoxLayout()
        maldi_label = QLabel("<b>MALDI Landmarks</b>")
        maldi_label.setAlignment(Qt.AlignCenter)
        self.maldi_list = QListWidget()
        maldi_list_layout.addWidget(maldi_label)
        maldi_list_layout.addWidget(self.maldi_list)
        
        # H&E landmarks
        he_list_layout = QVBoxLayout()
        he_label = QLabel("<b>H&E Landmarks</b>")
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
        
        # Status
        status_text = "Status: Ready. Click to add landmarks." if self.split_view else "Status: Ready. Select MALDI and click to add landmarks."
        self.status_label = QLabel(status_text)
        self.status_label.setStyleSheet("padding: 5px; background-color: #e0e0e0;")
        layout.addWidget(self.status_label)
        
        self.setLayout(layout)
        
    def setup_layers(self):
        """Add images to viewer(s) and setup point layers."""
        if self.split_view:
            self._setup_split_view()
        else:
            self._setup_combined_view()
    
    def _setup_combined_view(self):
        """Setup combined grid view in single viewer."""
        # Add MALDI image
        maldi_image = self.maldi_sd.images[self.maldi_image_key]
        maldi_data = self._extract_image_data(maldi_image)
        self.viewer.add_image(
            maldi_data,
            name="MALDI",
            colormap="gray"
        )
        
        # Add H&E image
        he_image = self.he_sd.images[self.he_image_key]
        he_data = self._extract_image_data(he_image)
        if he_data.ndim == 3 and he_data.shape[0] in [3, 4]:
            he_data = np.transpose(he_data, (1, 2, 0))
        self.viewer.add_image(
            he_data,
            name="H&E",
            rgb=True if he_data.ndim == 3 else False
        )
        
        # Add point layers
        self.maldi_points_layer = self.viewer.add_points(
            np.empty((0, 2)),
            name="MALDI Landmarks",
            face_color="green",
            size=20,
            border_color="white",
            border_width=2
        )
        
        self.he_points_layer = self.viewer.add_points(
            np.empty((0, 2)),
            name="H&E Landmarks",
            face_color="blue",
            size=20,
            border_color="white",
            border_width=2
        )
        
        # Connect callbacks
        self.maldi_points_layer.mouse_drag_callbacks.append(self.on_click_combined)
        self.he_points_layer.mouse_drag_callbacks.append(self.on_click_combined)
        
        # Set grid view with independent zooming/panning
        self.viewer.grid.enabled = True
        self.viewer.grid.shape = (1, 2)
        self.viewer.grid.stride = 1
    
    def _setup_split_view(self):
        """Setup separate viewers for MALDI and H&E."""
        # Add MALDI image and points
        maldi_image = self.maldi_sd.images[self.maldi_image_key]
        maldi_data = self._extract_image_data(maldi_image)
        self.maldi_viewer.add_image(
            maldi_data,
            name="MALDI",
            colormap="gray"
        )
        
        self.maldi_points_layer = self.maldi_viewer.add_points(
            np.empty((0, 2)),
            name="MALDI Landmarks",
            face_color="green",
            size=20,
            border_color="white",
            border_width=2
        )
        
        # Add H&E image and points
        he_image = self.he_sd.images[self.he_image_key]
        he_data = self._extract_image_data(he_image)
        if he_data.ndim == 3 and he_data.shape[0] in [3, 4]:
            he_data = np.transpose(he_data, (1, 2, 0))
        self.he_viewer.add_image(
            he_data,
            name="H&E",
            rgb=True if he_data.ndim == 3 else False
        )
        
        self.he_points_layer = self.he_viewer.add_points(
            np.empty((0, 2)),
            name="H&E Landmarks",
            face_color="blue",
            size=20,
            border_color="white",
            border_width=2
        )
        
        # Connect callbacks
        self.maldi_points_layer.mouse_drag_callbacks.append(
            lambda l, e: self.on_click_split(self.maldi_viewer, l, e)
        )
        self.he_points_layer.mouse_drag_callbacks.append(
            lambda l, e: self.on_click_split(self.he_viewer, l, e)
        )
        
    def set_active_dataset(self, dataset: str):
        """Set which dataset is active for adding landmarks."""
        self.active_dataset = dataset
        if dataset == "maldi":
            self.maldi_btn.setChecked(True)
            self.he_btn.setChecked(False)
            if self.split_view:
                self.status_label.setText("Status: Click on MALDI window to add landmarks")
            else:
                self.status_label.setText("Status: Click on MALDI image to add landmarks")
        else:
            self.maldi_btn.setChecked(False)
            self.he_btn.setChecked(True)
            if self.split_view:
                self.status_label.setText("Status: Click on H&E window to add landmarks")
            else:
                self.status_label.setText("Status: Click on H&E image to add landmarks")
    
    def on_click_combined(self, layer, event):
        """Handle mouse click in combined view."""
        if event.type == 'mouse_press' and event.button == 1:
            coords = layer.world_to_data(event.position)
            
            if self.active_dataset == "maldi":
                self.maldi_landmarks.append([coords[0], coords[1]])
                self.maldi_list.addItem(f"Point {len(self.maldi_landmarks)}: ({coords[0]:.1f}, {coords[1]:.1f})")
                self.maldi_points_layer.data = np.array(self.maldi_landmarks)
                show_info(f"Added MALDI landmark {len(self.maldi_landmarks)}")
            else:
                self.he_landmarks.append([coords[0], coords[1]])
                self.he_list.addItem(f"Point {len(self.he_landmarks)}: ({coords[0]:.1f}, {coords[1]:.1f})")
                self.he_points_layer.data = np.array(self.he_landmarks)
                show_info(f"Added H&E landmark {len(self.he_landmarks)}")
            
            self.update_status()
    
    def on_click_split(self, viewer, layer, event):
        """Handle mouse click in split view."""
        if event.type == 'mouse_press' and event.button == 1:
            coords = layer.world_to_data(event.position)
            
            if self.active_dataset == "maldi":
                self.maldi_landmarks.append([coords[0], coords[1]])
                self.maldi_list.addItem(f"Point {len(self.maldi_landmarks)}: ({coords[0]:.1f}, {coords[1]:.1f})")
                self.maldi_points_layer.data = np.array(self.maldi_landmarks)
                show_info(f"Added MALDI landmark {len(self.maldi_landmarks)}")
            else:
                self.he_landmarks.append([coords[0], coords[1]])
                self.he_list.addItem(f"Point {len(self.he_landmarks)}: ({coords[0]:.1f}, {coords[1]:.1f})")
                self.he_points_layer.data = np.array(self.he_landmarks)
                show_info(f"Added H&E landmark {len(self.he_landmarks)}")
            
            self.update_status()
    
    def update_status(self):
        """Update status message."""
        n_maldi = len(self.maldi_landmarks)
        n_he = len(self.he_landmarks)
        
        if n_maldi == n_he and n_maldi >= 3:
            self.status_label.setText(
                f"Status: {n_maldi} landmark pairs added. Ready to save!"
            )
            self.status_label.setStyleSheet("padding: 5px; background-color: #C8E6C9;")
        elif n_maldi == n_he:
            self.status_label.setText(
                f"Status: {n_maldi} landmark pairs added. Add at least 3 pairs."
            )
            self.status_label.setStyleSheet("padding: 5px; background-color: #FFF9C4;")
        else:
            self.status_label.setText(
                f"Status: Unequal landmarks (MALDI: {n_maldi}, H&E: {n_he}). Add corresponding points."
            )
            self.status_label.setStyleSheet("padding: 5px; background-color: #FFCDD2;")
    
    def undo_last(self):
        """Remove the last added landmark."""
        if self.active_dataset == "maldi" and self.maldi_landmarks:
            self.maldi_landmarks.pop()
            self.maldi_list.takeItem(self.maldi_list.count() - 1)
            self.maldi_points_layer.data = np.array(self.maldi_landmarks) if self.maldi_landmarks else np.empty((0, 2))
            show_info("Removed last MALDI landmark")
        elif self.active_dataset == "he" and self.he_landmarks:
            self.he_landmarks.pop()
            self.he_list.takeItem(self.he_list.count() - 1)
            self.he_points_layer.data = np.array(self.he_landmarks) if self.he_landmarks else np.empty((0, 2))
            show_info("Removed last H&E landmark")
        
        self.update_status()
    
    def clear_all(self):
        """Clear all landmarks."""
        reply = QMessageBox.question(
            self, 'Clear All',
            'Are you sure you want to clear all landmarks?',
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
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
    
    def save_landmarks(self):
        """Save landmarks to SpatialData objects."""
        n_maldi = len(self.maldi_landmarks)
        n_he = len(self.he_landmarks)
        
        if n_maldi != n_he:
            QMessageBox.warning(
                self, 'Unequal Landmarks',
                f'Number of landmarks must be equal.\nMALDI: {n_maldi}, H&E: {n_he}'
            )
            return
        
        if n_maldi < 3:
            QMessageBox.warning(
                self, 'Too Few Landmarks',
                f'At least 3 landmark pairs are required.\nCurrent: {n_maldi}'
            )
            return
        
        try:
            # Create landmarks as Points for MALDI
            maldi_df = pd.DataFrame({
                'x': [p[1] for p in self.maldi_landmarks],
                'y': [p[0] for p in self.maldi_landmarks],
            })
            maldi_points = PointsModel.parse(maldi_df)
            self.maldi_sd.points['maldi_landmarks'] = maldi_points
            
            # Create landmarks as Points for H&E
            he_df = pd.DataFrame({
                'x': [p[1] for p in self.he_landmarks],
                'y': [p[0] for p in self.he_landmarks],
            })
            he_points = PointsModel.parse(he_df)
            self.he_sd.points['he_landmarks'] = he_points
            
            QMessageBox.information(
                self, 'Success',
                f'Successfully saved {n_maldi} landmark pairs!\n\n'
                'Landmarks saved as:\n'
                '- maldi_sd.points["maldi_landmarks"]\n'
                '- he_sd.points["he_landmarks"]\n\n'
                'You can now use align_image_using_landmarks().'
            )
            
            show_info(f"Saved {n_maldi} landmark pairs to SpatialData objects")
            
        except Exception as e:
            QMessageBox.critical(
                self, 'Error',
                f'Failed to save landmarks:\n{str(e)}'
            )


def launch_landmark_gui(maldi_sd: SpatialData, he_sd: SpatialData,
                        maldi_image_key: str = "optical_image",
                        he_image_key: str = "he_image",
                        split_view: bool = False):
    """
    Launch the landmark alignment GUI.
    
    Parameters
    ----------
    maldi_sd : SpatialData
        MALDI spatial data object with image
    he_sd : SpatialData
        H&E spatial data object with image
    maldi_image_key : str
        Key for MALDI image in maldi_sd.images
    he_image_key : str
        Key for H&E image in he_sd.images
    split_view : bool
        If True, use separate windows for each image (fully independent zoom/pan).
        If False, use grid view in single window (independent zoom/pan per grid cell).
    
    Returns
    -------
    If split_view=False:
        viewer : napari.Viewer
            The napari viewer instance
        widget : LandmarkAlignmentWidget
            The landmark widget instance
    
    If split_view=True:
        main_viewer : napari.Viewer
            Control panel viewer
        maldi_viewer : napari.Viewer
            MALDI image viewer
        he_viewer : napari.Viewer
            H&E image viewer
        widget : LandmarkAlignmentWidget
            The landmark widget instance
    """
    if split_view:
        # Create separate viewers for each image
        maldi_viewer = napari.Viewer(title="MALDI Image")
        he_viewer = napari.Viewer(title="H&E Image")
        main_viewer = napari.Viewer(title="Landmark Alignment Control")
        
        widget = LandmarkAlignmentWidget(
            maldi_sd, he_sd,
            maldi_viewer=maldi_viewer,
            he_viewer=he_viewer,
            maldi_image_key=maldi_image_key,
            he_image_key=he_image_key,
            split_view=True
        )
        
        main_viewer.window.add_dock_widget(widget, area='right', name='Landmark Alignment')
        
        return main_viewer, maldi_viewer, he_viewer, widget
    
    else:
        # Combined grid view
        viewer = napari.Viewer(title="Landmark Alignment - Grid View")
        
        widget = LandmarkAlignmentWidget(
            maldi_sd, he_sd,
            combined_viewer=viewer,
            maldi_image_key=maldi_image_key,
            he_image_key=he_image_key,
            split_view=False
        )
        
        viewer.window.add_dock_widget(widget, area='right', name='Landmark Alignment')
        
        return viewer, widget


