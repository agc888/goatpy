# Changelog

## 0.1.0 (2025)

### New features
- `load_and_align`: automatic MALDI + H&E registration using cv2.warpAffine with full 360° rotation search
- `add_qupath_annotations`: add QuPath GeoJSON annotations to an existing sdata object
- `graphpca_spatialdata`: PCA with optional spatial smoothing via graph Laplacian
- `get_kmean_clusters`: k-means clustering on GraphPCA scores
- `Add_Pseudo_Image`: create spatial images from obs columns
- `launch_landmark_gui`: interactive napari landmark alignment tool
- `normalize_spatialdata`: TIC and RMS normalisation
- `he_spatialdata`: load H&E images as SpatialData objects

### Breaking changes from 0.0.1
- `load_and_align` now uses `cv2.warpAffine` for canvas construction (replacing PIL rotate), ensuring annotation and image coordinates are derived from the same matrix
- `_transform_geojson` signature changed: takes `M_stored` directly instead of individual rotation parameters
