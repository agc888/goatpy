# Changelog

## 1.1.0 (2026)

### New features

- `deconvolve_maldi_to_cells` — analyse mixed signals in spatial glycomics data and deconvolve pixels into single-cell intensities.
- `extract_channel_intensity` - extract corresponding IHC channel intensity values per pixel or cell
- `pixels_to_cells` - transfers pixel level intensities to single-cells
- `spatial_autocorrelation` - identifies spatially correlated features using MoransI and Pearson


## 1.0.0 (2025)

### New features

- `load_and_align`: automatic MALDI + H&E registration using cv2.warpAffine with full 360° rotation search
- `add_qupath_annotations`: add QuPath GeoJSON annotations to an existing sdata object
- `graphpca_spatialdata`: Run spatially informed GraphPCA on glycomics data
- `launch_landmark_gui`: interactive napari landmark alignment tool
- `normalize_spatialdata`: TIC and RMS normalisation
- `he_spatialdata`: load H&E images as SpatialData objects