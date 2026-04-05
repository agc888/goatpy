# Loading and aligning MALDI + H&E

## Overview

`load_and_align` is the core function of goatpy. It:

1. Reads a MALDI imzML dataset and extracts ion images for a list of glycan peaks
2. Loads an H&E whole-slide image at the MALDI pixel resolution
3. Finds the best rotation and translation to register the two images using normalised
   cross-correlation
4. Builds a `SpatialData` object with the H&E canvas, MALDI pixels, and optional annotations
   all in the same coordinate system

## Basic usage

```python
import goatpy as gp

sdata = gp.load_and_align(
    imzml_path="sample.imzML",
    he_path="sample.svs",
)
```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `imzml_path` | required | Path to `.imzML` file |
| `he_path` | required | Path to H&E image. SVS/NDPI require openslide |
| `peaks_path` | `None` | Path to peaks CSV. Uses bundled PEAKS.csv when `None` |
| `geojson_path` | `None` | Path to QuPath GeoJSON annotation export |
| `geojson_shapes_key` | `"annotations"` | Key for annotations in `sdata.shapes` |
| `maldi_pixel_um` | `None` | MALDI pixel size in µm. Auto-read from imzML |
| `he_pixel_um` | `None` | H&E native pixel size in µm. Auto-read from metadata |
| `coarse_rotation_step` | `15` | Degrees between coarse rotation candidates (0–360°) |
| `fine_rotation_range` | `5.0` | ±degrees for fine search around best coarse angle |
| `fine_rotation_step` | `1.0` | Degree increment for fine search |
| `buffer_px` | `150` | Canvas padding in pixels at registration resolution |
| `img_upscaling` | `10` | Upscaling factor for the output H&E canvas |
| `spectra_chunk_size` | `10` | Ion images loaded in parallel at once |

## Output SpatialData structure

```
sdata
├── images
│   └── he_image          # Full registered H&E canvas (upscaled)
├── shapes
│   ├── pixels            # One square polygon per MALDI pixel
│   └── annotations       # QuPath annotations (if geojson_path provided)
├── points
│   └── centroids         # Centroid of each MALDI pixel
└── tables
    └── maldi_adata       # AnnData: rows=pixels, cols=m/z peaks
        ├── X             # Ion intensities (n_pixels × n_peaks)
        ├── obs           # x, y, he_x, he_y, MPI columns
        ├── var           # m/z peak labels
        └── uns
            ├── spatial   # Scanpy-compatible spatial slot
            └── he_transform  # Registration parameters and affine matrix
```

## Custom peaks

By default, goatpy uses a bundled list of 121 glycan peaks. To use your own:

```python
sdata = gp.load_and_align(
    imzml_path="sample.imzML",
    he_path="sample.svs",
    peaks_path="my_peaks.csv",
)
```

The peaks CSV should have one m/z value per line (with a header row), in the same
format as the bundled `PEAKS.csv`.

## Registration quality

The registration score is logged during the run. A score above `0.5` generally indicates
a good registration. If the score is low, try:

- Increasing `coarse_rotation_step` resolution (e.g. `coarse_rotation_step=5`)
- Adjusting `buffer_px` if the MALDI tissue is near the image edge
- Explicitly passing `maldi_pixel_um` if the imzML metadata is incorrect
