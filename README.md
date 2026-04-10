<p align="center">
  <img src="https://raw.githubusercontent.com/agc888/goatpy/main/docs/source/_static/logo.png" width="200">
</p>

# goatpy — Spatial Glycomics Analysis Toolkit

[![PyPI version](https://badge.fury.io/py/goatpy.svg)](https://badge.fury.io/py/goatpy)
[![Documentation Status](https://readthedocs.org/projects/goatpy/badge/?version=latest)](https://goatpy.readthedocs.io/en/latest/?badge=latest)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

**goatpy** is a Python toolkit for spatial glycomics analysis, combining MALDI mass spectrometry imaging with H&E histology. It provides automatic image registration, pseudo-image generation, spatial PCA, and annotation tools built on top of the [SpatialData](https://spatialdata.scverse.org/) framework.

## Features

- **Automatic H&E registration** — align MALDI ion images to whole-slide H&E images using normalised cross-correlation with full 360° rotation search
- **QuPath annotation support** — transform GeoJSON annotations from QuPath into the registered coordinate system
- **Spatial GraphPCA** — dimensionality reduction with optional spatial smoothing via k-nearest-neighbour graphs
- **Pseudo-image generation** — create spatial images from categorical or continuous obs columns
- **Landmark alignment GUI** — interactive napari-based landmark alignment tool
- **SpatialData native** — all outputs are standard `SpatialData` objects, compatible with the scverse ecosystem

## Installation

### Recommended: conda environment

Download [`environment.yml`](https://github.com/agc888/goatpy/blob/main/environment.yml) and create the environment:

```bash
conda env create -f environment.yml
conda activate maldi
pip install goatpy
```

### PyPI

```bash
pip install goatpy
```

For napari visualisation support:

```bash
pip install "goatpy[napari]"
```

### From source

```bash
pip install git+https://github.com/agc888/goatpy.git
```

## Quick start

```python
import goatpy as gp

# Load and register MALDI + H&E
sdata = gp.load_and_align(
    imzml_path="my_sample.imzML",
    he_path="my_sample.svs",
    geojson_path="annotations.geojson",  # optional QuPath annotations
)

# Normalise intensities
sdata = gp.normalize_spatialdata(sdata, table_name="maldi_adata", method="TIC")

# Dimensionality reduction
sdata = gp.graphpca_spatialdata(sdata, n_components=30, alpha=0.5)

# Cluster
sdata = gp.get_kmean_clusters(sdata, n_clusters=8)

# Visualise in napari
from napari_spatialdata import Interactive
Interactive([sdata]).run()
```

## Documentation

Full documentation is available at [goatpy.readthedocs.io](https://goatpy.readthedocs.io).

## Citation

If you use goatpy in your research, please cite:

> Causer, A. (2025). goatpy: Spatial Glycomics Analysis Toolkit. https://github.com/agc888/goatpy

## License

MIT — see [LICENSE](LICENSE) for details.