# Installation

## Requirements

- Python 3.10 or later
- A conda environment is strongly recommended due to the number of spatial biology dependencies

## Recommended: conda environment

Download the [`environment.yml`](https://github.com/agc888/goatpy/blob/main/environment.yml) file and run:

```bash
conda env create -f environment.yml
conda activate maldi
pip install goatpy
```

This installs all required and optional dependencies, including napari for visualisation.

## PyPI

For a minimal install without napari:

```bash
pip install goatpy
```

With napari support:

```bash
pip install "goatpy[napari]"
```

## From source (latest development version)

```bash
pip install git+https://github.com/agc888/goatpy.git
```

## Optional dependencies

| Extra | What it adds |
|-------|-------------|
| `napari` | Interactive visualisation via napari and napari-spatialdata |
| `docs` | Sphinx documentation build dependencies |
| `dev` | Testing, linting, and formatting tools |

## WSI support (SVS, NDPI, CZI files)

Whole-slide image formats require openslide:

```bash
conda install -c conda-forge openslide openslide-python
```
