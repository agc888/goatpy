# Quickstart

## Load and register MALDI + H&E

The main entry point is `load_and_align`, which reads an imzML dataset,
loads an H&E image, registers them automatically, and returns a
`SpatialData` object.

```python
import goatpy as gp

sdata = gp.load_and_align(
    imzml_path="my_sample.imzML",
    he_path="my_sample.svs",
)
```

With QuPath annotations:

```python
sdata = gp.load_and_align(
    imzml_path="my_sample.imzML",
    he_path="my_sample.svs",
    geojson_path="annotations.geojson",
)
```

## Normalise

```python
sdata = gp.normalize_spatialdata(sdata, table_name="maldi_adata", method="TIC")
```

## Dimensionality reduction

```python
sdata = gp.graphpca_spatialdata(sdata, n_components=30)
```

With spatial smoothing:

```python
sdata = gp.graphpca_spatialdata(sdata, n_components=30, alpha=0.5, n_neighbors=10)
```

## Clustering

```python
sdata = gp.get_kmean_clusters(sdata, n_clusters=8)
```

## Visualise

```python
# Pseudo-image from a cluster column
sdata = gp.Add_Pseudo_Image(sdata, "GPCA_clusters")

# Interactive napari viewer
from napari_spatialdata import Interactive
Interactive([sdata]).run()
```

## Add annotations to an existing sdata object

If you have already run `load_and_align` and want to add annotations later:

```python
sdata = gp.add_qupath_annotations(sdata, geojson_path="annotations.geojson")
```
