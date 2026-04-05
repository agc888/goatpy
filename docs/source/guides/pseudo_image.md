# Pseudo-image generation

`Add_Pseudo_Image` creates a spatial image from any categorical or continuous
column in `adata.obs` and registers it in the SpatialData object.

## Categorical columns

```python
sdata = gp.Add_Pseudo_Image(sdata, "GPCA_clusters")
```

Each unique category gets a randomly assigned colour. To use a specific
matplotlib colourmap:

```python
sdata = gp.Add_Pseudo_Image(sdata, "GPCA_clusters", cmap="tab20")
```

## Continuous columns

Set `is_continuous=True` to automatically bin the values and apply a
sequential colourmap:

```python
sdata = gp.Add_Pseudo_Image(sdata, "MPI", is_continuous=True, cmap="viridis")
```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `image_ident` | required | Column name in `adata.obs` to visualise |
| `tables` | `"maldi_adata"` | Table name in `sdata.tables` |
| `library_id` | `"Spatial"` | Key in `adata.uns["spatial"]` |
| `cmap` | `None` | Matplotlib colourmap name or dict mapping categories to RGB |
| `is_continuous` | `False` | Bin continuous values before colouring |
| `img_upscaling` | `1` | Additional upscaling for the pseudo-image |

## Output

The pseudo-image is stored in `sdata.images["optical_image"]` as a multiscale
`Image2DModel`, compatible with napari and spatialdata-plot.
