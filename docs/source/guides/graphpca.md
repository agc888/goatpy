# GraphPCA and clustering

## GraphPCA

`graphpca_spatialdata` performs PCA on the ion intensity matrix with optional
spatial smoothing via a k-nearest-neighbour graph on the pixel coordinates.

```python
sdata = gp.graphpca_spatialdata(
    sdata,
    tables="maldi_adata",
    n_components=50,
    alpha=0.0,        # 0 = no smoothing, higher = more smoothing
    n_neighbors=10,
)
```

Results are stored in `sdata["maldi_adata"].obsm["GraphPCA"]`.

### Spatial smoothing

Setting `alpha > 0` smooths the PCA scores over the tissue using a graph
Laplacian regulariser. This reduces noise while preserving spatial structure.

```python
sdata = gp.graphpca_spatialdata(sdata, n_components=30, alpha=0.5, n_neighbors=10)
```

A higher `alpha` produces smoother results but may over-smooth fine structures.
Values between `0.1` and `1.0` are a good starting range.

## K-means clustering

```python
sdata = gp.get_kmean_clusters(
    sdata,
    n_clusters=8,
    cluster_key="GPCA_clusters",
)
```

Cluster labels are stored in `sdata["maldi_adata"].obs["GPCA_clusters"]` as
a categorical column.

## Visualising clusters as a pseudo-image

```python
sdata = gp.Add_Pseudo_Image(sdata, "GPCA_clusters")
```

This creates a colour image in `sdata.images["optical_image"]` where each
pixel is coloured by its cluster label.
