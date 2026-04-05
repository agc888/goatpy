# QuPath annotations

goatpy can transform QuPath GeoJSON annotation exports into the registered
coordinate system so they overlay correctly on the H&E canvas.

## During registration

Pass `geojson_path` directly to `load_and_align`:

```python
sdata = gp.load_and_align(
    imzml_path="sample.imzML",
    he_path="sample.svs",
    geojson_path="annotations.geojson",
    geojson_shapes_key="annotations",         # key in sdata.shapes
    geojson_classification_key="classification",  # column in the GeoDataFrame
)
```

## After registration

If `sdata` was already built without annotations, add them afterwards:

```python
sdata = gp.add_qupath_annotations(
    sdata,
    geojson_path="annotations.geojson",
    shapes_key="annotations",
)
```

This reads the affine matrix stored in `sdata["maldi_adata"].uns["he_transform"]`
to reproduce the exact same transform used during registration.

## Exporting from QuPath

In QuPath, export annotations via:

**File → Export annotations → GeoJSON**

Make sure to export with **"Include default excluded objects"** unchecked, and
leave coordinates in the default native pixel space (do not apply any scaling).

## Accessing annotation labels

The classification column is stored as a categorical in the shapes GeoDataFrame:

```python
ann = sdata.shapes["annotations"]
print(ann["classification"].unique())

# Filter to a specific class
tumor = ann[ann["classification"] == "Tumor"]
```

## Visualising with spatialdata-plot

```python
sdata.pl.render_images("he_image").pl.render_shapes(
    "annotations",
    color="classification",
    fill_alpha=0.3,
    outline_alpha=0.9,
).pl.show(coordinate_systems="global")
```
