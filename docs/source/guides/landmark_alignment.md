# Landmark alignment GUI

For cases where automatic registration is insufficient, goatpy provides an
interactive napari GUI for manual landmark-based alignment.

## Launch the GUI

```python
import goatpy as gp

maldi_sdata = gp.load_and_align("sample.imzML", "sample.svs")
he_sdata    = gp.he_spatialdata("sample.svs")

viewer, widget = gp.launch_landmark_gui(
    maldi_sd=maldi_sdata,
    he_sd=he_sdata,
    maldi_image_key="optical_image",
    he_image_key="he_image",
)
```

## Workflow

1. The GUI opens a napari window showing both images side by side
2. Select **"Add MALDI Landmarks"** and click corresponding points on the MALDI image
3. Select **"Add H&E Landmarks"** and click the same anatomical points on the H&E image
4. Add at least 3 pairs (5 or more recommended for best accuracy)
5. Click **"Save Landmarks"** — coordinates are automatically scaled to full resolution
6. Run `align_image_using_landmarks()` to compute and apply the transform

## Apply the alignment

```python
maldi_sdata = gp.align_image_using_landmarks(
    maldi_sd=maldi_sdata,
    he_sd=he_sdata,
    maldi_landmark_key="maldi_landmarks",
    he_landmark_key="he_landmarks",
    maldi_image_key="optical_image",
    he_image_key="he_image",
)
```

## Multiscale images

For large WSI files, specify a coarser scale level for display to avoid memory issues:

```python
viewer, widget = gp.launch_landmark_gui(
    maldi_sd=maldi_sdata,
    he_sd=he_sdata,
    maldi_scale_level="scale0",
    he_scale_level="scale2",   # lower resolution for display
)
```

Landmark coordinates are automatically scaled back to full resolution before saving.
