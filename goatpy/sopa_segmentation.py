import sopa
import numpy as np, cv2
import pandas as pd
import geopandas as gpd
import spatialdata as sd
import matplotlib.pyplot as plt

from shapely.affinity import affine_transform
from shapely.geometry import Polygon
from spatialdata.transformations import Affine, set_transformation, get_transformation
from spatialdata.models import Image2DModel, ShapesModel, TableModel
from scipy.ndimage import binary_fill_holes
from skimage.color import rgb2hsv
from skimage.filters import threshold_otsu, gaussian
from skimage.morphology import remove_small_objects


def make_gp_image_patches(sdata, image_key="he_image", coordinate_system="aligned", roi_key="pixels",patch_width=2000, patch_overlap=50, key_added=None):
    """
    Generates image patches for the given image based on the specified coordinate system (i.e. aligned). 
    Based on sopa.make_image_patches, but uses a goatpy generated coordinate system to map the pixels to image patched
    """
    
    # the affine that maps he_image's raw pixel space -> 'aligned'
    aligned_transform = get_transformation(sdata.images[image_key], to_coordinate_system=coordinate_system)
    matrix = aligned_transform.to_affine_matrix(input_axes=("x", "y"), output_axes=("x", "y"))

    # invert it: 'aligned' -> he_image's raw pixel space
    inv = np.linalg.inv(matrix)
    a, b, tx = inv[0]
    c, d, ty = inv[1]

    pixels_gdf = sdata.shapes[roi_key]
    pixels_raw_geom = pixels_gdf.geometry.apply(
        lambda geom: affine_transform(geom, [a, b, c, d, tx, ty])
    )
    pixels_raw = gpd.GeoDataFrame(geometry=pixels_raw_geom)

    # gets an Identity transform by default -> matches he_image's own raw ('global') space
    sdata.shapes["in_maldi_pixels"] = ShapesModel.parse(pixels_raw)

    sopa.make_image_patches(sdata,patch_width=patch_width,patch_overlap=patch_overlap,roi_key="in_maldi_pixels",image_key=image_key,key_added=key_added)

    del sdata.shapes["in_maldi_pixels"]    
    
    return(sdata)



def add_n_cells_per_pixel(
    sdata,
    table_key: str = "maldi_adata",
    cells_key: str = "cell_segmentation",
    pixels_key: str = "pixels",
    coordinate_system: str = "aligned",
):
    # bring both shapes into the same coordinate system
    cells = sd.transform(sdata.shapes[cells_key], to_coordinate_system=coordinate_system)
    pixels = sd.transform(sdata.shapes[pixels_key], to_coordinate_system=coordinate_system)

    # use cell centroids so each cell is assigned to exactly one pixel
    cell_pts = gpd.GeoDataFrame(geometry=cells.geometry.centroid, index=cells.index)
    joined = gpd.sjoin(cell_pts, pixels[["geometry"]], how="left", predicate="within")
    counts = joined.groupby("index_right").size()

    n_cells = pd.Series(0, index=pixels.index, dtype=int)
    n_cells.loc[counts.index] = counts.values

    adata = sdata.tables[table_key]
    instance_key = adata.uns[TableModel.ATTRS_KEY]["instance_key"]  # usually "instance_id"

    adata.obs["n_cells"] = n_cells.reindex(adata.obs[instance_key]).to_numpy()
    return sdata



def add_pyramid(sdata, image_key="he_image", key_added="he_image_pyr",
                scale_factors=(4, 4, 4), chunks=(3, 2048, 2048)):
    """Build a multiscale copy of a single-scale image (skipped if it already exists)."""
    if key_added in sdata.images:
        return key_added
    full = sdata[image_key]["scale0"]["image"]  # DataArray with transforms attached
    sdata[key_added] = Image2DModel.parse(
        full, scale_factors=list(scale_factors), chunks=chunks
    )
    return key_added


def tissue_mask(sdata, image_key="he_image_pyr", roi_key="region_of_interest",
               source_key="he_image", level=2, thr_mult=0.4, close_size=41,
               min_obj=500, sigma=1, plot=True):
    """Saturation-based tissue mask -> polygons stored in sdata.shapes[roi_key].

    level      : pyramid level used for the mask (2 = 1/16 scale)
    thr_mult   : multiplier on the Otsu threshold; lower = more inclusive
    close_size : closing kernel (px at `level`); bigger bridges gaps in adipose
    min_obj    : minimum object size (px at `level`) to keep
    """
    pyr = sdata[image_key]
    scales = list(pyr.keys())
    arr = pyr[scales[level]]["image"].values              # (c, y, x)
    full_shape = pyr[scales[0]]["image"].shape
    sy = full_shape[1] / arr.shape[1]
    sx = full_shape[2] / arr.shape[2]

    rgb = np.moveaxis(arr[:3], 0, -1)
    sat = gaussian(rgb2hsv(rgb)[..., 1], sigma=sigma)

    mask = sat > threshold_otsu(sat) * thr_mult
    mask = remove_small_objects(mask, min_size=min_obj)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_size, close_size))
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, k)
    mask = binary_fill_holes(mask).astype(np.uint8)

    if plot:
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        ax[0].imshow(rgb)
        ax[1].imshow(rgb)
        ax[1].contour(mask, levels=[0.5], colors="k", linewidths=1)
        plt.show()

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polys = [Polygon(c[:, 0, :] * [sx, sy]) for c in contours if len(c) >= 3]
    polys = [p.buffer(0) for p in polys if p.area > 0]

    if roi_key in sdata.shapes:
        del sdata[roi_key]
    sdata[roi_key] = ShapesModel.parse(
        gpd.GeoDataFrame(geometry=polys),
        transformations=get_transformation(sdata[source_key], get_all=True),
    )
    return mask



def fullres_to_sdata_affine(sdata, full_w, full_h):
    """3x3 affine: full-res image coords -> sdata 'global' coords."""
    t = sdata["maldi_adata"].uns["he_transform"]
    M_stored = np.array(t["affine_matrix"], dtype=float)  # reg px -> cropped canvas px
    us = float(t["img_upscaling"])
    ref_h, ref_w = t["he_reg_size"]                       # registered reference size (after ref_scale)
    sx, sy = ref_w / full_w, ref_h / full_h               # full-res px -> reg px

    # full-res coords (pixel edge = 0) -> reg-res pixel index
    to_reg = np.array([[sx, 0, -0.5], [0, sy, -0.5], [0, 0, 1]])
    # canvas pixel index -> sdata coords (matches the nearest-neighbour upsample)
    up = np.array([[us, 0, 0.5 * us], [0, us, 0.5 * us], [0, 0, 1]])
    return up @ M_stored @ to_reg