import pandas as pd
import numpy as np
import scanpy as sc
from anndata import AnnData
from spatialdata import SpatialData
from spatialdata.models import PointsModel, Image2DModel, TableModel
from spatialdata.transformations import Identity

from PIL import Image
from PIL.ExifTags import TAGS

Image.MAX_IMAGE_PIXELS = None 


from metaspace_converter.constants import (
    COL,
    COORD_SYS_GLOBAL,
    INSTANCE_KEY,
    MICROMETER,
    OPTICAL_IMAGE_KEY,
    POINTS_KEY,
    REGION_KEY,
    XY,
    YX,
    YXC,
    X,
    Y,
)
from spatialdata.transformations import Affine, Scale, Sequence as SequenceTransform, Translation

import matplotlib.pyplot as plt


def Add_Pseudo_Image(sdata, image_ident, tables = "maldi_adata",library_id = "Spatial", convert_to_int = True, cmap = None, is_continous = False):
    
    adata = sdata.tables[tables]

    if 'x' in adata.obs.columns and 'y' in adata.obs.columns:
        adata.obs["x_coord"] = adata.obs["x"] - adata.obs["x"].min() 
        adata.obs["y_coord"] = adata.obs["y"] - adata.obs["y"].min()
        if is_continous:
            binned_values = generate_continuous_bins(adata.obs[image_ident].to_numpy())
            adata.obs['image_bin'] = pd.Categorical(binned_values)
            image_ident = 'image_bin' 
            if cmap is None:
                cmap = 'viridis'  
        else:
            adata.obs[image_ident] = adata.obs[image_ident].astype("category")
        adata.obsm["spatial"] = adata.obs[["x_coord","y_coord"]].to_numpy()
        adata.obsm["spatial"] = adata.obsm["spatial"].astype(float)
        adata = add_uns(adata,image_ident, library_id, cmap = cmap)
    else:
        raise KeyError("The required columns 'x' and 'y' are missing from the adata.obs slot!")
        

    ig = adata.uns["spatial"][library_id]["images"]["hires"]

    if convert_to_int:
        ig = (ig * 255).astype(np.uint8)
        
    # Create an identity affine transformation matrix
    identity_matrix = np.array([[1, 0, 0],
                                 [0, 1, 0],
                                 [0, 0, 1]])
    
    # Create an affine transformation object
    optical_to_ion_image = Affine(matrix=identity_matrix, input_axes=XY, output_axes=XY)
    
    # Use the transformation in your Image2DModel.parse call
    optical_image = Image2DModel.parse(
        ig,
        transformations={COORD_SYS_GLOBAL: optical_to_ion_image},
        dims=YXC,
        scale_factors=(2, 2, 2),
        axis_units={Y: MICROMETER, X: MICROMETER},
        rgb = None,
    )

    
    sdata.images[OPTICAL_IMAGE_KEY] = optical_image

    sdata.tables[tables] = adata

    return(sdata)

def add_uns(adata,ident, library_id, cmap = None, data_type = "categorical"):
    
    unique_categories = np.unique(adata.obs[ident])
    if cmap is None:
        color_map = generate_random_colors(unique_categories)
    elif isinstance(cmap, str):
        colours = plt.get_cmap(str(cmap), len(unique_categories))
        color_map = {cat: np.array(colours(i)[:3])*255 for i, cat in enumerate(unique_categories)}
        color_map = {k: v.astype(np.uint8) for k,v in color_map.items()}
    else:
        color_map = cmap
        
    image = create_image_from_data(adata.obsm["spatial"], adata.obs[ident], color_map)
    imgarr = np.array(image)

    
    # Create spatial dictionary
    adata.uns["spatial"] = {}
    adata.uns["spatial"][library_id] = {}
    adata.uns["spatial"][library_id]["images"] = {}
    adata.uns["spatial"][library_id]["images"]["hires"] = imgarr
    adata.uns["spatial"][library_id]["use_quality"] = "hires"
    adata.uns["spatial"][library_id]["scalefactors"] = {}
    adata.uns["spatial"][library_id]["scalefactors"][
        "tissue_" + "hires" + "_scalef"
    ] = 1.0
    adata.uns["spatial"][library_id]["scalefactors"][
        "spot_diameter_fullres"
    ] = 15
    return adata

def generate_random_colors(categories):
    """
    Generate a random color for each category.

    Parameters:
    - categories: List or array of unique category values.

    Returns:
    - A dictionary mapping categories to colors.
    """
    num_categories = len(categories)
    colors = np.random.randint(0, 256, size=(num_categories, 3), dtype=np.uint8)  # RGB colors
    color_map = dict(zip(categories, colors))
    return color_map

def map_categories_to_colors(categories, color_map):
    """
    Map category values to colors using the color map.

    Parameters:
    - categories: Array of category values.
    - color_map: Dictionary mapping continents to colors.

    Returns:
    - An array of RGB colors.
    """
    colors = np.array([color_map[cat] for cat in categories])
    return colors

def generate_continuous_bins(values):
    """
    Generate bins for continuous values.

    Parameters:
    - values: Array of continuous values.

    Returns:
    - Binned values as integers.
    """
    iqr = np.subtract(*np.percentile(values, [75, 25]))
    bin_width = 2 * iqr / (len(values) ** (1/3))
    n_bins = max(int((values.max() - values.min()) / bin_width), 1)

    bins = pd.qcut(values, q=n_bins, labels=False, duplicates="drop")
    return bins


def create_image_from_data(coords_array, category_values, color_map):
    """
    Create an image from x, y coordinates and associated categorical values.

    Parameters:
    - coords_array: 2D NumPy array with shape (n, 2) where each row is [x, y].
    - category_values: Array of categorical values corresponding to each coordinate.
    - color_map: Dictionary mapping categories to RGB colors.

    Returns:
    - PIL.Image object.
    """
    # Extract x and y coordinates
    x = coords_array[:, 0]
    y = coords_array[:, 1]

    # Determine image dimensions
    min_x, max_x = np.min(x), np.max(x)
    min_y, max_y = np.min(y), np.max(y)

    # Calculate the image size
    width = int(np.ceil(max_x - min_x + 1))
    height = int(np.ceil(max_y - min_y + 1))

    # Initialize a blank image with zeros
    image_array = np.zeros((height, width, 3), dtype=np.uint8)  # RGB image

    # Map categories to colors
    color_values = map_categories_to_colors(category_values, color_map)

    # Map coordinates to image array
    for xi, yi, color in zip(x, y, color_values):
        # Translate coordinates to fit in the image
        img_x = int(xi - min_x)
        img_y = int(yi - min_y)

        if 0 <= img_x < width and 0 <= img_y < height:
            image_array[img_y, img_x] = color

    # Create an image from the array
    image = Image.fromarray(image_array, 'RGB')

    # Get current dimensions
    width, height = image.size
    
    # Check if dimensions are divisible by 2
    def is_divisible_by_2(x):
        return x % 2 == 0
    
    # Adjust dimensions if not divisible by 2
    if not is_divisible_by_2(width):
        width -= 1
    if not is_divisible_by_2(height):
        height -= 1
    
    # Resize the image to new dimensions if needed
    if width != image.size[0] or height != image.size[1]:
        image = image.resize((width, height), Image.BILINEAR)
    

    return image


def he_spatialdata(he_image_path):


    he_img = Image.open(he_image_path)

    img = np.array(he_img) 

    image_cyx = np.transpose(img, (2, 0, 1))
    img_model = Image2DModel.parse(
        image_cyx,
        dims=("c", "y", "x"),
        transformations={"global": Identity()},
    )

    # get dimensions
    _, ydim, xdim = image_cyx.shape

    # create coordinate grid
    y_coords, x_coords = np.mgrid[0:ydim, 0:xdim]

    # flatten to pixel list
    points_df = pd.DataFrame({
        "x": x_coords.ravel(),
        "y": y_coords.ravel(),
    })

    points_model = PointsModel.parse(
        points_df,
        coordinates={"x": "x", "y": "y"},
        transformations={"global": Identity()},  # or Scale(...) if you want pixel scaling
    )

    sdata = SpatialData(
        images={"he_image": img_model},
        points={"pixels": points_model},
    )

    return(sdata)

