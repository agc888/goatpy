import sopa
import numpy as np
import pandas as pd
import geopandas as gpd
import spatialdata as sd

from shapely.affinity import affine_transform
from spatialdata.transformations import get_transformation
from spatialdata.models import ShapesModel, TableModel


def prep_pixel_segmentation(sdata, image_key="he_image", coordinate_system="aligned", roi_key="pixels"):
    """
    Get the 'pixels' shapes in the raw pixel space of the given image.
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

    sopa.make_image_patches(sdata,patch_width=2000,patch_overlap=50,roi_key="in_maldi_pixels",image_key="he_image")

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