import warnings
import sopa
import pandas as pd
import numpy as np
import scanpy as sc
import geopandas as gpd

from anndata import AnnData
from spatialdata import SpatialData
from spatialdata.models import Image2DModel, ShapesModel
from spatialdata.transformations import Identity
from shapely.affinity import scale as shapely_scale

from PIL import Image
from PIL.ExifTags import TAGS


def he_spatialdata(he_image_path, backend = 'tiffslide'):

    try:
        sdata = sopa.io.wsi(he_image_path,  backend=backend)
        # Rename the image key to 'he_image'
        old_key = list(sdata.images.keys())[0]
        sdata["he_image"] = sdata.images[old_key]
        del sdata[old_key]
        print("Successfully loaded with sopa.io.wsi")
        return sdata
    
    except Exception as e:
        warnings.warn(f"sopa.io.wsi failed ({type(e).__name__}: {e}). Falling back to PIL method.")
    
    # Fallback method
    he_img = Image.open(he_image_path)
    img = np.array(he_img)
    image_cyx = np.transpose(img, (2, 0, 1))
    img_model = Image2DModel.parse(
        image_cyx,
        dims=("c", "y", "x"),
        transformations={"global": Identity()},
    )
    sdata = SpatialData(
        images={"he_image": img_model},
    )
    return sdata


def add_qupath_annotations(sdata, geojson_path, name = "Annotations"):
    
    gdf = gpd.read_file(geojson_path)   
    gdf["colour"] = gdf["classification"].apply(lambda x: x["color"])
    gdf["classification"] = pd.Categorical(gdf["classification"].apply(lambda x: x["name"]))

    shapes = ShapesModel.parse(gdf)
    sdata.shapes[name] = shapes

    return(sdata)
