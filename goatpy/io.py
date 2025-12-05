import numpy as np
from pyimzml.ImzMLParser import ImzMLParser, getionimage
from joblib import Parallel, delayed
from functools import partial
import anndata as ad
import pandas as pd
import numpy as np
from anndata import AnnData
from spatialdata import SpatialData
from spatialdata.models import PointsModel, Image2DModel, TableModel
from spatialdata.transformations import Identity
import pkg_resources
import os



def parmap(f, X, nprocs=None):
    """
    Parallel map using joblib (more robust for Jupyter).
    
    Parameters
    ----------
    f : callable
        Function to apply to each element
    X : iterable
        Input data
    nprocs : int, optional
        Number of processes (default: -1, all CPUs)
    
    Returns
    -------
    list
        Results in same order as input
    """
    if nprocs is None:
        nprocs = -1  # Use all CPUs
    
    return Parallel(n_jobs=nprocs, backend='loky')(
        delayed(f)(x) for x in X
    )


def getimage(peak, path): 
    p =  ImzMLParser(path) #individual file pointers otherwise parsing is corrupted
    return getionimage(p, peak, tol=0.1, reduce_func=max)
    

def rd_peaks(fn):
    data = []
    with open(fn) as f:
        f.readline() #header
        for line in f:
            ss = line.split()
            if ss[0].strip('"') == 'M': continue
            data.append(float(ss[1]))
    return data


def rd_peaks_from_package():

    # Try to get the file from the package
    peaks_path = pkg_resources.resource_filename('goatpy', 'data/PEAKS.csv')
    
    if not os.path.exists(peaks_path):
        raise FileNotFoundError(f"PEAKS.csv not found at {peaks_path}")
    
    with open(peaks_path, 'r') as f:
        data = []
        f.readline()  # skip header
        for line in f:
            ss = line.split()
            if ss[0].strip('"') == 'M': 
                continue
            data.append(float(ss[1]))
    return data


def glyco_spatialdata(imzml_path, peaks_path = None):

    # Load Peaks
    if peaks_path is None:
        peaks = rd_peaks_from_package()
    else:
        peaks = rd_peaks(peaks_path)

    # Load ImzML data
    getimg = partial(getimage, path=imzml_path)

    spectra_all = np.stack(
        parmap(getimg, peaks, 10),
        axis=-1
    )

    # Load Spatial Info
    p = ImzMLParser(imzml_path)
    coords = np.array(p.coordinates)[:, :2]  # (x, y)
    coords = coords - 1  # convert from 1-based to 0-based indexing
    



    # Create AnnData Object
    spectra_flat = np.array([spectra_all[y-1, x-1, :] for x, y in coords])
    anndata = ad.AnnData(spectra_flat, dtype=np.float32)
    anndata.var_names = np.array(["%.1f" % p for p in peaks])
    anndata.obs_names = np.array(list(map(str, range(len(coords)))))
    anndata.obs["full_x"] = coords[:, 0]
    anndata.obs["full_y"] = coords[:, 1]

    anndata.obs["x"] = anndata.obs["full_x"] - anndata.obs["full_x"].min() 
    anndata.obs["y"] = anndata.obs["full_y"] - anndata.obs["full_y"].min()
    
    anndata.obsm["spatial"] = np.column_stack([anndata.obs["x"], anndata.obs["y"]])


    # Calculate Total Ion Count (TIC)
    anndata.obs["TIC"] = np.ravel(anndata.X.sum(axis=1))

    
    # Create SpatialData Object
    coords = pd.DataFrame({
        "x": [c for c in anndata.obs["x"]],
        "y": [c for c in anndata.obs["y"]],
    })

    coords["point_id"] = coords.index.astype(str)      # unique ID for each pixel
    coords["region"] = "maldi_pixels"                  # must exist for TableModel

    df = pd.concat(
        [
            coords.reset_index(drop=True),
            pd.DataFrame(anndata.X, columns=("mz-" + anndata.var.index))
        ],
        axis=1
    )

    
    points = PointsModel.parse(df)
    sdata = SpatialData(points={"maldi_pixels": points})

    adata = AnnData(
        X=anndata.X,
        obs=coords,  # contains x, y, point_id, region
        var=pd.DataFrame(index=("mz-" + anndata.var.index))
    )

    adata.obs = pd.concat(
        [
            adata.obs.reset_index(drop=True),
            anndata.obs.drop(columns=["x", "y"]).iloc[:adata.n_obs].reset_index(drop=True)
        ],
        axis=1
    )

    coords = np.array(adata.obs[["x", "y"]])

    adata.obsm["spatial"] = coords

    table = TableModel.parse(
        adata,
        region="maldi_pixels",      # name of your PointsModel
        region_key="region",        # must exist in adata.obs
        instance_key="point_id"     # unique per row
    )

    # --- 7. Add to SpatialData ---
    sdata.tables["maldi_adata"] = table

    return sdata




