import numpy as np
from spatialdata import SpatialData
from copy import deepcopy

def normalize_matrix(X, method="TIC", eps=1e-12):
    """
    Normalize a (n_pixels × n_mz) matrix.
    """
    if method.upper() == "TIC":
        denom = X.sum(axis=1, keepdims=True)
    elif method.upper() == "RMS":
        denom = np.sqrt((X ** 2).mean(axis=1, keepdims=True))
    else:
        raise ValueError("method must be 'TIC' or 'RMS'")

    return X / (denom + eps)




def normalize_spatialdata(
    sdata: SpatialData,
    table_name: str,
    method: str = "TIC",
    inplace: bool = False,
    ):
    """
    Normalize intensities per pixel in a SpatialData object.

    Parameters
    ----------
    sdata : SpatialData
    table_name : str
        Name of the table containing intensities (pixels × mz)
    method : 'TIC' | 'RMS'
    inplace : bool
        If False, returns a new SpatialData object
    """

    if not inplace:
        sdata = deepcopy(sdata)

    table = sdata.tables[table_name]

    X = table.X  # NumPy or Dask array

    X_norm = normalize_matrix(X, method=method)

    table.X = X_norm

    # Record provenance
    table.uns["normalization"] = {
        "method": method,
        "axis": "per-pixel",
    }

    return sdata
