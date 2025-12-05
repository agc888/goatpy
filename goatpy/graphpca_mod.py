"""
gpca_mod_spatial.py

GraphPCA replacement with spatial smoothing using cKDTree.
"""

from typing import Optional, Tuple
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from numpy.linalg import svd
from scipy.spatial import cKDTree
import warnings
from sklearn.cluster import KMeans

def kneighbors_graph_spatial(
    coords: np.ndarray,
    n_neighbors: int = 10,
    include_self: bool = False,
    mode: str = "connectivity",
) -> sparse.csr_matrix:
    """
    Build a KNN graph (sparse adjacency) using spatial coordinates.
    """
    coords = np.asarray(coords)
    n_samples = coords.shape[0]
    k = n_neighbors + (1 if include_self else 0)
    
    tree = cKDTree(coords)
    # Remove n_jobs argument
    dists, inds = tree.query(coords, k=k)
    
    rows, cols, data = [], [], []
    for i in range(n_samples):
        idxs = inds[i]
        d = dists[i]
        if not include_self:
            mask = idxs != i
            idxs = idxs[mask][:n_neighbors]
            d = d[mask][:n_neighbors]
        else:
            idxs = idxs[:k]
            d = d[:k]

        vals = np.ones_like(idxs) if mode == "connectivity" else d
        rows.extend([i]*len(idxs))
        cols.extend(idxs.tolist())
        data.extend(vals.tolist())
    
    A = sparse.csr_matrix((data, (rows, cols)), shape=(n_samples, n_samples))
    A = A.maximum(A.T)
    if not include_self:
        A.setdiag(0)
        A.eliminate_zeros()
    return A

def _laplacian_from_adjacency(A: sparse.spmatrix) -> sparse.spmatrix:
    degrees = np.array(A.sum(axis=1)).ravel()
    D = sparse.diags(degrees)
    return D - A


def _smooth_scores_on_graph(scores: np.ndarray, A: sparse.spmatrix, alpha: float) -> np.ndarray:
    if alpha <= 0:
        return scores.copy()
    L = _laplacian_from_adjacency(A)
    n = L.shape[0]
    I = sparse.eye(n, format="csr")
    M = I + alpha * L
    smoothed = np.zeros_like(scores)
    for j in range(scores.shape[1]):
        try:
            smoothed[:, j] = spsolve(M, scores[:, j])
        except Exception as e:
            warnings.warn(f"spsolve failed for component {j}: {e}, adding jitter")
            M2 = M + 1e-8 * sparse.eye(n)
            smoothed[:, j] = spsolve(M2, scores[:, j])
    return smoothed


def graphpca_spatialdata(
    sd,
    tables: str = "maldi_adata",
    library_id: Optional[str] = 'spatial',
    n_components: int = 50,
    n_neighbors: int = 10,
    alpha: float = 0.0,
    center: bool = True,
    kneighbors_mode: str = "connectivity",
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[sparse.csr_matrix]]:
    """
    PCA with optional spatial smoothing.
    
    Parameters
    ----------
    sd : spatialdata object containing glycomics data
    tables : str, table name in sd.tables
    library_id : str, key in adata.obsm for spatial coordinates
    n_components : int
    n_neighbors : int for spatial KNN
    alpha : float, smoothing strength
    center : bool
    return_scores : bool
    return_adjacency : bool
    
    Returns
    -------
    components : (n_components, n_features)
    scores : (n_samples, n_components)
    adjacency : CSR adjacency used for smoothing
    """
    adata = sd.tables[tables]
    X = adata.X
    location = adata.obsm[library_id]

    X = np.asarray(X, dtype=float)
    location = np.asarray(location, dtype=float)
    if center:
        Xc = X - X.mean(axis=0, keepdims=True)
    else:
        Xc = X.copy()

    # PCA
    U, S, Vt = svd(Xc, full_matrices=False)
    components = Vt[:n_components].copy()
    scores = (U[:, :n_components] * S[:n_components])

    adjacency = None
    if alpha > 0:
        adjacency = kneighbors_graph_spatial(location, n_neighbors=n_neighbors, 
                                             include_self=False, mode=kneighbors_mode)
        scores = _smooth_scores_on_graph(scores, adjacency, alpha)

        # refit components to smoothed scores
        comps = []
        for j in range(scores.shape[1]):
            z = scores[:, j]
            w, *_ = np.linalg.lstsq(Xc, z, rcond=None)
            comps.append(w)
        components = np.vstack(comps)

    sd.tables[tables].obsm["GraphPCA"] = scores

    return sd




def get_kmean_clusters(sd, tables: str = "maldi_adata", n_clusters=8, cluster_key: str = "GPCA_clusters"):
    """
    Perform KMeans clustering on GraphPCA scores stored in adata.obsm.
    
    Parameters
    ----------
    sd : SpatialData object with GraphPCA scores in sd.tables[tables].obsm["GraphPCA"]
    tables : str, table name in sd.tables (default "maldi_adata")
    n_clusters : int, number of clusters for KMeans (default 8)
    cluster_key : str, key for storing cluster labels in adata.obs (default "GPCA_clusters")
    
    
    Returns
    -------
    adata with cluster labels in adata.obs["GPCA_pred"]
    """

    adata = sd.tables[tables]
    scores = adata.obsm["GraphPCA"]
        
    # Run KMeans on GPCA scores
    estimator = KMeans(n_clusters=n_clusters)
    res = estimator.fit(scores)  # Z is (n_samples, n_components)
    label_pred = res.labels_

    # Save predictions to AnnData
    sd.tables[tables].obs[cluster_key] = label_pred
    sd.tables[tables].obs[cluster_key] = sd.tables[tables].obs[cluster_key].astype('category')

    return sd