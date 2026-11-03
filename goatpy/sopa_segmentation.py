from __future__ import annotations

import warnings
import sopa
import numpy as np, cv2
import pandas as pd
import geopandas as gpd
import spatialdata as sd
import matplotlib.pyplot as plt
import scipy.sparse as sp

from anndata import AnnData
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


def fullres_to_sdata_affine(sdata, full_w, full_h, transform_key="he_transform"):
    """3x3 affine: full-res image coords -> sdata 'global' coords."""
    t = sdata["maldi_adata"].uns[transform_key]
    M_stored = np.array(t["affine_matrix"], dtype=float)   # reg px -> cropped canvas px
    us = float(t["img_upscaling"])
    ref_h, ref_w = t["he_reg_size"]                        # registered reference size (after ref_scale)
    sx, sy = ref_w / full_w, ref_h / full_h                # full-res px -> reg px

    to_reg = np.array([[sx, 0, -0.5], [0, sy, -0.5], [0, 0, 1]])
    up = np.array([[us, 0, 0.5 * us], [0, us, 0.5 * us], [0, 0, 1]])
    return up @ M_stored @ to_reg


def add_fullres_to_sdata(
    sdata,
    he,
    image_key="he_image_pyr",          # key in `he` (multiscale full-res image)
    shapes_keys=("cell_segmentation",),  # keys in `he` to carry across
    image_key_out="he_fullres",
    table_key="maldi_adata",
    coordinate_system="global",
    reload=True,
):
    """Add the full-res H&E (and shapes such as cell segmentation) from `he`
    into `sdata`, using the registration stored in sdata[table_key].uns['he_transform'].

    reload=True re-reads elements from he's zarr store so `he` itself is not modified.
    """
    if reload:
        if he.path is None:
            raise ValueError("`he` has no zarr store; write it first or use reload=False.")
        src = sd.read_zarr(he.path)
    else:
        src = he

    img = src[image_key]
    scale0 = img["scale0"]["image"] if hasattr(img, "keys") else img
    _, H, W = scale0.shape

    A = fullres_to_sdata_affine(sdata, W, H)
    T = Affine(A, input_axes=("x", "y"), output_axes=("x", "y"))

    set_transformation(img, T, to_coordinate_system=coordinate_system)
    sdata[image_key_out] = img

    for key in shapes_keys:
        shp = src[key]
        set_transformation(shp, T, to_coordinate_system=coordinate_system)
        sdata[key] = shp

    return sdata




def _get_polygons(sdata, name, coordinate_system):
    """Fetch a shapes element in the target coordinate system as polygons."""
    gdf = sd.transform(sdata.shapes[name], to_coordinate_system=coordinate_system)
    gdf = gdf.copy()

    # circles (Point + radius) -> polygons
    if (gdf.geometry.geom_type == "Point").all():
        if "radius" not in gdf.columns:
            raise ValueError(f"'{name}' contains points but has no 'radius' column.")
        gdf["geometry"] = gdf.geometry.buffer(gdf["radius"])

    # fix invalid geometries (common in segmentation output)
    invalid = ~gdf.geometry.is_valid
    if invalid.any():
        gdf.loc[invalid, "geometry"] = gdf.loc[invalid, "geometry"].buffer(0)

    gdf = gdf[~gdf.geometry.is_empty]
    return gdf[["geometry"]]


def pixels_to_cells(
    sdata: sd.SpatialData,
    pixel_layer: str = "pixels",
    cell_layer: str = "cell_segmentation",
    table_name: str = "maldi_adata",
    coordinate_system: str = "aligned",
    layer: str | None = None,
    key_added: str = "cell_glycans",
    min_coverage: float = 0.0,
    overwrite: bool = False,
    inplace: bool = True,
) -> AnnData:
    """
    Transfer MALDI pixel intensities onto segmented cells using area-weighted averaging.

    For each cell:  I_cell = sum_i(overlap_area_i * I_pixel_i) / sum_i(overlap_area_i)

    Parameters
    ----------
    sdata
        SpatialData object.
    pixel_layer
        Shapes layer holding the MALDI pixel footprints.
    cell_layer
        Shapes layer holding the single-cell polygons.
    table_name
        Table with the glycan intensities (annotating ``pixel_layer``).
    coordinate_system
        Coordinate system in which overlaps are computed.
    layer
        Layer of the table to use (default: ``.X``). Use a normalised layer.
    key_added
        Name of the new table annotating ``cell_layer``.
    min_coverage
        Drop cells where less than this fraction of the cell area overlaps
        MALDI pixels (0-1). Cells with no overlap are always dropped.
    overwrite
        Overwrite ``key_added`` if it already exists.
    inplace
        Add the table to ``sdata``. Always returns the AnnData.

    Returns
    -------
    AnnData (cells x glycans) with QC columns in ``.obs``:
        n_pixels            number of pixels overlapping the cell
        frac_cell_covered   fraction of cell area that overlaps MALDI pixels
        max_pixel_weight    weight of the dominant pixel (1.0 = cell sits in one pixel)
        cell_area           cell area in coordinate-system units
    """
    if key_added in sdata.tables and not overwrite:
        raise ValueError(f"Table '{key_added}' already exists. Use overwrite=True.")

    # ---- 1. intensity table, matched to pixel shapes -----------------------
    adata = sdata.tables[table_name]
    attrs = adata.uns["spatialdata_attrs"]
    region_key, instance_key = attrs["region_key"], attrs["instance_key"]

    adata_pix = adata[adata.obs[region_key] == pixel_layer]
    if adata_pix.n_obs == 0:
        raise ValueError(f"Table '{table_name}' has no rows annotating '{pixel_layer}'.")

    X = adata_pix.layers[layer] if layer is not None else adata_pix.X
    X = sp.csr_matrix(X) if sp.issparse(X) else np.asarray(X, dtype=float)
    if not sp.issparse(X) and np.isnan(X).any():
        warnings.warn(
            "NaNs found in intensity matrix; affected cells will be NaN. "
            "Consider filling or removing off-tissue pixels first."
        )

    # ---- 2. geometries in a common coordinate system ----------------------
    pix = _get_polygons(sdata, pixel_layer, coordinate_system)
    cel = _get_polygons(sdata, cell_layer, coordinate_system)

    # column j of the weight matrix <-> row j of X
    ids = adata_pix.obs[instance_key].to_numpy()
    pix = pix.reindex(ids)
    if pix.geometry.isna().any():
        raise ValueError("Some table rows have no matching shape in the pixel layer.")
    pix = pix.reset_index(drop=True)
    pix["pix_idx"] = np.arange(len(pix))

    cell_ids = cel.index.to_numpy()
    cel = cel.reset_index(drop=True)
    cel["cell_idx"] = np.arange(len(cel))
    cell_area = cel.geometry.area.to_numpy()
    
    # centroids in the same coordinate system as the overlap calculation
    centroids = cel.geometry.centroid
    xy = np.column_stack([centroids.x.to_numpy(), centroids.y.to_numpy()])   # x, y

        # ---- 3. overlap areas -> sparse weight matrix --------------------------
    ov = gpd.overlay(
        cel[["cell_idx", "geometry"]],
        pix[["pix_idx", "geometry"]],
        how="intersection",
        keep_geom_type=True,
    )
    ov["area"] = ov.geometry.area
    ov = ov[ov["area"] > 0]

    W = sp.csr_matrix(
        (ov["area"].to_numpy(), (ov["cell_idx"].to_numpy(), ov["pix_idx"].to_numpy())),
        shape=(len(cel), len(pix)),
    )
    W.sum_duplicates()
    covered = np.asarray(W.sum(axis=1)).ravel()   # cell area that overlaps data
    safe = np.where(covered > 0, covered, 1.0)

    # row-normalised weights; force CSR so .data/.indptr are reliable
    W_norm = sp.csr_matrix(sp.diags(1.0 / safe) @ W)
    W_norm.eliminate_zeros()

    # per-row max straight from the CSR structure
    row_counts = np.diff(W_norm.indptr)
    max_w = np.zeros(W_norm.shape[0])
    nonempty = row_counts > 0
    max_w[nonempty] = np.maximum.reduceat(W_norm.data, W_norm.indptr[:-1][nonempty])

    # sanity checks: every covered row should sum to 1
    row_sums = np.asarray(W_norm.sum(axis=1)).ravel()
    assert np.allclose(row_sums[covered > 0], 1.0), "Weights are not normalised."

    # ---- 4. transfer -------------------------------------------------------
    X_cells = W_norm @ X
    if sp.issparse(X_cells):
        X_cells = sp.csr_matrix(X_cells)

    obs = pd.DataFrame(
        {
            "n_pixels": row_counts,
            "frac_cell_covered": np.clip(covered / cell_area, 0, 1),
            "max_pixel_weight": max_w,
            "cell_area": cell_area,
        }
    )
    obs["cell_id"] = cell_ids
    obs["region"] = cell_layer

    # ---- 5. filter, build AnnData, register in sdata -----------------------
    keep = (covered > 0) & (obs["frac_cell_covered"].to_numpy() >= min_coverage)
    n_dropped = int((~keep).sum())
    if n_dropped:
        print(f"Dropping {n_dropped}/{len(keep)} cells with insufficient MALDI coverage.")

    obs = obs.loc[keep].reset_index(drop=True)
    obs.index = obs["cell_id"].astype(str).to_numpy()
    out = AnnData(X=X_cells[keep], obs=obs, var=adata_pix.var.copy())

    # cells x pixels weights, aligned to out.obs; columns <-> uns["pixel_ids"]
    out.obsm["pixel_weights"] = W_norm[keep]
    out.uns["pixel_ids"] = np.asarray(ids).astype(str)

    out.obsm["spatial"] = xy[keep]
    out.uns["spatial_coordinate_system"] = coordinate_system


    out.uns["pixels_to_cells"] = {
        "pixel_layer": pixel_layer,
        "cell_layer": cell_layer,
        "source_table": table_name,
        "source_layer": layer if layer is not None else "X",
        "coordinate_system": coordinate_system,
        "method": "area_weighted_mean",
        "min_coverage": min_coverage,
    }

    out = TableModel.parse(
        out, region=cell_layer, region_key="region", instance_key="cell_id"
    )

    if inplace:
        if key_added in sdata.tables:
            del sdata.tables[key_added]
        sdata.tables[key_added] = out
        if sdata.is_backed():
            sdata.write_element(key_added, overwrite=True)

    return out
    obs = pd.DataFrame(
        {
            "n_pixels": row_counts,
            "frac_cell_covered": np.clip(covered / cell_area, 0, 1),
            "max_pixel_weight": max_w,
            "cell_area": cell_area,
        }
    )
    obs["cell_id"] = cell_ids
    obs["region"] = cell_layer

    # ---- 5. filter, build AnnData, register in sdata -----------------------
    keep = (covered > 0) & (obs["frac_cell_covered"].to_numpy() >= min_coverage)
    n_dropped = int((~keep).sum())
    if n_dropped:
        print(f"Dropping {n_dropped}/{len(keep)} cells with insufficient MALDI coverage.")

    obs = obs.loc[keep].reset_index(drop=True)
    obs.index = obs["cell_id"].astype(str).to_numpy()
    out = AnnData(X=X_cells[keep], obs=obs, var=adata_pix.var.copy())

    # cells x pixels weights, aligned to out.obs; columns <-> uns["pixel_ids"]
    out.obsm["pixel_weights"] = W_norm[keep]
    out.uns["pixel_ids"] = np.asarray(ids).astype(str)

    out.uns["pixels_to_cells"] = {
        "pixel_layer": pixel_layer,
        "cell_layer": cell_layer,
        "source_table": table_name,
        "source_layer": layer if layer is not None else "X",
        "coordinate_system": coordinate_system,
        "method": "area_weighted_mean",
        "min_coverage": min_coverage,
    }

    out = TableModel.parse(
        out, region=cell_layer, region_key="region", instance_key="cell_id"
    )

    if inplace:
        if key_added in sdata.tables:
            del sdata.tables[key_added]
        sdata.tables[key_added] = out
        if sdata.is_backed():
            sdata.write_element(key_added, overwrite=True)

    return out
    obs = pd.DataFrame(
        {
            "n_pixels": row_counts,
            "frac_cell_covered": np.clip(covered / cell_area, 0, 1),
            "max_pixel_weight": max_w,
            "cell_area": cell_area,
        }
    )
    obs["cell_id"] = cell_ids
    obs["region"] = cell_layer

    # ---- 5. filter, build AnnData, register in sdata -----------------------
    keep = (covered > 0) & (obs["frac_cell_covered"].to_numpy() >= min_coverage)
    n_dropped = int((~keep).sum())
    if n_dropped:
        print(f"Dropping {n_dropped}/{len(keep)} cells with insufficient MALDI coverage.")

    obs = obs.loc[keep].reset_index(drop=True)
    obs.index = obs["cell_id"].astype(str).to_numpy()
    out = AnnData(X=X_cells[keep], obs=obs, var=adata_pix.var.copy())

    # cells x pixels weights, aligned to out.obs; columns <-> uns["pixel_ids"]
    out.obsm["pixel_weights"] = W_norm[keep]
    out.uns["pixel_ids"] = np.asarray(ids).astype(str)

    out.uns["pixels_to_cells"] = {
        "pixel_layer": pixel_layer,
        "cell_layer": cell_layer,
        "source_table": table_name,
        "source_layer": layer if layer is not None else "X",
        "coordinate_system": coordinate_system,
        "method": "area_weighted_mean",
        "min_coverage": min_coverage,
    }

    out = TableModel.parse(
        out, region=cell_layer, region_key="region", instance_key="cell_id"
    )

    if inplace:
        if key_added in sdata.tables:
            del sdata.tables[key_added]
        sdata.tables[key_added] = out
        if sdata.is_backed():
            sdata.write_element(key_added, overwrite=True)

    return out
    obs = pd.DataFrame(
        {
            "n_pixels": row_counts,
            "frac_cell_covered": np.clip(covered / cell_area, 0, 1),
            "max_pixel_weight": max_w,
            "cell_area": cell_area,
        }
    )
    obs["cell_id"] = cell_ids
    obs["region"] = cell_layer

    # ---- 5. filter, build AnnData, register in sdata -----------------------
    keep = (covered > 0) & (obs["frac_cell_covered"].to_numpy() >= min_coverage)
    n_dropped = int((~keep).sum())
    if n_dropped:
        print(f"Dropping {n_dropped}/{len(keep)} cells with insufficient MALDI coverage.")

    obs = obs.loc[keep].reset_index(drop=True)
    obs.index = obs["cell_id"].astype(str).to_numpy()
    out = AnnData(X=X_cells[keep], obs=obs, var=adata_pix.var.copy())

    # cells x pixels weights, aligned to out.obs; columns <-> uns["pixel_ids"]
    out.obsm["pixel_weights"] = W_norm[keep]
    out.uns["pixel_ids"] = np.asarray(ids).astype(str)

    out.uns["pixels_to_cells"] = {
        "pixel_layer": pixel_layer,
        "cell_layer": cell_layer,
        "source_table": table_name,
        "source_layer": layer if layer is not None else "X",
        "coordinate_system": coordinate_system,
        "method": "area_weighted_mean",
        "min_coverage": min_coverage,
    }

    out = TableModel.parse(
        out, region=cell_layer, region_key="region", instance_key="cell_id"
    )

    if inplace:
        if key_added in sdata.tables:
            del sdata.tables[key_added]
        sdata.tables[key_added] = out
        if sdata.is_backed():
            sdata.write_element(key_added, overwrite=True)

    return out
    obs = pd.DataFrame(
        {
            "n_pixels": row_counts,
            "frac_cell_covered": np.clip(covered / cell_area, 0, 1),
            "max_pixel_weight": max_w,
            "cell_area": cell_area,
        }
    )
    obs["cell_id"] = cell_ids
    obs["region"] = cell_layer

    # ---- 5. filter, build AnnData, register in sdata -----------------------
    keep = (covered > 0) & (obs["frac_cell_covered"].to_numpy() >= min_coverage)
    n_dropped = int((~keep).sum())
    if n_dropped:
        print(f"Dropping {n_dropped}/{len(keep)} cells with insufficient MALDI coverage.")

    obs = obs.loc[keep].reset_index(drop=True)
    obs.index = obs["cell_id"].astype(str).to_numpy()
    out = AnnData(X=X_cells[keep], obs=obs, var=adata_pix.var.copy())

    # cells x pixels weights, aligned to out.obs; columns <-> uns["pixel_ids"]
    out.obsm["pixel_weights"] = W_norm[keep]
    out.uns["pixel_ids"] = np.asarray(ids).astype(str)

    out.uns["pixels_to_cells"] = {
        "pixel_layer": pixel_layer,
        "cell_layer": cell_layer,
        "source_table": table_name,
        "source_layer": layer if layer is not None else "X",
        "coordinate_system": coordinate_system,
        "method": "area_weighted_mean",
        "min_coverage": min_coverage,
    }

    out = TableModel.parse(
        out, region=cell_layer, region_key="region", instance_key="cell_id"
    )

    if inplace:
        if key_added in sdata.tables:
            del sdata.tables[key_added]
        sdata.tables[key_added] = out
        if sdata.is_backed():
            sdata.write_element(key_added, overwrite=True)

    return out