from PIL import Image
import numpy as np
import pandas as pd
from functools import partial

from spatialdata import SpatialData
from spatialdata.models import PointsModel, Image2DModel, TableModel, ShapesModel
from spatialdata.transformations import Identity
from shapely.geometry import box
import geopandas as gpd

from .io import *
from .registration import *



def get_crop(img, cutoff=0.1):
    sm = np.sum(img, axis=0)
    #print("HELLO", sm)
    mr = max(np.where(sm<cutoff)[0])
    sm = np.sum(img, axis=1)
    #print("HI", sm)
    mc = max(np.where(sm<cutoff)[0])
    return mr, mc



def normalize_maldi_img(maldi_img, mask=None, scale=False):
    #return maldi_img
    from scipy.ndimage import convolve
    import matplotlib.pyplot as plt
    kernel = np.ones((40, 40))    
    #fig, ax = plt.subplots(1,3, figsize=(10,30))
    #ax[0].matshow(maldi_img)

    if mask is not None:
        sumval = convolve(maldi_img*mask.astype(np.float32) , kernel, mode='reflect')
        normval = convolve(mask.astype(np.float32), kernel, mode='reflect')
        mean = sumval/normval
        
        #plt.matshow(normval)
        #plt.colorbar()
        #plt.show()
    else:
        mean = convolve(maldi_img, kernel, mode='reflect')/kernel.size
    if scale:
        squared_diff = (maldi_img - mean)**2
        std = np.sqrt(convolve(squared_diff, kernel, mode='reflect')/kernel.size)
        maldi_img = (maldi_img-mean)/std
    else:
        maldi_img = maldi_img - mean


    #ax[1].matshow(np.clip(maldi_img, 0, None))
    #ax[2].matshow(mean)
    #plt.colorbar()
    #plt.show()
    return maldi_img


def generate_MPI_img(shp, adata):
    cluster_label = adata.obs['MPI'].to_numpy().astype(np.int32)
    cluster_id = adata.obs.index.to_numpy().astype(np.int32)
    print(cluster_label.shape)

    cluster_img = np.zeros((shp[0], shp[1]), dtype=np.int32)

    for k,v in zip(map(int, cluster_id), cluster_label):
        idx = np.unravel_index(k, (shp[0], shp[1]))
        cluster_img[idx[0], idx[1]] = v+1
    return cluster_img


def load_and_align(imzml_path, 
                   he_path,
                   extra_img_path = None,
                   peaks_path = None, 
                   target_resolution = 0.5, 
                   maldi_pixel_size = 10,
                   maldi_threshold = 1,
                   sparsity = 0.3,
                   he_pixel_size = 0.2527,
                   he_threshold = 200,
                   f_pixel_size = 0.5,
                   f_threshold = 50,
                   buffer = 2000
                   ):


    scale_factor = int(maldi_pixel_size/target_resolution) #integer may be critical here
    he_ratio=target_resolution/he_pixel_size
    f_ratio = target_resolution/f_pixel_size

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

    p = ImzMLParser(imzml_path)


    ave_spec = {}
    TIC = np.zeros((p.imzmldict['max count of pixels y'], p.imzmldict['max count of pixels x']))
    ave_spec['mean_spectrum'] = np.zeros_like(p.getspectrum(0)[1])
    ave_spec['mzs'] = p.getspectrum(0)[0]
    count = 0

    #matrix_peak = getimage(1289.15)
    for idx, (x,y,z) in enumerate(p.coordinates):
        mzs, spectrum = p.getspectrum(idx)
        TIC[y-1,x-1] = np.sum(spectrum)
        if TIC[y-1,x-1]>0: # and matrix_peak[y-1,x-1]<0.3:
            #ave_spec['mean_spectrum'] += spectrum #Aligned spectrum can't show
            count+=1
            
    ave_spec['mean_spectrum'] = np.zeros_like(ave_spec['mzs'])

    maldi_img = np.nansum(spectra_all, axis=-1)
    try: 
        mr, mc = get_crop(maldi_img, cutoff=0.5) #cells
    except ValueError: #None below threshold
        mr, mc = 0, 0

    maldi_img = maldi_img[mc:, mr:]
    spectra_all = spectra_all[mc:,mr:,:]

    ### Load Images

    maldi_img = normalize_maldi_img(maldi_img, scale=True)
    maldi_mask = maldi_img<0.25 #binarized cell area based on local-contrast enhanced z-score

    #spectra_all = np.maximum(spectra_all - 0.2, 0)
    id_dict = {(y-1-mc, x-1-mr): i for i, (x, y, z_) in enumerate(p.coordinates) if ((x-1-mr)>=0 and (y-1-mc)>=0)}
    he_img = Image.open(he_path)
    images = he_img.split()

    #log_message("Starting Part3 ...")
    #Binarize images and map to shared resolution
    maldi_img_array = prepare_maldi_img(maldi_img, scale_factor, maldi_threshold, sparsity=sparsity)
    he_img_bin, buffered_he_shp, target_he_size = prepare_he_img(he_img, buffer=buffer, he_ratio=he_ratio, he_threshold=he_threshold)
    #f_img_bin, target_f_shp, target_f_size = prepare_f_img(f_img, f_ratio=f_ratio, f_threshold=f_threshold)

    #Find common rotation between MALDI and HE
    he_container, idx, _, max_he_rotation, max_hbuff_x, max_hbuff_y = \
        optimize_he_rotation_to_maldi(he_img_bin, maldi_img_array, [180, -1.0, -0.5, 0, 0.5, 1.0, 1.5], target_he_size, buffered_he_shp, buffer=buffer) #-0.15, 180 (slide1)
        #optimize_he_rotation_to_maldi(he_img_bin, maldi_img_array, [0.], target_he_size, buffered_he_shp, buffer=buffer) #-0.15, 180 (slide1)
        #optimize_he_rotation_to_maldi(he_img_bin, maldi_img_array, [180], target_he_size, buffered_he_shp, buffer=buffer) #-0.15, 180 (slide1)
        

    #Map original images to shared resolution and rotation
    he_img_resize = he_img.resize(target_he_size, resample=Image.Resampling.NEAREST)
    he_img_container = put_img_in_container(he_img_resize, (*buffered_he_shp, 3), max_hbuff_x, -max_hbuff_x, 
        max_hbuff_y, -max_hbuff_y, max_he_rotation, dtype=np.uint8)

    mzs = ["%.1f"%peak for peak in  peaks[-5:]]

    if extra_img_path: 
        segment_img = Image.open(extra_img_path)
        segment_img = segment_img.resize(target_he_size, resample=Image.Resampling.NEAREST)
        cell_id_container = put_img_in_container(segment_img, buffered_he_shp, max_hbuff_x, -max_hbuff_x, 
            max_hbuff_y, -max_hbuff_y, max_he_rotation, dtype=np.int32)
    else:
        cell_id_container = np.zeros((he_img_container.shape[0], he_img_container.shape[1]), dtype=np.int32)
  



    ### Create AnnData Object

    spectra_shp = spectra_all.shape
    adata = ad.AnnData(spectra_all.reshape(-1, spectra_all.shape[-1]), dtype=np.float32)
    adata.var_names = np.array(["%.1f"%p for p in peaks])
    adata.obs_names = np.array(list(map(str, range(spectra_shp[0]*spectra_shp[1]))))


    # Extract original coordinates from ImzML
    coords = np.array(p.coordinates)[:, :2]  # (x, y)
    coords = coords - 1  # convert from 1-based to 0-based indexing

    # Convert to integer indices
    x_coords, y_coords = coords[:, 0].astype(int), coords[:, 1].astype(int)

    # Reshape spectra_all to match how adata rows were created
    h, w, _ = spectra_all.shape

    # Build coordinate grids corresponding to flattened order (y-major)
    yy, xx = np.mgrid[mc:h+mc, mr:w+mr]
    yy = yy.flatten()
    xx = xx.flatten()

    # Add them as columns to adata.obs
    adata.obs["x"] = xx
    adata.obs["y"] = yy

    # Also store as spatial coordinates for plotting
    adata.obsm["spatial"] = np.vstack([xx, yy]).T
    adata.obs["MPI"] = np.ravel(adata.X.sum(axis=1))


    cluster_img = generate_MPI_img(spectra_shp, adata)
    cluster_img0 = np.asarray(Image.fromarray(cluster_img).resize((cluster_img.shape[1]*scale_factor, cluster_img.shape[0]*scale_factor), resample=Image.Resampling.NEAREST))
    container = np.zeros((cell_id_container.shape[0], cell_id_container.shape[1]), dtype=np.int32)
    container[idx[0]:(idx[0]+cluster_img0.shape[0]), idx[1]:(idx[1]+cluster_img0.shape[1])] = cluster_img0

    cell_id = adata.obs.index.to_numpy().astype(np.int32)
    cell_map = np.zeros((maldi_img.shape[0], maldi_img.shape[1]), dtype=np.int32)
    for k in cell_id:
        cell_idx = np.unravel_index(k, (maldi_img.shape[0], maldi_img.shape[1]))
        cell_map[cell_idx[0], cell_idx[1]] = k
    he_segment = np.asarray(Image.fromarray(cell_map).resize((scale_factor*cell_map.shape[1],scale_factor*cell_map.shape[0]), resample=Image.Resampling.NEAREST))
    cell_id_container[idx[0]:(idx[0]+he_segment.shape[0]), idx[1]:(idx[1]+he_segment.shape[1])] = he_segment


    ### Generate SpatialData Object

    # Find indices where values are non-zero
    y, x = np.nonzero(cell_id_container)

    # Build DataFrame
    df = pd.DataFrame({
        "x": x,
        "y": y,
        "value": cell_id_container[y, x]
    })

    # df: x, y, value  (value == adata.obs.index)
    df_points = df.rename(columns={"value": "cell_id"})
    df_points["cell_id"] = df_points["cell_id"].astype(str)

    valid_cell_ids = df_points["cell_id"].unique()

    # Filter adata to only these cells
    adata = adata[valid_cell_ids, :].copy()
    adata.obs.index = adata.obs.index.astype(str)

    pixel_boxes = (
        df_points
        .groupby("cell_id")
        .agg(
            x_min=("x", "min"),
            x_max=("x", "max"),
            y_min=("y", "min"),
            y_max=("y", "max"),
        )
        .reset_index()
    )

    pixel_boxes["x_centroid"] = (pixel_boxes["x_min"] + pixel_boxes["x_max"]) / 2
    pixel_boxes["y_centroid"] = (pixel_boxes["y_min"] + pixel_boxes["y_max"]) / 2

    centroids = PointsModel.parse(
        pixel_boxes[["x_centroid", "y_centroid", "cell_id"]]
            .rename(columns={"x_centroid": "x", "y_centroid": "y"}),
    )

    pixel_boxes["geometry"] = pixel_boxes.apply(
        lambda r: box(
            r.x_min,
            r.y_min,
            r.x_max + 1,  # +1 to make inclusive
            r.y_max + 1,
        ),
        axis=1,
    )


    adata.obs["adata_idx"] = adata.obs.index
    adata_df = adata.obs
    adata_df.index = adata_df.index.astype(str)

    pixel_boxes["MPI"] = pixel_boxes["cell_id"].map(adata_df["MPI"])
    pixel_boxes["adata_idx"] = pixel_boxes["cell_id"].map(adata_df["adata_idx"])


    pixel_boxes = (
        pixel_boxes
        .set_index("adata_idx")
        .loc[adata.obs.index]
    )

    # pixel_boxes already has: cell_id, geometry (shapely)
    gdf = gpd.GeoDataFrame(pixel_boxes[["cell_id","MPI", "geometry"]], geometry="geometry")


    shapes = ShapesModel.parse(
        gdf,
        transformations={"global": Identity()},
    )


    image_cyx = np.transpose(he_img_container, (2, 0, 1))
    img_model = Image2DModel.parse(
        image_cyx,
        dims=("c", "y", "x"),
        transformations={"global": Identity()},
    )   

    sdata = SpatialData(
    images={"he_image": img_model},
    points={"centroids": centroids},
    shapes={"pixels": shapes},
    )

    adata.obs["instance_id"] = sdata["pixels"].index
    adata.obs["region"] = "pixels"
    adata.obs["region"].astype("category")

    table = TableModel.parse(adata, region="pixels", region_key="region", instance_key="instance_id")
    sdata["maldi_adata"] = table


    return sdata
