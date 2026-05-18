from spatialdata import SpatialData
import numpy as np
import pandas as pd

from pathlib import Path
import pkg_resources
from collections import defaultdict


def annotate_glycans(
    sdata,
    glycan_list=None,
    threshold=0.5,
    adata_slot="maldi_adata",
    duplicate_glycans="combine",
    print_stats=True
):

    adata = sdata[adata_slot].copy()
    original_mz = adata.var_names.astype(str).copy()

    names = []

    if glycan_list is None:

        path = pkg_resources.resource_filename('goatpy', 'data/glycan_list.csv')
        glycans = pd.read_csv(path)

    else:

        path = Path(glycan_list)

        if not path.exists() or not path.is_file():
            raise ValueError(
                f"{glycan_list} is not a valid file path"
            )

        if path.suffix.lower() != ".csv":
            raise ValueError(
                f"{glycan_list} is not a CSV file"
            )

        glycans = pd.read_csv(path)

        if glycans.shape[1] != 2:
            raise ValueError(
                f"{glycan_list} must contain exactly "
                f"2 columns, found {glycans.shape[1]}"
            )

    for mz in original_mz:

        gly_name = []

        mz_float = np.float64(mz)

        upper = mz_float + threshold
        lower = mz_float - threshold

        for idx, x in enumerate(
            np.array(glycans.iloc[:, 0])
        ):

            if lower <= x <= upper:

                gly_name.append(
                    np.array(glycans.iloc[:, 1])[idx]
                )

        # Multiple annotations
        if len(gly_name) > 1:
            names.append(", ".join(gly_name))

        # No annotations
        elif len(gly_name) < 1:
            names.append(f"mz-{mz_float}")

        # Single annotation
        else:
            names.append(gly_name[0])


    adata.var_names = names

    # Statistics
    var_names = list(adata.var_names)
    unannotated = [
        x for x in var_names
        if str(x).startswith("mz-")
    ]

    annotated = [
        x for x in var_names
        if not str(x).startswith("mz-")
    ]

    multi_annotation = [
        x for x in annotated
        if "," in str(x)
    ]

    glycan_to_mz = defaultdict(list)
    for mz, ann in zip(original_mz, var_names):

        if str(ann).startswith("mz-"):
            continue

        split_ann = [
            x.strip()
            for x in str(ann).split(",")
        ]

        for glycan in split_ann:

            glycan_to_mz[glycan].append(mz)


    duplicate_annotations = {
        k: v
        for k, v in glycan_to_mz.items()
        if len(v) > 1
    }

    stats = {

        "total_peaks": len(var_names),
        "annotated_peaks": len(annotated),
        "unannotated_peaks": len(unannotated),
        "multi_annotation_peaks": len(
            multi_annotation
        ),
        "unique_glycans": len(glycan_to_mz),
        "duplicate_glycans": len(
            duplicate_annotations
        ),
        "duplicate_glycan_mz": duplicate_annotations,
        "annotation_rate": round(
            len(annotated)
            / len(var_names)
            * 100,
            2
        )
    }


    if print_stats:

        print("\nAnnotation Statistics")
        print("-" * 40)
        for k, v in stats.items():

            if k != "duplicate_glycan_mz":

                print(f"{k}: {v}")

        print("\nDuplicate Glycans")
        print("-" * 40)

        if len(duplicate_annotations) == 0:

            print("None")
        else:

            for glycan, mzs in (
                duplicate_annotations.items()
            ):

                mz_string = ", ".join(mzs)

                print(
                    f"{glycan}: {mz_string}"
                )

    adata.uns[
        "glycan_annotation_stats"
    ] = stats


    # Handle duplicate var_names
    duplicated = adata.var_names.duplicated()

    if duplicated.any():
        if duplicate_glycans == "combine":

            X = adata.X

            if not isinstance(
                X,
                np.ndarray
            ):

                X = X.toarray()

            df = pd.DataFrame(
                X,
                columns=adata.var_names
            )

            # Combine duplicate glycans
            df_combined = (
                df.groupby(
                    axis=1,
                    level=0
                ).sum()
            )

            adata = adata[
                :,
                :df_combined.shape[1]
            ].copy()

            adata.X = (
                df_combined.values
            )

            adata.var_names = (
                df_combined.columns
            )

        elif duplicate_glycans == "separate":

            adata.var_names_make_unique()

        else:

            raise ValueError(
                "duplicate_glycans must "
                "be either "
                "'combine' or 'separate'"
            )

    sdata[adata_slot] = adata

    return sdata



def merge_spatialdata(
    sdatas: list,
    batch_names: list[str],
    table_name: str = "maldi_adata",
    offset_coords: bool = True,
    feature_join: str = "inner",
) -> "SpatialData":
    """
    Merge a list of SpatialData objects into a single SpatialData object,
    suffixing each element key with its batch name to avoid collisions.

    Parameters
    ----------
    sdatas : list of SpatialData
    batch_names : list of str
        Must match length of sdatas. Used as suffix for all element keys
        and added to adata.obs["batch"].
    table_name : str
        Key of the AnnData table in each SpatialData object.
    offset_coords : bool
        If True, offset he_x / he_y / spatial coords so batches sit
        side-by-side rather than overlapping in the merged canvas.
    feature_join : "inner" | "outer"
        How to handle m/z features not present in all samples.

        * ``"inner"`` (default) — keep only features common to ALL samples.
          Any m/z not found in every batch is dropped. The resulting matrix
          has no missing values.
        * ``"outer"`` — keep ALL features across all samples. Pixels from a
          batch that lacks a feature are filled with 0. Useful when samples
          were acquired with slightly different peak lists.

    Returns
    -------
    SpatialData

    Examples
    --------
    >>> # Only common m/z features (default)
    >>> merged = merge_spatialdata([sdata1, sdata2], ["batch1", "batch2"])

    >>> # Keep all features, fill missing with 0
    >>> merged = merge_spatialdata(
    ...     [sdata1, sdata2], ["batch1", "batch2"], feature_join="outer"
    ... )
    """
    import numpy as np
    import anndata as ad
    import geopandas as gpd
    import pandas as pd
    import re
    from shapely.affinity import translate
    from spatialdata import SpatialData
    from spatialdata.models import Image2DModel, PointsModel, ShapesModel, TableModel
    from spatialdata.transformations import Identity

    if len(sdatas) != len(batch_names):
        raise ValueError(
            f"sdatas ({len(sdatas)}) and batch_names ({len(batch_names)}) "
            f"must have the same length."
        )

    if feature_join not in ("inner", "outer"):
        raise ValueError(
            f"feature_join must be 'inner' or 'outer', got '{feature_join}'."
        )

    # Validate batch names
    for b in batch_names:
        if not re.match(r'^[A-Za-z0-9_.\-]+$', b):
            raise ValueError(
                f"batch_name '{b}' contains invalid characters. "
                f"Use only alphanumeric characters, underscores, dots or hyphens."
            )

    all_images = {}
    all_points = {}
    all_shapes = {}
    all_adatas = []
    x_offset   = 0.0

    def _make_key(base: str, batch: str) -> str:
        safe_base = re.sub(r'[^A-Za-z0-9_.\-]', '_', base)
        return f"{safe_base}_{batch}"

    def _strip_transforms(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        gdf = gdf.copy()
        gdf.attrs = {}
        drop_cols = [c for c in gdf.columns if c.startswith("transform")]
        return gdf.drop(columns=drop_cols, errors="ignore")

    # ----------------------------------------------------------------
    # Report feature overlap before merging
    # ----------------------------------------------------------------
    all_var_sets = [set(sd.tables[table_name].var_names) for sd in sdatas]
    common_features = set.intersection(*all_var_sets)
    all_features    = set.union(*all_var_sets)
    unique_per_batch = {
        b: all_var_sets[i] - common_features
        for i, b in enumerate(batch_names)
    }

    print("Feature summary:")
    print(f"  Common to all batches   : {len(common_features):,}")
    print(f"  Total across all batches: {len(all_features):,}")
    for b, uniq in unique_per_batch.items():
        print(f"  Unique to {b:20s}: {len(uniq):,}")
    print(
        f"  feature_join='{feature_join}' → keeping "
        f"{'common features only' if feature_join == 'inner' else 'all features (filling missing with 0)'}"
    )

    # ----------------------------------------------------------------
    # Main loop — images, shapes, points, AnnData per batch
    # ----------------------------------------------------------------
    for sdata, batch in zip(sdatas, batch_names):

        adata = sdata.tables[table_name].copy()
        n_obs = adata.n_obs

        if offset_coords and "he_x" in adata.obs.columns:
            batch_width = float(adata.obs["he_x"].max()) + 100.0
        else:
            batch_width = 0.0

        # Images
        for key, img in sdata.images.items():
            new_key = _make_key(key, batch)
            try:
                img_np = img["scale0"].ds[
                    list(img["scale0"].ds.data_vars)[0]
                ].values
            except Exception:
                img_np = img.values if hasattr(img, "values") else np.array(img)
            if img_np.ndim == 4:
                img_np = img_np[0]
            all_images[new_key] = Image2DModel.parse(
                img_np,
                dims=("c", "y", "x"),
                transformations={"global": Identity()},
            )

        # Shapes
        for key, gdf in sdata.shapes.items():
            new_key  = _make_key(key, batch)
            gdf_copy = _strip_transforms(gdf)
            if offset_coords and x_offset != 0.0:
                gdf_copy["geometry"] = gdf_copy["geometry"].apply(
                    lambda geom: translate(geom, xoff=x_offset, yoff=0.0)
                )
            all_shapes[new_key] = ShapesModel.parse(
                gdf_copy,
                transformations={"global": Identity()},
            )

        # Points
        for key, pts in sdata.points.items():
            new_key = _make_key(key, batch)
            pts_df  = pts.compute() if hasattr(pts, "compute") else pts.copy()
            pts_df  = pts_df.reset_index(drop=True)
            pts_df  = pts_df.drop(
                columns=[c for c in pts_df.columns if c.startswith("transform")],
                errors="ignore",
            )
            if offset_coords and x_offset != 0.0 and "x" in pts_df.columns:
                pts_df["x"] = pts_df["x"] + x_offset
            all_points[new_key] = PointsModel.parse(
                pts_df,
                transformations={"global": Identity()},
            )

        # AnnData
        pixel_region_key    = _make_key("pixels", batch)
        adata.obs["batch"]  = batch
        adata.obs["batch"]  = adata.obs["batch"].astype("category")
        adata.obs["region"] = pixel_region_key
        adata.obs["region"] = adata.obs["region"].astype("category")
        adata.obs_names     = [f"{batch}_{i}" for i in range(n_obs)]
        adata.uns.pop("spatialdata_attrs", None)

        if offset_coords and x_offset != 0.0:
            if "he_x" in adata.obs.columns:
                adata.obs["he_x"] = adata.obs["he_x"] + x_offset
            if "spatial" in adata.obsm:
                adata.obsm["spatial"] = adata.obsm["spatial"].copy()
                adata.obsm["spatial"][:, 0] += x_offset

        all_adatas.append(adata)
        x_offset += batch_width

    # ----------------------------------------------------------------
    # Concatenate AnnData with chosen join strategy
    # ----------------------------------------------------------------
    merged_adata = ad.concat(
        all_adatas,
        join=feature_join,  # "inner" drops unique features; "outer" fills with 0
        fill_value=0.0,     # only applied when join="outer"
    )
    merged_adata.obs["instance_id"] = np.arange(len(merged_adata)).astype(str)

    all_pixel_regions = [_make_key("pixels", b) for b in batch_names]
    merged_adata.obs["region"] = pd.Categorical(
        merged_adata.obs["region"],
        categories=all_pixel_regions,
    )

    # ----------------------------------------------------------------
    # Build fresh SpatialData
    # ----------------------------------------------------------------
    merged = SpatialData(
        images=all_images,
        points=all_points,
        shapes=all_shapes,
    )

    merged[table_name] = TableModel.parse(
        merged_adata,
        region=all_pixel_regions,
        region_key="region",
        instance_key="instance_id",
    )

    print(f"\nMerged {len(sdatas)} SpatialData objects:")
    for b, sd in zip(batch_names, sdatas):
        print(f"  {b}: {sd.tables[table_name].n_obs:,} pixels")
    print(f"  Total pixels  : {len(merged_adata):,}")
    print(f"  Total features: {merged_adata.n_vars:,}")
    print(f"  Images : {list(all_images.keys())}")
    print(f"  Shapes : {list(all_shapes.keys())}")
    print(f"  Points : {list(all_points.keys())}")

    return merged


