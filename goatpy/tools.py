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
    feature_join: str = "inner",
) -> "SpatialData":
    """
    Merge a list of SpatialData objects into a single SpatialData object,
    suffixing each element key with its batch name to avoid collisions.

    SpatialData elements (images, shapes, points) are never modified.
    When offset_coords=True, only adata.obsm["spatial"] and adata.uns["spatial"]
    are shifted so batches appear side-by-side in scanpy/squidpy plots.

    Parameters
    ----------
    sdatas : list of SpatialData
    batch_names : list of str
        One label per SpatialData. Used as suffix for all element keys
        and added to adata.obs["batch"].
    table_name : str
        Key of the AnnData table in each SpatialData object.
    offset_coords : bool
        If True, shift obsm["spatial"] and uns["spatial"] image coords
        so batches appear side-by-side in scanpy/squidpy plots.
        SpatialData elements are never modified regardless.
    feature_join : "inner" | "outer"
        "inner" keeps only features common to ALL samples.
        "outer" keeps all features, filling missing pixels with 0.

    Returns
    -------
    SpatialData
    """
    import numpy as np
    import anndata as ad
    import geopandas as gpd
    import pandas as pd
    import re
    from scipy.sparse import issparse
    from spatialdata import SpatialData
    from spatialdata.models import Image2DModel, PointsModel, ShapesModel, TableModel
    from spatialdata.transformations import Identity


    offset_coords = True

    if len(sdatas) != len(batch_names):
        raise ValueError(
            f"sdatas ({len(sdatas)}) and batch_names ({len(batch_names)}) "
            f"must have the same length."
        )
    if feature_join not in ("inner", "outer"):
        raise ValueError(
            f"feature_join must be 'inner' or 'outer', got '{feature_join}'."
        )
    for b in batch_names:
        if not re.match(r'^[A-Za-z0-9_.\-]+$', b):
            raise ValueError(
                f"batch_name '{b}' contains invalid characters. "
                f"Use only alphanumeric characters, underscores, dots or hyphens."
            )

    def _make_key(base: str, batch: str) -> str:
        safe_base = re.sub(r'[^A-Za-z0-9_.\-]', '_', base)
        return f"{safe_base}_{batch}"

    def _strip_transforms(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        gdf = gdf.copy()
        gdf.attrs = {}
        return gdf.drop(
            columns=[c for c in gdf.columns if c.startswith("transform")],
            errors="ignore",
        )

    # ------------------------------------------------------------------
    # Report feature overlap
    # ------------------------------------------------------------------
    all_var_sets    = [set(sd.tables[table_name].var_names) for sd in sdatas]
    common_features = set.intersection(*all_var_sets)
    all_features    = set.union(*all_var_sets)

    print("Feature summary:")
    print(f"  Common to all batches   : {len(common_features):,}")
    print(f"  Total across all batches: {len(all_features):,}")
    for i, b in enumerate(batch_names):
        uniq = all_var_sets[i] - common_features
        print(f"  Unique to {b:20s}: {len(uniq):,}")
    print(
        f"  feature_join='{feature_join}' → keeping "
        f"{'common features only' if feature_join == 'inner' else 'all features (filling missing with 0)'}"
    )

    # ------------------------------------------------------------------
    # Determine the shared feature axis up front so we can slice each
    # batch to exactly this set (and order) BEFORE concat.
    # This prevents ad.concat from reordering columns, which would
    # decouple X rows from their spatial obs metadata.
    # ------------------------------------------------------------------
    if feature_join == "inner":
        shared_vars = sorted(common_features, key=lambda v: float(v) if _is_float(v) else v)
    else:
        shared_vars = sorted(all_features, key=lambda v: float(v) if _is_float(v) else v)

    # ------------------------------------------------------------------
    # Main loop — build spatialdata elements and adatas
    # ------------------------------------------------------------------
    all_images          = {}
    all_points          = {}
    all_shapes          = {}
    all_adatas          = []
    per_batch_categoricals = []  # stores {col: [categories]} per batch
    x_offset            = 0.0

    for sdata, batch in zip(sdatas, batch_names):

        # ----------------------------------------------------------------
        # Extract X and obs directly from the underlying AnnData so that
        # spatialdata's internal indexing/reordering cannot affect row order.
        # We reconstruct a clean AnnData from scratch to guarantee that
        # row i of X always corresponds to row i of obs.
        # ----------------------------------------------------------------
        src = sdata.tables[table_name]
        n_obs = src.n_obs

        # --- Dense X, row order preserved ---
        X_raw = src.X.toarray() if issparse(src.X) else np.array(src.X, dtype=np.float32)

        # Sanity check: X rows must match obs rows
        assert X_raw.shape[0] == n_obs, (
            f"[{batch}] X has {X_raw.shape[0]} rows but obs has {n_obs} rows — "
            "AnnData is internally inconsistent."
        )

        # Verify spatial obs columns are aligned with X before doing anything
        if "he_x" in src.obs.columns:
            he_x_vals = src.obs["he_x"].values
            # Spot-check: he_x should have the same length as X rows
            assert len(he_x_vals) == X_raw.shape[0], (
                f"[{batch}] he_x length {len(he_x_vals)} != X rows {X_raw.shape[0]}"
            )

        # --- Slice to shared feature set in the correct order ---
        src_var_index = list(src.var_names)
        if feature_join == "inner":
            col_idx = [src_var_index.index(v) for v in shared_vars if v in src_var_index]
            vars_this_batch = [shared_vars[i] for i, v in enumerate(shared_vars) if v in src_var_index]
        else:
            # outer: pad missing columns with zeros
            col_map = {v: i for i, v in enumerate(src_var_index)}
            X_padded = np.zeros((n_obs, len(shared_vars)), dtype=np.float32)
            for j, v in enumerate(shared_vars):
                if v in col_map:
                    X_padded[:, j] = X_raw[:, col_map[v]]
            X_raw = X_padded
            col_idx = list(range(len(shared_vars)))
            vars_this_batch = shared_vars

        if feature_join == "inner":
            X_slice = X_raw[:, col_idx]
        else:
            X_slice = X_raw  # already padded above

        # --- Build clean obs DataFrame (positional, no spatialdata linkage) ---
        obs_clean = src.obs.copy()

        # Drop spatialdata linkage columns — will be reassigned positionally after concat
        drop_cols = ["instance_id", "region"]
        obs_clean.drop(columns=drop_cols, errors="ignore", inplace=True)

        # Record which columns are categorical BEFORE converting, so we can
        # restore them after concat (spatialdata_plot needs real Categoricals)
        categorical_cols = {
            col: list(obs_clean[col].cat.categories)
            for col in obs_clean.select_dtypes("category").columns
        }

        # Convert categoricals to plain strings so ad.concat does not reorder
        # rows trying to align category levels across batches
        for col in list(categorical_cols.keys()):
            obs_clean[col] = obs_clean[col].astype(str)

        per_batch_categoricals.append(categorical_cols)

        # Assign batch label
        obs_clean["batch"] = batch

        # Use positional obs_names with batch prefix — guarantees uniqueness
        obs_names = [f"{batch}_{i}" for i in range(n_obs)]

        # --- Build the clean AnnData ---
        adata = ad.AnnData(
            X=X_slice.astype(np.float32),
            obs=obs_clean,
            var=pd.DataFrame(index=vars_this_batch if feature_join == "inner" else shared_vars),
        )
        adata.obs_names = obs_names

        # Copy obsm (spatial coordinates etc.) — positional, so safe to copy directly
        for key, arr in src.obsm.items():
            adata.obsm[key] = np.array(arr, dtype=np.float32)

        # Copy uns (scalefactors, images, transform metadata etc.)
        import copy
        adata.uns = copy.deepcopy(dict(src.uns))
        adata.uns.pop("spatialdata_attrs", None)

        # --- Final alignment assertion ---
        if "he_x" in adata.obs.columns:
            assert len(adata.obs["he_x"]) == adata.n_obs, (
                f"[{batch}] he_x length mismatch after AnnData reconstruction"
            )
            assert adata.n_obs == X_slice.shape[0], (
                f"[{batch}] obs/X row count mismatch after reconstruction"
            )
            print(f"  [{batch}] alignment OK: {n_obs:,} pixels, "
                  f"he_x[0]={adata.obs['he_x'].iloc[0]:.1f}, "
                  f"X[0,0]={float(X_slice[0,0]):.4f}")

        # --- Compute x_offset width from this batch's spatial coords ---
        if "he_x" in adata.obs.columns:
            batch_width = float(adata.obs["he_x"].max()) + 100.0
        else:
            batch_width = float(adata.obsm["spatial"][:, 0].max()) + 100.0

        # --- Apply spatial offset for side-by-side scanpy plots ---
        if offset_coords and x_offset != 0.0:
            if "spatial" in adata.obsm:
                adata.obsm["spatial"] = adata.obsm["spatial"].copy()
                adata.obsm["spatial"][:, 0] += x_offset
            if "spatial" in adata.uns:
                for lib_id, lib in adata.uns["spatial"].items():
                    lib["x_offset"] = float(x_offset)

        all_adatas.append(adata)
        x_offset += batch_width

        # ---- Images: stored as-is, keyed by batch, no coord changes ----
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

        # ---- Shapes: native coords, no geometry changes ----
        for key, gdf in sdata.shapes.items():
            new_key  = _make_key(key, batch)
            gdf_copy = _strip_transforms(gdf)
            all_shapes[new_key] = ShapesModel.parse(
                gdf_copy,
                transformations={"global": Identity()},
            )

        # ---- Points: native coords, no changes ----
        for key, pts in sdata.points.items():
            new_key = _make_key(key, batch)
            pts_df  = pts.compute() if hasattr(pts, "compute") else pts.copy()
            pts_df  = pts_df.reset_index(drop=True).drop(
                columns=[c for c in pts_df.columns if c.startswith("transform")],
                errors="ignore",
            )
            all_points[new_key] = PointsModel.parse(
                pts_df,
                transformations={"global": Identity()},
            )

    # ------------------------------------------------------------------
    # Concatenate AnnData
    # All adatas now have:
    #   - identical var order (shared_vars)
    #   - unique obs_names (batch_i prefix)
    #   - no spatialdata linkage columns
    #   - no categoricals that could trigger reordering
    # So ad.concat is safe to use with axis=0.
    # ------------------------------------------------------------------
    merged_adata = ad.concat(
        all_adatas,
        axis=0,
        join=feature_join,       # vars are pre-aligned so inner == exact here
        fill_value=0.0,
        index_unique=None,  # obs_names already unique
    )

    # ------------------------------------------------------------------
    # Post-concat alignment verification
    # ------------------------------------------------------------------
    offset = 0
    for batch, adata_orig in zip(batch_names, all_adatas):
        n = adata_orig.n_obs
        merged_sub_he_x = merged_adata.obs["he_x"].iloc[offset: offset + n].values if "he_x" in merged_adata.obs.columns else None
        orig_he_x       = adata_orig.obs["he_x"].values if "he_x" in adata_orig.obs.columns else None

        if merged_sub_he_x is not None and orig_he_x is not None:
            if not np.allclose(merged_sub_he_x, orig_he_x, atol=1e-3):
                raise RuntimeError(
                    f"FATAL: he_x mismatch after concat for batch '{batch}'. "
                    f"Row order was corrupted. First 5 merged={merged_sub_he_x[:5]}, "
                    f"orig={orig_he_x[:5]}"
                )
            print(f"  [{batch}] post-concat alignment verified ✓")
        offset += n

    # ------------------------------------------------------------------
    # Assign fresh instance_id and region linkage
    # ------------------------------------------------------------------
    merged_adata.obs["instance_id"] = np.arange(len(merged_adata)).astype(str)

    # Reconstruct region column from batch labels (was dropped earlier)
    region_values = []
    for batch, adata_orig in zip(batch_names, all_adatas):
        pixel_region_key = _make_key("pixels", batch)
        region_values.extend([pixel_region_key] * adata_orig.n_obs)

    all_pixel_regions = [_make_key("pixels", b) for b in batch_names]
    merged_adata.obs["region"] = pd.Categorical(
        region_values,
        categories=all_pixel_regions,
    )

    # Restore categorical columns that were temporarily stringified for concat.
    # spatialdata_plot requires these to be proper Categoricals to render correctly.
    # Collect all categorical column names across all batches.
    all_categorical_cols = {}
    for cat_dict in per_batch_categoricals:
        for col, cats in cat_dict.items():
            if col not in all_categorical_cols:
                all_categorical_cols[col] = set(cats)
            else:
                all_categorical_cols[col].update(cats)

    for col, cats in all_categorical_cols.items():
        if col in merged_adata.obs.columns:
            # Sort categories: numeric if possible, else alphabetic
            try:
                sorted_cats = sorted(cats, key=float)
            except (ValueError, TypeError):
                sorted_cats = sorted(cats)
            merged_adata.obs[col] = pd.Categorical(
                merged_adata.obs[col].astype(str),
                categories=[str(c) for c in sorted_cats],
            )

    # Fix shapes GDF indices to match instance_id so spatialdata_plot
    # can join the table to the shapes for rendering colour columns.
    # Each batch's pixels_<batch> shapes must be re-indexed to match the
    # instance_id values assigned to that batch's rows in merged_adata.
    offset = 0
    for batch, adata_orig in zip(batch_names, all_adatas):
        n = adata_orig.n_obs
        pixel_region_key = _make_key("pixels", batch)
        if pixel_region_key in all_shapes:
            batch_instance_ids = merged_adata.obs["instance_id"].iloc[offset: offset + n].values
            gdf = all_shapes[pixel_region_key].copy()
            # Drop any embedded transformation attrs before re-indexing
            # so ShapesModel.parse doesn't see duplicate transform specs
            gdf.attrs = {}
            gdf.index = pd.Index(batch_instance_ids)
            all_shapes[pixel_region_key] = ShapesModel.parse(
                gdf,
                transformations={"global": Identity()},
            )
        offset += n

    # ------------------------------------------------------------------
    # Build SpatialData and attach merged table
    # ------------------------------------------------------------------
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


def _is_float(v: str) -> bool:
    """Helper: return True if string v can be parsed as float."""
    try:
        float(v)
        return True
    except (ValueError, TypeError):
        return False