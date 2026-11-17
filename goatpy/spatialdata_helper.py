
from pathlib import Path
from tempfile import mkdtemp
import os
import shutil
import uuid

import spatialdata as sd
import zarr


_ELEMENT_TYPES = ("images", "labels", "points", "shapes", "tables")


def save_element(sdata, element_name, *, verify=True):
    """
    Save one SpatialData element to its existing local Zarr store.

    Supports: images, labels, points, shapes, tables.

    - Stages and validates the replacement before touching the original.
    - Does not rewrite unrelated elements.
    - Uses a same-filesystem backup for rollback.
    - Returns a freshly loaded SpatialData object.

    Requirements:
        Local filesystem Zarr directory.
        No concurrent readers/writers during replacement.
        Enough free disk space for a temporary copy of the element.

    Not a fully atomic transaction or crash-proof filesystem operation.
    """

    if sdata.path is None:
        raise ValueError("SpatialData has no associated Zarr store.")

    root = Path(sdata.path).resolve()

    if not root.is_dir():
        raise ValueError(f"Expected local Zarr directory: {root}")

    matches = [
        kind for kind in _ELEMENT_TYPES
        if element_name in getattr(sdata, kind)
    ]

    if len(matches) != 1:
        raise ValueError(
            f"Expected one element named {element_name!r}; "
            f"found in {matches}"
        )

    kind = matches[0]
    element = getattr(sdata, kind)[element_name]

    if not element_name or element_name in (".", ".."):
        raise ValueError("Invalid element name.")

    if "/" in element_name or "\\" in element_name:
        raise ValueError("Element names cannot contain path separators.")

    target = root / kind / element_name

    # Exclusive cooperative lock, outside the Zarr hierarchy.
    lock = root.parent / f".{root.name}.goatpy.lock"

    try:
        lock.mkdir()
    except FileExistsError:
        raise RuntimeError(
            f"GOATpy save lock already exists: {lock}. "
            "Check for another writer before removing it."
        ) from None

    stage_dir = None
    backup = None
    installed = False
    original_moved = False

    try:
        # Stage on the same filesystem as the original store.
        stage_dir = Path(
            mkdtemp(
                prefix=f".{root.name}.stage-",
                dir=root.parent,
            )
        )

        staged_store = stage_dir / "staged.zarr"

        kwargs = {kind: {element_name: element}}
        staged_sdata = sd.SpatialData(**kwargs)

        # SpatialData handles the element's native serialization.
        staged_sdata.write(staged_store)

        staged_element = staged_store / kind / element_name

        if not staged_element.is_dir():
            raise RuntimeError(
                "Staging failed: element directory was not created."
            )

        # Validate the staged element before touching the original.
        if verify:
            check = sd.read_zarr(staged_store)

            if element_name not in getattr(check, kind):
                raise RuntimeError("Staged element failed validation.")

            saved = getattr(check, kind)[element_name]

            if kind == "tables":
                if saved.shape != element.shape:
                    raise RuntimeError("Staged table shape mismatch.")

                if not saved.obs_names.equals(element.obs_names):
                    raise RuntimeError("Staged observation IDs mismatch.")

                if not saved.var_names.equals(element.var_names):
                    raise RuntimeError("Staged variable IDs mismatch.")

                if set(saved.obs.columns) != set(element.obs.columns):
                    raise RuntimeError("Staged observation columns mismatch.")

            elif kind == "shapes":
                if len(saved) != len(element):
                    raise RuntimeError("Staged shapes count mismatch.")

                if list(saved.columns) != list(element.columns):
                    raise RuntimeError("Staged shapes columns mismatch.")

        # Keep the old element as a rollback backup.
        backup = root.parent / (
            f".{root.name}.{kind}.{element_name}."
            f"backup-{uuid.uuid4().hex}"
        )

        target.parent.mkdir(parents=True, exist_ok=True)

        if target.exists():
            os.replace(target, backup)
            original_moved = True

        # Install the fully written replacement.
        os.replace(staged_element, target)
        installed = True

        # Refresh consolidated metadata after changing the hierarchy.
        zarr.consolidate_metadata(str(root))

        # Check that the full store is still readable.
        result = sd.read_zarr(root)

        if element_name not in getattr(result, kind):
            raise RuntimeError("Installed element not found after reload.")

        # Success: the old element is no longer required.
        if original_moved:
            shutil.rmtree(backup)
            backup = None

        return result

    except Exception:
        # Roll back if installation or validation failed.
        try:
            if installed and target.exists():
                shutil.rmtree(target)

            if original_moved and backup is not None:
                os.replace(backup, target)
                backup = None

            # Restore consolidated metadata after rollback.
            if installed or original_moved:
                zarr.consolidate_metadata(str(root))

        except Exception as rollback_error:
            raise RuntimeError(
                "Save failed and automatic rollback was unsuccessful. "
                f"Inspect the store and backup at {backup}. "
                "Do not delete recovery files."
            ) from rollback_error

        raise

    finally:
        if stage_dir is not None and stage_dir.exists():
            shutil.rmtree(stage_dir)

        # Preserve an unrecovered backup.
        if backup is None or not backup.exists():
            lock.rmdir()
