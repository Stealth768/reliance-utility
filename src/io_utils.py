"""Input/Output utilities enforcing write-once raw storage and atomic writes."""

import json
import os
import uuid
from pathlib import Path
from typing import Any, Union


def write_raw(path: Union[str, Path], data: Any) -> Path:
    """Write data to disk atomically, strictly refusing to overwrite existing files.

    Raw evaluation outputs are designed to be immutable and write-once. If a file
    at `path` already exists, a `FileExistsError` is raised.

    Writing is atomic: data is first written to a temporary file in the target directory,
    synced to persistent storage, and then renamed to the destination path.

    Parameters
    ----------
    path : Union[str, Path]
        Destination file path.
    data : Any
        Data to write. Supports bytes/bytearray, strings, and JSON-serializable structures (dict, list).

    Returns
    -------
    Path
        Path to the successfully written file.

    Raises
    ------
    FileExistsError
        If the destination file already exists.
    TypeError
        If the data type cannot be serialized.
    """
    dest_path = Path(path).resolve()

    if dest_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing file '{dest_path}'. Raw results are write-once by design."
        )

    dest_path.parent.mkdir(parents=True, exist_ok=True)

    # Use a unique temporary file in the same parent directory to ensure same-filesystem atomic rename
    temp_path = dest_path.with_name(f".{dest_path.name}.tmp.{uuid.uuid4().hex}")

    try:
        if isinstance(data, (bytes, bytearray)):
            with open(temp_path, "wb") as f:
                f.write(data)
                f.flush()
                os.fsync(f.fileno())
        elif isinstance(data, str):
            with open(temp_path, "w", encoding="utf-8") as f:
                f.write(data)
                f.flush()
                os.fsync(f.fileno())
        elif isinstance(data, (dict, list, int, float, bool)) or data is None:
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
                f.flush()
                os.fsync(f.fileno())
        else:
            raise TypeError(
                f"Unsupported data type '{type(data).__name__}'. "
                "Expected bytes, str, or JSON-serializable structure (dict, list, primitive)."
            )

        # Guard against concurrent creation between check and rename
        if dest_path.exists():
            raise FileExistsError(
                f"Refusing to overwrite existing file '{dest_path}'. Raw results are write-once by design."
            )

        # Atomic replacement
        os.replace(temp_path, dest_path)
    except Exception:
        # Clean up temporary file if write or rename failed
        if temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                pass
        raise

    return dest_path
