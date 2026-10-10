"""Compute and record SHA-256 hashes of all raw result files."""

import hashlib
import os
from pathlib import Path
from typing import Optional, Union


def compute_file_sha256(filepath: Union[str, Path]) -> str:
    """Compute SHA-256 hash of a single file in binary chunks."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def hash_raw_results(
    raw_dir: Optional[Union[str, Path]] = None,
    output_path: Optional[Union[str, Path]] = None,
) -> Path:
    """Walk raw results directory and write MANIFEST.sha256.

    Parameters
    ----------
    raw_dir : Optional[Union[str, Path]]
        Directory to scan, defaults to 'results/raw'.
    output_path : Optional[Union[str, Path]]
        File path for manifest output, defaults to 'results/raw/MANIFEST.sha256'.

    Returns
    -------
    Path
        Path to the written MANIFEST.sha256 file.
    """
    if raw_dir is None:
        # Resolve relative to repo root (parent of scripts/)
        repo_root = Path(__file__).resolve().parent.parent
        raw_path = repo_root / "results" / "raw"
    else:
        raw_path = Path(raw_dir).resolve()

    if output_path is None:
        manifest_out = raw_path / "MANIFEST.sha256"
    else:
        manifest_out = Path(output_path).resolve()

    entries = []
    if raw_path.exists():
        for root, _, files in os.walk(raw_path):
            for filename in files:
                filepath = Path(root) / filename
                # Skip the manifest itself if placed within the same directory
                if filepath == manifest_out or filename == "MANIFEST.sha256":
                    continue
                file_hash = compute_file_sha256(filepath)
                rel_path = filepath.relative_to(raw_path).as_posix()
                entries.append((rel_path, file_hash))

    # Sort deterministically by relative path
    entries.sort(key=lambda item: item[0])

    manifest_out.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_out, "w", encoding="utf-8") as f:
        for rel_path, file_hash in entries:
            f.write(f"{file_hash}  {rel_path}\n")

    return manifest_out


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Hash files in results/raw/ to MANIFEST.sha256")
    parser.add_argument("--raw-dir", default=None, help="Directory containing raw results")
    parser.add_argument("--output", default=None, help="Output path for MANIFEST.sha256")

    args = parser.parse_args()
    out = hash_raw_results(args.raw_dir, args.output)
    print(f"Computed SHA-256 manifest written to: {out}")
