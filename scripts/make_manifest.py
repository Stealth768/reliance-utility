"""Manifest generation utility for experiment runs."""

import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Union


def get_git_commit_hash(cwd: Optional[Union[str, Path]] = None) -> str:
    """Retrieve current git commit hash, handling non-git environments gracefully."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
            cwd=cwd,
        )
        if result.returncode == 0:
            commit_hash = result.stdout.strip()
            if commit_hash:
                return commit_hash
        return "not_a_git_repo"
    except (FileNotFoundError, PermissionError, OSError):
        return "not_a_git_repo"


def write_manifest(
    run_dir: Union[str, Path],
    config_dict: Dict[str, Any],
    seed: int,
    model_name: str,
) -> Dict[str, Any]:
    """Write run manifest containing git hash, config SHA-256, seed, model, and timestamp.

    Parameters
    ----------
    run_dir : Union[str, Path]
        Directory where manifest.json will be saved.
    config_dict : Dict[str, Any]
        Configuration dictionary for the run.
    seed : int
        Random seed used for the run.
    model_name : str
        Name or identifier of the evaluated model.

    Returns
    -------
    Dict[str, Any]
        The manifest dictionary written to disk.
    """
    run_path = Path(run_dir)
    run_path.mkdir(parents=True, exist_ok=True)

    sorted_config_str = json.dumps(config_dict, sort_keys=True)
    config_sha256 = hashlib.sha256(sorted_config_str.encode("utf-8")).hexdigest()

    manifest_data = {
        "git_commit": get_git_commit_hash(cwd=run_path),
        "config_sha256": config_sha256,
        "seed": seed,
        "model_name": model_name,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }

    manifest_file = run_path / "manifest.json"
    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)

    return manifest_data


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate experiment manifest.json")
    parser.add_argument("--run-dir", required=True, help="Directory to save manifest.json")
    parser.add_argument("--config", required=True, help="Path to JSON config file")
    parser.add_argument("--seed", type=int, default=42, help="Seed used for run")
    parser.add_argument("--model-name", required=True, help="Model identifier")

    args = parser.parse_args()
    with open(args.config, "r", encoding="utf-8") as cfg_file:
        cfg = json.load(cfg_file)

    manifest = write_manifest(args.run_dir, cfg, args.seed, args.model_name)
    print(f"Manifest written to {Path(args.run_dir) / 'manifest.json'}")
