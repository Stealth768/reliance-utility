"""Unit tests for seed setting and manifest utilities."""

import hashlib
import json
import random
from pathlib import Path
import numpy as np

from scripts.hash_results import compute_file_sha256, hash_raw_results
from scripts.make_manifest import write_manifest
from src.seed import set_seed


def test_set_seed_deterministic():
    """set_seed produces reproducible random and numpy sequences."""
    set_seed(123)
    val_py1 = random.random()
    val_np1 = np.random.rand()

    set_seed(123)
    val_py2 = random.random()
    val_np2 = np.random.rand()

    assert val_py1 == val_py2
    assert val_np1 == val_np2


def test_write_manifest(tmp_path: Path):
    """write_manifest generates manifest.json with sha256 of sorted config, seed, model, timestamp."""
    run_dir = tmp_path / "run_01"
    config = {"batch_size": 16, "alpha": 0.05, "corruption": "gaussian_noise"}

    manifest = write_manifest(run_dir, config, seed=42, model_name="test-vlm")
    manifest_file = run_dir / "manifest.json"
    assert manifest_file.exists()

    with open(manifest_file, "r", encoding="utf-8") as f:
        loaded = json.load(f)

    sorted_cfg_str = json.dumps(config, sort_keys=True)
    expected_hash = hashlib.sha256(sorted_cfg_str.encode("utf-8")).hexdigest()

    assert loaded["config_sha256"] == expected_hash
    assert loaded["seed"] == 42
    assert loaded["model_name"] == "test-vlm"
    assert "timestamp_utc" in loaded
    assert "git_commit" in loaded


def test_hash_raw_results(tmp_path: Path):
    """hash_raw_results computes hashes for all files in directory."""
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()

    file1 = raw_dir / "item1.json"
    file1.write_text("content1", encoding="utf-8")

    out_manifest = hash_raw_results(raw_dir=raw_dir)
    assert out_manifest.exists()
    content = out_manifest.read_text(encoding="utf-8")
    assert "item1.json" in content
    assert compute_file_sha256(file1) in content
