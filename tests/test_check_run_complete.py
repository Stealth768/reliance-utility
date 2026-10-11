"""CPU tests for scripts/check_run_complete.py.

All tests use temporary directories with minimal JSON shard files.
No torch, no GPU required.
"""

import json
import tempfile
from pathlib import Path

import pytest

import sys
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.check_run_complete import check_run_complete


def _write_shard(directory: Path, model_slug: str, corruption: str, seed: int, shard_idx: int) -> Path:
    """Write a minimal shard JSON to directory."""
    filename = f"{model_slug}_{corruption}_seed{seed}_shard{shard_idx}.json"
    shard_data = {"model_name": model_slug, "shard_idx": shard_idx, "items": []}
    shard_path = directory / filename
    shard_path.write_text(json.dumps(shard_data), encoding="utf-8")
    return shard_path


class TestCompleteRunExits0:
    """All shards present -> check_run_complete returns True (exits 0)."""

    def test_complete_run_exits_0(self, tmp_path):
        model = "Qwen/Qwen2-VL-2B-Instruct"
        model_slug = "Qwen2-VL-2B-Instruct"
        n = 50
        shard_size = 25
        n_shards = 2  # 50 / 25

        for idx in range(n_shards):
            _write_shard(tmp_path, model_slug, "gaussian", seed=0, shard_idx=idx)

        result = check_run_complete(
            out_dir=str(tmp_path),
            model=model,
            corruption="gaussian",
            seed=0,
            n=n,
            shard_size=shard_size,
        )
        assert result is True


class TestIncompleteRunExits1:
    """Missing shards -> check_run_complete returns False (exits 1)."""

    def test_incomplete_run_exits_1(self, tmp_path):
        model = "Qwen/Qwen2-VL-2B-Instruct"
        model_slug = "Qwen2-VL-2B-Instruct"
        n = 75
        shard_size = 25
        n_shards = 3  # ceil(75 / 25)

        # Write only shards 0 and 2, missing shard 1
        _write_shard(tmp_path, model_slug, "gaussian", seed=0, shard_idx=0)
        _write_shard(tmp_path, model_slug, "gaussian", seed=0, shard_idx=2)

        result = check_run_complete(
            out_dir=str(tmp_path),
            model=model,
            corruption="gaussian",
            seed=0,
            n=n,
            shard_size=shard_size,
        )
        assert result is False

    def test_missing_all_shards_exits_1(self, tmp_path):
        result = check_run_complete(
            out_dir=str(tmp_path),
            model="HuggingFaceTB/SmolVLM-Instruct",
            corruption="patch",
            seed=1,
            n=25,
            shard_size=25,
        )
        assert result is False


class TestLegacyShardsAsSeed0:
    """Legacy shards named without seed should be identified as seed 0."""

    def test_legacy_shards_recognized_as_seed_0(self, tmp_path):
        model = "Qwen/Qwen2-VL-2B-Instruct"
        model_slug = "Qwen2-VL-2B-Instruct"
        n = 50
        shard_size = 25

        # Write legacy shards without seed in filename
        for idx in range(2):
            fname = f"{model_slug}_gaussian_shard{idx}.json"
            (tmp_path / fname).write_text(json.dumps({
                "model_name": model_slug,
                "shard_idx": idx,
                "items": [],
            }), encoding="utf-8")

        result = check_run_complete(
            out_dir=str(tmp_path),
            model=model,
            corruption="gaussian",
            seed=0,
            n=n,
            shard_size=shard_size,
        )
        assert result is True
