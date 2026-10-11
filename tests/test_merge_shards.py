"""CPU tests for scripts/merge_shards.py.

All tests use temporary directories and JSON files — no torch, no GPU required.
"""

import json
import tempfile
from pathlib import Path

import pytest

# Import functions under test
import sys
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.merge_shards import _parse_shard_name, merge_shards


def _write_shard(directory: Path, model_slug: str, corruption: str, seed: int, shard_idx: int,
                 git_commit: str = "abc1234", data_ids_hash: str = "deadbeef") -> Path:
    """Write a minimal shard JSON and its companion manifest to directory."""
    filename = f"{model_slug}_{corruption}_seed{seed}_shard{shard_idx}.json"
    shard_data = {
        "model_name": model_slug,
        "corruption": corruption,
        "shard_idx": shard_idx,
        "seed": seed,
        "start_idx": shard_idx * 25,
        "end_idx": (shard_idx + 1) * 25,
        "n_questions": 25,
        "S_grid": [0.0, 0.5, 1.0],
        "items": [],
    }
    shard_path = directory / filename
    shard_path.write_text(json.dumps(shard_data), encoding="utf-8")

    # Companion manifest
    manifest_dir = directory / f"manifest_{shard_path.stem}"
    manifest_dir.mkdir(exist_ok=True)
    manifest_data = {
        "git_commit": git_commit,
        "seed": seed,
        "data_ids_hash": data_ids_hash,
        "model_name": model_slug,
    }
    (manifest_dir / "manifest.json").write_text(json.dumps(manifest_data), encoding="utf-8")
    return shard_path


class TestMergeNoOverwrite:
    """Copying a shard that already exists in the output should be skipped."""

    def test_merge_no_overwrite(self, tmp_path):
        src_dir = tmp_path / "src"
        src_dir.mkdir()
        out_dir = tmp_path / "out"
        out_dir.mkdir()

        _write_shard(src_dir, "Qwen2-VL-2B", "gaussian", seed=0, shard_idx=0)
        # Pre-populate the output with the same shard
        _write_shard(out_dir, "Qwen2-VL-2B", "gaussian", seed=0, shard_idx=0)

        # Record original mtime of the file in output
        dest_file = out_dir / "Qwen2-VL-2B_gaussian_seed0_shard0.json"
        original_stat = dest_file.stat()

        copied = merge_shards([str(src_dir)], str(out_dir))

        # No file should have been overwritten
        assert copied == 0, f"Expected 0 copied (no overwrite), got {copied}"
        # File in output is unchanged
        new_stat = dest_file.stat()
        assert new_stat.st_mtime == original_stat.st_mtime


class TestMergeManifestMismatch:
    """Two shards with same identity but different git_commit should refuse to merge."""

    def test_merge_manifest_mismatch(self, tmp_path, capsys):
        src1 = tmp_path / "src1"
        src1.mkdir()
        src2 = tmp_path / "src2"
        src2.mkdir()
        out_dir = tmp_path / "out"
        out_dir.mkdir()

        # Write shard to src1 with git_commit=abc
        _write_shard(src1, "Qwen2-VL-2B", "gaussian", seed=0, shard_idx=0, git_commit="abc1234")
        # Write same shard identity to src2 with different git_commit
        _write_shard(src2, "Qwen2-VL-2B", "gaussian", seed=0, shard_idx=0, git_commit="xyz9999")

        copied = merge_shards([str(src1), str(src2)], str(out_dir))

        captured = capsys.readouterr()
        # The conflicting shard from src2 should have been skipped with a warning
        assert "conflict" in captured.out.lower() or "WARN" in captured.out, (
            "Expected conflict warning in output"
        )
        # Only one shard (from src1) should have been copied — the conflicting one skipped
        assert copied == 1


class TestMergeCoverageReport:
    """Coverage report should correctly identify missing shards."""

    def test_merge_coverage_report(self, tmp_path, capsys):
        src_dir = tmp_path / "src"
        src_dir.mkdir()
        out_dir = tmp_path / "out"
        out_dir.mkdir()

        # Write shards 0 and 2, but NOT shard 1
        _write_shard(src_dir, "Qwen2-VL-2B", "gaussian", seed=0, shard_idx=0)
        _write_shard(src_dir, "Qwen2-VL-2B", "gaussian", seed=0, shard_idx=2)

        merge_shards([str(src_dir)], str(out_dir))
        captured = capsys.readouterr()

        # Coverage report should mention 2 shards and missing shard 1
        assert "Coverage" in captured.out or "coverage" in captured.out.lower()
        # Missing shard 1 should be flagged
        assert "MISSING" in captured.out or "missing" in captured.out.lower()


class TestMergeLegacyShards:
    """Legacy shards without seed must be refused without flag, accepted with flag."""

    def test_legacy_shard_refused_without_flag(self, tmp_path, capsys):
        src_dir = tmp_path / "src"
        src_dir.mkdir()
        out_dir = tmp_path / "out"
        out_dir.mkdir()

        # Write legacy shard without seed in name
        legacy_name = "Qwen2-VL-2B_gaussian_shard0.json"
        (src_dir / legacy_name).write_text(json.dumps({
            "model_name": "Qwen2-VL-2B",
            "corruption": "gaussian",
            "shard_idx": 0,
            "items": [],
        }), encoding="utf-8")

        copied = merge_shards([str(src_dir)], str(out_dir), allow_legacy=False)
        captured = capsys.readouterr()

        assert copied == 0
        assert not (out_dir / legacy_name).exists()
        assert "REFUSED" in captured.err or "REFUSED" in captured.out

    def test_legacy_shard_accepted_with_flag(self, tmp_path, capsys):
        src_dir = tmp_path / "src"
        src_dir.mkdir()
        out_dir = tmp_path / "out"
        out_dir.mkdir()

        legacy_name = "Qwen2-VL-2B_gaussian_shard0.json"
        (src_dir / legacy_name).write_text(json.dumps({
            "model_name": "Qwen2-VL-2B",
            "corruption": "gaussian",
            "shard_idx": 0,
            "items": [],
        }), encoding="utf-8")

        # Manifest missing fields
        m_dir = src_dir / "manifest_Qwen2-VL-2B_gaussian_shard0"
        m_dir.mkdir()
        (m_dir / "manifest.json").write_text(json.dumps({"seed": 0}), encoding="utf-8")

        copied = merge_shards([str(src_dir)], str(out_dir), allow_legacy=True)
        captured = capsys.readouterr()

        assert copied == 1
        assert (out_dir / legacy_name).exists()
        assert "WARN" in captured.out
