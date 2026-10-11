"""Merge shard JSON files from multiple source directories or zip archives.

Usage
-----
    python scripts/merge_shards.py --src <dir_or_zip> [--src ...] --out results/raw/vlm_main

Behavior
--------
- Accepts multiple --src (directories or .zip files); unzips zips to a temp dir.
- Copies shard JSON files that do not already exist in --out. Refuses to overwrite.
- Validates manifest consistency: two shards with the same logical identity
  (model, corruption, seed, shard_idx) must agree on git_commit, seed, data_ids_hash.
  Disagreements print a warning and skip the conflicting shard.
- Prints a coverage report: for each (model, corruption, seed), which shard indices
  are present vs missing, and the exact rerun command.
- After merge, calls scripts/hash_results.py on --out (as subprocess).
- CPU-only. No torch imports.
"""

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Pattern for new-format shard filenames: {model_slug}_{corruption}_seed{seed}_shard{idx}.json
_SHARD_PATTERN = re.compile(
    r"^(?P<model_slug>.+)_(?P<corruption>[^_]+)_seed(?P<seed>\d+)_shard(?P<shard_idx>\d+)\.json$"
)

# Pattern for legacy shard filenames without seed: {model_slug}_{corruption}_shard{idx}.json
_LEGACY_SHARD_PATTERN = re.compile(
    r"^(?P<model_slug>.+)_(?P<corruption>[^_]+)_shard(?P<shard_idx>\d+)\.json$"
)


def _parse_shard_name(filename: str) -> Optional[Dict[str, Any]]:
    """Parse a shard filename. Returns dict with metadata or None if invalid."""
    m = _SHARD_PATTERN.match(filename)
    if m is not None:
        return {
            "model_slug": m.group("model_slug"),
            "corruption": m.group("corruption"),
            "seed": m.group("seed"),
            "shard_idx": m.group("shard_idx"),
            "is_legacy": False,
        }
    m_leg = _LEGACY_SHARD_PATTERN.match(filename)
    if m_leg is not None:
        return {
            "model_slug": m_leg.group("model_slug"),
            "corruption": m_leg.group("corruption"),
            "seed": "0",
            "shard_idx": m_leg.group("shard_idx"),
            "is_legacy": True,
        }
    return None


def _load_manifest(shard_path: Path) -> Dict:
    """Try to load manifest.json from the sibling manifest directory for a shard file."""
    # Shard: {model}_{corruption}_seed{seed}_shard{idx}.json
    # Manifest dir: manifest_{model}_{corruption}_seed{seed}_shard{idx}/
    stem = shard_path.stem  # e.g. "Qwen2-VL-2B_gaussian_seed0_shard0"
    manifest_dir = shard_path.parent / f"manifest_{stem}"
    manifest_file = manifest_dir / "manifest.json"
    if manifest_file.exists():
        try:
            with open(manifest_file, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _collect_shards(src_paths: List[Path]) -> Dict[str, Path]:
    """Collect all shard JSON files from a list of directories, keyed by filename."""
    collected: Dict[str, Path] = {}
    for src in src_paths:
        for p in src.rglob("*.json"):
            info = _parse_shard_name(p.name)
            if info is not None:
                # If same filename seen from multiple sources, keep first
                if p.name not in collected:
                    collected[p.name] = p
    return collected


def _manifest_identity_key(info: Dict[str, str]) -> Tuple[str, str, str, str]:
    return (info["model_slug"], info["corruption"], info["seed"], info["shard_idx"])


def merge_shards(src_list: List[str], out_dir: str, allow_legacy: bool = False) -> int:
    """Merge shards from src_list into out_dir. Returns number of files copied."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    temp_dirs: List[tempfile.TemporaryDirectory] = []
    resolved_src_paths: List[Path] = []

    for src in src_list:
        p = Path(src)
        if p.suffix.lower() == ".zip":
            tmp = tempfile.TemporaryDirectory()
            temp_dirs.append(tmp)
            print(f"  Extracting zip: {p} -> {tmp.name}")
            with zipfile.ZipFile(p, "r") as zf:
                zf.extractall(tmp.name)
            resolved_src_paths.append(Path(tmp.name))
        elif p.is_dir():
            resolved_src_paths.append(p)
        else:
            print(f"  [WARN] Source not found or not a dir/zip: {p}", file=sys.stderr)

    try:
        src_shards = _collect_shards(resolved_src_paths)
        copied = 0
        skipped_overwrite = 0
        skipped_conflict = 0
        skipped_legacy = 0

        # Track manifest data per shard identity for conflict detection
        identity_manifests: Dict[Tuple, Dict] = {}

        for filename, src_file in sorted(src_shards.items()):
            info = _parse_shard_name(filename)
            if info is None:
                continue

            manifest = _load_manifest(src_file)

            # Check legacy format
            if info.get("is_legacy", False):
                if not allow_legacy:
                    print(
                        f"  [REFUSED] Legacy shard format without seed detected: {filename}. "
                        "Refusing to merge legacy shards without --allow_legacy flag.",
                        file=sys.stderr,
                    )
                    skipped_legacy += 1
                    continue
                else:
                    for field in ("git_commit", "seed", "data_ids_hash", "lib_versions", "gpu_name"):
                        if not manifest.get(field):
                            print(f"  [WARN] Legacy shard {filename}: manifest missing field '{field}'")

            key = _manifest_identity_key(info)

            # Conflict check: same identity but different provenance
            if key in identity_manifests:
                existing_m = identity_manifests[key]
                conflicts = []
                for field in ("git_commit", "seed", "data_ids_hash"):
                    ev = existing_m.get(field)
                    nv = manifest.get(field)
                    if ev is not None and nv is not None and ev != nv and ev != "unknown" and nv != "unknown":
                        conflicts.append(f"{field}: {ev!r} vs {nv!r}")
                if conflicts:
                    print(f"  [WARN] Manifest conflict for {filename}: {'; '.join(conflicts)} — SKIPPING")
                    skipped_conflict += 1
                    continue
            else:
                identity_manifests[key] = manifest

            dest_file = out_path / filename
            if dest_file.exists():
                skipped_overwrite += 1
                continue

            shutil.copy2(src_file, dest_file)
            # Also copy manifest directory if present
            stem = src_file.stem
            src_manifest_dir = src_file.parent / f"manifest_{stem}"
            if src_manifest_dir.is_dir():
                dest_manifest_dir = out_path / f"manifest_{stem}"
                if not dest_manifest_dir.exists():
                    shutil.copytree(src_manifest_dir, dest_manifest_dir)
            copied += 1

        print(f"\nMerge complete: {copied} copied, {skipped_overwrite} already existed, "
              f"{skipped_conflict} skipped (manifest conflict), {skipped_legacy} refused (legacy format)")

        # Coverage report
        _print_coverage_report(out_path)

        # Run hash_results.py
        hash_script = Path(__file__).parent / "hash_results.py"
        if hash_script.exists():
            print(f"\nRunning hash_results.py on {out_path}...")
            subprocess.run([sys.executable, str(hash_script), "--dir", str(out_path)], check=False)

        return copied
    finally:
        for tmp in temp_dirs:
            tmp.cleanup()


def _print_coverage_report(out_path: Path) -> None:
    """Print coverage: for each (model, corruption, seed), which shards are present vs missing."""
    all_present: Dict[Tuple[str, str, str], List[int]] = {}
    for p in sorted(out_path.glob("*.json")):
        info = _parse_shard_name(p.name)
        if info is None:
            continue
        group = (info["model_slug"], info["corruption"], info["seed"])
        all_present.setdefault(group, []).append(int(info["shard_idx"]))

    if not all_present:
        print("\n[Coverage] No new-format shards found in output directory.")
        return

    print("\n=== Coverage Report ===")
    for (model_slug, corruption, seed), present_indices in sorted(all_present.items()):
        present_set = set(present_indices)
        max_idx = max(present_indices)
        expected_set = set(range(max_idx + 1))
        missing = sorted(expected_set - present_set)
        status = "COMPLETE" if not missing else f"MISSING {len(missing)} shards"
        print(f"  {model_slug} / {corruption} / seed={seed}: {len(present_set)} shards | {status}")
        for m_idx in missing:
            # Attempt to reconstruct rerun command
            print(
                f"    [RERUN] python scripts/run_vlm_pilot.py "
                f"--model <hf_id_for_{model_slug}> "
                f"--corruption {corruption} --seed {seed} "
                f"--shard_size 25 --out <out_dir>  # shard {m_idx}"
            )
    print("=== End Coverage Report ===\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Merge VLM benchmark shard files from multiple sources.")
    parser.add_argument(
        "--src",
        action="append",
        required=True,
        dest="src_list",
        help="Source directory or .zip file. Can be specified multiple times.",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Output directory to merge shards into.",
    )
    parser.add_argument(
        "--allow_legacy",
        action="store_true",
        help="Allow merging legacy shard filenames without seed and warn about missing manifest fields.",
    )
    args = parser.parse_args()
    merge_shards(args.src_list, args.out, allow_legacy=args.allow_legacy)
    return 0


if __name__ == "__main__":
    sys.exit(main())
