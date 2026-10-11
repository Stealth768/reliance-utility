"""Check whether all expected shards for a run are present in the output directory.

Usage
-----
    python scripts/check_run_complete.py \
        --dir results/raw/vlm_main \
        --model Qwen/Qwen2-VL-2B-Instruct \
        --corruption gaussian \
        --seed 0 \
        --n 500 \
        --shard_size 25

Exits with code 0 if all shards are present, code 1 if any are missing.
Prints the exact rerun command for any missing shard.
CPU-only. No torch imports.
"""

import argparse
import math
import sys
from pathlib import Path


def check_run_complete(
    out_dir: str,
    model: str,
    corruption: str,
    seed: int,
    n: int,
    shard_size: int,
) -> bool:
    """Check that all expected shards exist. Returns True if complete, False otherwise."""
    out_path = Path(out_dir)
    model_slug = model.split("/")[-1]
    n_shards = math.ceil(n / shard_size)

    present = []
    missing = []

    for shard_idx in range(n_shards):
        shard_file = out_path / f"{model_slug}_{corruption}_seed{seed}_shard{shard_idx}.json"
        if not shard_file.exists() and seed == 0:
            legacy_file = out_path / f"{model_slug}_{corruption}_shard{shard_idx}.json"
            if legacy_file.exists():
                shard_file = legacy_file

        if shard_file.exists():
            present.append(shard_idx)
        else:
            missing.append(shard_idx)

    print(f"Run: model={model_slug}, corruption={corruption}, seed={seed}, n={n}, shard_size={shard_size}")
    print(f"Expected {n_shards} shards | Present: {len(present)} | Missing: {len(missing)}")

    if present:
        print(f"  Present shards: {present}")

    if missing:
        print(f"  Missing shards: {missing}")
        for m_idx in missing:
            start_idx = m_idx * shard_size
            end_idx = min(start_idx + shard_size, n)
            print(
                f"\n  [RERUN shard {m_idx} (items {start_idx}..{end_idx - 1})]:"
            )
            print(
                f"    python scripts/run_vlm_pilot.py \\\n"
                f"        --model {model} \\\n"
                f"        --corruption {corruption} \\\n"
                f"        --seed {seed} \\\n"
                f"        --n {n} \\\n"
                f"        --shard_size {shard_size} \\\n"
                f"        --out {out_dir}"
            )
        return False

    print("  [OK] All shards present.")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Check completeness of a VLM benchmark run.")
    parser.add_argument("--dir", required=True, help="Output directory containing shard files.")
    parser.add_argument("--model", required=True, help="HuggingFace model identifier.")
    parser.add_argument("--corruption", required=True, help="Corruption name (gaussian, patch, swapblend).")
    parser.add_argument("--seed", type=int, required=True, help="Random seed used for the run.")
    parser.add_argument("--n", type=int, default=500, help="Total number of questions.")
    parser.add_argument("--shard_size", type=int, default=25, help="Questions per shard.")
    args = parser.parse_args()

    complete = check_run_complete(
        out_dir=args.dir,
        model=args.model,
        corruption=args.corruption,
        seed=args.seed,
        n=args.n,
        shard_size=args.shard_size,
    )
    return 0 if complete else 1


if __name__ == "__main__":
    sys.exit(main())
