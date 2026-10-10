"""VLM reliance-utility pilot experiment runner (GPU execution, e.g. Kaggle T4).

Runs Qwen2-VL-2B-Instruct on sampled A-OKVQA validation questions across:
- Null modalities: 'removed' (text-only) and 'black' (all-zeros image)
- Corruptions: 'gaussian' (gaussian_blend) and 'patch' (patch_mask) across S grid
- Swapped control image

Saves resumable shard JSON files (25 questions per shard) via write_raw.
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

# Ensure repo root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from scripts.make_manifest import write_manifest
from src.corruptions.image import black_image, gaussian_blend, patch_mask, swapped
from src.io_utils import write_raw
from src.vlm.data import load_aokvqa_validation
from src.vlm.scoring import letter_probs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run VLM pilot benchmark on A-OKVQA")
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2-VL-2B-Instruct",
        help="HuggingFace model identifier",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=200,
        help="Number of questions to evaluate (default: 200)",
    )
    parser.add_argument(
        "--grid",
        type=int,
        default=9,
        help="Number of severity grid points in [0, 1] (default: 9 -> 0, 0.125, ..., 1.0)",
    )
    parser.add_argument(
        "--corruptions",
        type=str,
        default="gaussian,patch",
        help="Comma-separated corruptions to evaluate (default: gaussian,patch)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for data sampling and corruptions (default: 0)",
    )
    parser.add_argument(
        "--shard_size",
        type=int,
        default=25,
        help="Number of questions per output shard file (default: 25)",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="results/raw/vlm_pilot",
        help="Output directory for raw evaluation shards (e.g. /kaggle/working/results/raw/vlm_pilot)",
    )
    return parser.parse_args()


def load_model_and_processor(model_id: str):
    """Load Qwen2-VL model in fp16 on CUDA with memory-capped processor."""
    import torch
    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

    print(f"Loading model: {model_id} in float16 on CUDA...")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()

    # Configure processor with capped pixels so a T4 GPU does not OOM
    # 256*28*28 = 200,704 min pixels; 1024*28*28 = 802,816 max pixels
    min_pixels = 256 * 28 * 28
    max_pixels = 1024 * 28 * 28
    size_dict = {"shortest_edge": min_pixels, "longest_edge": max_pixels}

    print(f"Loading processor with size constraints: {size_dict}...")
    try:
        processor = AutoProcessor.from_pretrained(
            model_id,
            size=size_dict,
        )
    except TypeError:
        # Fallback if size argument is passed differently
        processor = AutoProcessor.from_pretrained(
            model_id,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )

    return model, processor


def format_eta(seconds: float) -> str:
    """Format seconds into HH:MM:SS or MM:SS."""
    if math.isnan(seconds) or seconds < 0:
        return "--:--"
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def run_pilot():
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    corruptions_list = [c.strip() for c in args.corruptions.split(",") if c.strip()]
    S_grid = np.linspace(0.0, 1.0, args.grid).tolist()

    # Load dataset pilot examples
    print(f"\n--- Loading A-OKVQA validation items (N={args.n}, seed={args.seed}) ---")
    pilot_examples = load_aokvqa_validation(
        pilot_ids_path="results/derived/pilot_ids.json",
        n_sample=args.n,
        seed=args.seed,
    )
    if len(pilot_examples) > args.n:
        pilot_examples = pilot_examples[: args.n]

    n_total = len(pilot_examples)
    n_shards = (n_total + args.shard_size - 1) // args.shard_size
    print(f"Total questions: {n_total} across {n_shards} shards of size {args.shard_size}")

    # Clean model identifier for filename
    model_slug = args.model.split("/")[-1]

    # Check which shards need processing
    needed_tasks = []
    for corruption in corruptions_list:
        for shard_idx in range(n_shards):
            shard_file = out_dir / f"{model_slug}_{corruption}_shard{shard_idx}.json"
            if not shard_file.exists():
                needed_tasks.append((corruption, shard_idx, shard_file))

    if not needed_tasks:
        print("\nAll shards already exist. Nothing to do!")
        return

    print(f"Pending tasks to execute: {len(needed_tasks)} shards")

    # Load model and processor only when work is needed
    model, processor = load_model_and_processor(args.model)

    start_time = time.time()
    completed_items = 0
    total_items_to_process = len(needed_tasks) * args.shard_size

    for corruption, shard_idx, shard_file in needed_tasks:
        start_idx = shard_idx * args.shard_size
        end_idx = min(start_idx + args.shard_size, n_total)
        shard_examples = pilot_examples[start_idx:end_idx]

        print(
            f"\n>>> Processing Shard {shard_idx + 1}/{n_shards} "
            f"[{corruption}] (Items {start_idx}..{end_idx - 1}) <<<"
        )

        items_records = []
        for q_local_idx, ex in enumerate(shard_examples):
            q_global_idx = start_idx + q_local_idx
            q_id = ex["question_id"]
            question = ex["question"]
            choices = ex["choices"]
            correct_idx = ex["correct_choice_idx"]
            orig_image = ex["image"]

            # 1. Null 'removed' (text-only)
            p_null_removed, mass_removed = letter_probs(
                model, processor, None, question, choices
            )

            # 2. Null 'black'
            blk_img = black_image(orig_image.size)
            p_null_black, mass_black = letter_probs(
                model, processor, blk_img, question, choices
            )

            # 3. Swapped-image control (pick next question's image cyclically)
            swapped_donor_idx = (q_global_idx + 1) % n_total
            donor_image = pilot_examples[swapped_donor_idx]["image"]
            swp_img = swapped(donor_image)
            p_swapped, mass_swapped = letter_probs(
                model, processor, swp_img, question, choices
            )

            # 4. Probs across severity grid S
            probs_per_s = []
            mass_per_s = []

            for s_idx, S in enumerate(S_grid):
                item_seed = args.seed * 10000 + q_global_idx * 100 + int(round(S * 100))
                if corruption == "gaussian":
                    corrupted_img = gaussian_blend(orig_image, S=S, seed=item_seed)
                elif corruption == "patch":
                    corrupted_img = patch_mask(orig_image, S=S, seed=item_seed, patch=28)
                else:
                    raise ValueError(f"Unknown corruption kind: {corruption}")

                ps, mass = letter_probs(
                    model, processor, corrupted_img, question, choices
                )
                probs_per_s.append(ps.tolist())
                mass_per_s.append(mass)

            item_dict = {
                "question_id": q_id,
                "correct_choice_idx": correct_idx,
                "null_probs": {
                    "removed": p_null_removed.tolist(),
                    "black": p_null_black.tolist(),
                },
                "null_mass": {
                    "removed": mass_removed,
                    "black": mass_black,
                },
                "swapped_probs": p_swapped.tolist(),
                "swapped_mass": mass_swapped,
                "probs_per_S": probs_per_s,
                "mass_per_S": mass_per_s,
            }
            items_records.append(item_dict)

            completed_items += 1
            elapsed = time.time() - start_time
            rate = completed_items / max(elapsed, 1e-4)
            remaining_sec = (total_items_to_process - completed_items) / max(rate, 1e-4)

            if (q_local_idx + 1) % 5 == 0 or (q_local_idx + 1) == len(shard_examples):
                print(
                    f"  [{corruption} shard {shard_idx}] "
                    f"Done {q_local_idx + 1}/{len(shard_examples)} items | "
                    f"Overall: {completed_items}/{total_items_to_process} | "
                    f"Rate: {rate:.2f} items/s | "
                    f"ETA: {format_eta(remaining_sec)}"
                )

        shard_data: Dict[str, Any] = {
            "model_name": args.model,
            "corruption": corruption,
            "shard_idx": shard_idx,
            "start_idx": start_idx,
            "end_idx": end_idx,
            "n_questions": len(items_records),
            "S_grid": S_grid,
            "items": items_records,
        }

        # Write atomically, skip if already exists
        if not shard_file.exists():
            write_raw(shard_file, shard_data)
            print(f"  -> Successfully saved shard: {shard_file.name}")

        # Write run manifest
        config_dict = {
            "model": args.model,
            "corruption": corruption,
            "shard_idx": shard_idx,
            "n_items": len(items_records),
            "grid_points": len(S_grid),
            "seed": args.seed,
        }
        manifest_dir = out_dir / f"manifest_{model_slug}_{corruption}_shard{shard_idx}"
        write_manifest(manifest_dir, config_dict, seed=args.seed, model_name=args.model)

    print(f"\nAll pilot runs completed in {format_eta(time.time() - start_time)}.")


if __name__ == "__main__":
    run_pilot()
