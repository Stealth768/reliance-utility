"""Unified VLM benchmark runner supporting multiple architectures and corruptions.

Architectures supported via adapters:
- Qwen/Qwen2-VL-2B-Instruct
- HuggingFaceTB/SmolVLM-Instruct
- OpenGVLab/InternVL2-2B

Corruptions supported:
- gaussian (gaussian_blend)
- patch (patch_mask)
- swapblend (swap_blend with fixed seeded derangement)

Modes:
- Normal run: processes shards and saves resumable shard JSON files.
- --check: evaluates 10 questions clean and text-only, prints letter mass,
  and refuses to continue if mean letter mass < 0.9.
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
from src.corruptions.image import black_image, gaussian_blend, patch_mask, swap_blend, swapped
from src.io_utils import write_raw
from src.vlm.adapters import get_adapter
from src.vlm.data import load_aokvqa_validation, sample_main_disjoint_ids


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run VLM reliance-utility benchmark on A-OKVQA")
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2-VL-2B-Instruct",
        help="HuggingFace model identifier (e.g. Qwen/Qwen2-VL-2B-Instruct, OpenGVLab/InternVL2-2B, HuggingFaceTB/SmolVLM-Instruct)",
    )
    parser.add_argument(
        "--data_ids",
        type=str,
        default="results/derived/main_ids.json",
        help="Path to question IDs JSON (default: results/derived/main_ids.json for N=500)",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=500,
        help="Number of questions to evaluate (default: 500)",
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
        default="gaussian,patch,swapblend",
        help="Comma-separated corruptions to evaluate (default: gaussian,patch,swapblend)",
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
        default="results/raw/vlm_main",
        help="Output directory for raw evaluation shards (e.g. /kaggle/working/results/raw/vlm_main)",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Run 10 questions clean validation check and verify mean letter mass >= 0.9",
    )
    parser.add_argument(
        "--assistant_prefix",
        type=str,
        default=None,
        choices=["answer_colon", "the_answer_is"],
        help=(
            "SmolVLM only: named assistant pre-fill variant to append before reading logits. "
            "'answer_colon' -> 'Answer:', 'the_answer_is' -> 'The answer is'. "
            "Ignored for Qwen2-VL and InternVL2."
        ),
    )
    return parser.parse_args()


def format_eta(seconds: float) -> str:
    """Format seconds into HH:MM:SS or MM:SS."""
    if math.isnan(seconds) or seconds < 0:
        return "--:--"
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def get_derangement(n: int, seed: int) -> np.ndarray:
    """Generate a fixed-point-free permutation (derangement) of size n."""
    if n <= 1:
        raise ValueError("Derangement requires n >= 2")
    rng = np.random.default_rng(seed)
    p = rng.permutation(n)
    while np.any(p == np.arange(n)):
        p = rng.permutation(n)
    return p


def run_check_mode(adapter: Any, examples: List[Dict[str, Any]]):
    """Run 10 questions clean and text-only, verify mean letter mass >= 0.9.

    For adapters that have a ``_build_prompt_text`` method (SmolVLM, LLaVA),
    also prints the top-5 next-token probabilities for the clean forward pass on
    the first item, to diagnose letter mass failures.
    """
    import torch

    check_items = examples[:10]
    print(f"\n==================================================")
    print(f"RUNNING MODEL CHECK ON {len(check_items)} ITEMS: {adapter.model_id}")
    if hasattr(adapter, "_prefix_str") and adapter._prefix_str:
        print(f"  Assistant prefix: '{adapter._prefix_str}'")
    print(f"==================================================")

    clean_correct = 0
    text_correct = 0
    letter_masses = []

    for i, ex in enumerate(check_items):
        q = ex["question"]
        c = ex["choices"]
        target = ex["correct_choice_idx"]
        img = ex["image"]

        # Clean image forward pass
        p_clean, mass_clean = adapter.letter_probs(img, q, c)
        letter_masses.append(mass_clean)
        if int(np.argmax(p_clean)) == target:
            clean_correct += 1

        # Text-only forward pass
        p_text, mass_text = adapter.letter_probs(None, q, c)
        letter_masses.append(mass_text)
        if int(np.argmax(p_text)) == target:
            text_correct += 1

        print(
            f"  Item {i+1:2d}: target={target} | "
            f"clean_pred={np.argmax(p_clean)} (p={p_clean[target]:.3f}, mass={mass_clean:.4f}) | "
            f"text_pred={np.argmax(p_text)} (p={p_text[target]:.3f}, mass={mass_text:.4f})"
        )

        # For adapters with _build_prompt_text (SmolVLM, LLaVA): print top-5 next tokens
        if i == 0 and hasattr(adapter, "_build_prompt_text"):
            print("\n  [DIAGNOSTIC] Top-5 next tokens after clean forward pass (item 1):")
            ex0 = check_items[0]
            prompt_text = adapter._build_prompt_text(ex0["question"], ex0["choices"], with_image=True)
            inputs = adapter.processor(
                text=prompt_text,
                images=[ex0["image"]],
                return_tensors="pt",
            )
            inputs = {k: v.to(adapter.model.device) for k, v in inputs.items()}
            with torch.no_grad():
                raw_logits = adapter.model(**inputs).logits[0, -1, :]
            full_probs = torch.softmax(raw_logits, dim=-1)
            topk = torch.topk(full_probs, 5)
            for rank, (prob_val, tok_id) in enumerate(
                zip(topk.values.tolist(), topk.indices.tolist()), start=1
            ):
                tok_str = repr(adapter.tokenizer.decode([tok_id]))
                print(f"    #{rank}: {tok_str:20s} | ID: {tok_id:6d} | Prob: {prob_val:.4f}")
            print()

    mean_mass = float(np.mean(letter_masses))
    clean_acc = clean_correct / len(check_items)
    text_acc = text_correct / len(check_items)

    print("-" * 50)
    print(f"Mean Letter Probability Mass : {mean_mass:.4f}")
    print(f"Clean Multimodal Accuracy    : {clean_acc:.2%}")
    print(f"Text-Only Accuracy          : {text_acc:.2%}")
    print("-" * 50)

    if mean_mass < 0.90:
        print(f"\n[FAIL] Mean letter probability mass ({mean_mass:.4f}) is below 0.90 threshold!")
        print("Refusing to continue. The model does not conform to the multiple choice format.")
        sys.exit(1)
    else:
        print(f"\n[PASS] Model check successful! Mean letter mass >= 0.90.")


def run_benchmark():
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load questions
    data_ids_path = Path(args.data_ids)
    if not data_ids_path.exists():
        if "main_ids" in str(data_ids_path):
            sample_main_disjoint_ids(main_ids_path=str(data_ids_path), n_sample=args.n, seed=args.seed + 1)
        else:
            raise FileNotFoundError(f"Requested dataset IDs file not found: {data_ids_path}")

    all_examples = load_aokvqa_validation(
        pilot_ids_path=str(data_ids_path),
        n_sample=args.n,
        seed=args.seed,
    )
    if len(all_examples) > args.n:
        all_examples = all_examples[: args.n]

    n_total = len(all_examples)
    corruptions_list = [c.strip() for c in args.corruptions.split(",") if c.strip()]
    S_grid = np.linspace(0.0, 1.0, args.grid).tolist()

    # 2. Model adapter initialization
    adapter = get_adapter(args.model, assistant_prefix=args.assistant_prefix)
    model_slug = args.model.split("/")[-1]

    # Pre-compute derangement permutation for swapblend corruption
    swap_permutation = get_derangement(n_total, seed=args.seed + 100)

    # 3. Check mode execution
    if args.check:
        adapter.load(device="cuda", dtype="float16")
        run_check_mode(adapter, all_examples)
        return

    # 4. Filter needed shard tasks
    n_shards = (n_total + args.shard_size - 1) // args.shard_size
    needed_tasks = []
    for corruption in corruptions_list:
        for shard_idx in range(n_shards):
            shard_file = out_dir / f"{model_slug}_{corruption}_shard{shard_idx}.json"
            if not shard_file.exists():
                needed_tasks.append((corruption, shard_idx, shard_file))

    if not needed_tasks:
        print(f"\nAll {len(corruptions_list) * n_shards} shards already exist in {out_dir}. Nothing to do!")
        return

    print(f"\nTarget Model: {args.model}")
    print(f"Total Questions: {n_total} across {n_shards} shards")
    print(f"Corruptions: {corruptions_list}")
    print(f"Pending tasks: {len(needed_tasks)} shards")

    # Load model onto GPU
    adapter.load(device="cuda", dtype="float16")

    start_time = time.time()
    completed_items = 0
    total_items_to_process = len(needed_tasks) * args.shard_size

    for corruption, shard_idx, shard_file in needed_tasks:
        start_idx = shard_idx * args.shard_size
        end_idx = min(start_idx + args.shard_size, n_total)
        shard_examples = all_examples[start_idx:end_idx]

        print(
            f"\n>>> Shard {shard_idx + 1}/{n_shards} "
            f"[{corruption}] (Questions {start_idx}..{end_idx - 1}) <<<"
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
            p_null_removed, mass_removed = adapter.letter_probs(None, question, choices)

            # 2. Null 'black'
            blk_img = black_image(orig_image.size)
            p_null_black, mass_black = adapter.letter_probs(blk_img, question, choices)

            # 3. Swapped-image control (derangement donor)
            donor_idx = int(swap_permutation[q_global_idx])
            donor_image = all_examples[donor_idx]["image"]
            swp_img = swapped(donor_image)
            p_swapped, mass_swapped = adapter.letter_probs(swp_img, question, choices)

            # 4. Probs across severity grid S
            probs_per_s = []
            mass_per_s = []

            for s_idx, S in enumerate(S_grid):
                item_seed = args.seed * 10000 + q_global_idx * 100 + int(round(S * 100))
                if corruption == "gaussian":
                    corrupted_img = gaussian_blend(orig_image, S=S, seed=item_seed)
                elif corruption == "patch":
                    corrupted_img = patch_mask(orig_image, S=S, seed=item_seed, patch=28)
                elif corruption == "swapblend":
                    corrupted_img = swap_blend(orig_image, donor_image, S=S)
                else:
                    raise ValueError(f"Unknown corruption kind: {corruption}")

                ps, mass = adapter.letter_probs(corrupted_img, question, choices)
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
                    f"Done {q_local_idx + 1}/{len(shard_examples)} | "
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

    print(f"\nAll benchmark tasks completed in {format_eta(time.time() - start_time)}.")


if __name__ == "__main__":
    run_benchmark()
