"""Open-ended VLM reliance-utility benchmark runner on A-OKVQA.

Evaluates free-form generation without multiple-choice options:
- Generation prompt: "Answer the question with a short phrase."
- Decoding: greedy, max_new_tokens=8.
- Reference answer A*: generated greedy output under clean image (S=0).
- Utility: VQA soft accuracy min(matches / 3.0, 1.0) against direct_answers after canonical normalization.
- Reliance: Teacher-forced token-level JSD across answer token positions against null baseline.
  Normalized reliance: R_hat(S) = R(S) / R(0).
- Secondary: answer-change rate against A*.

Modes:
- Normal run: processes shards and saves resumable write-once shard JSON files.
- --check: evaluates 10 questions clean and null, prints generated answers, mean length,
  fraction empty, sanity check R=0 between identical conditions, time per question,
  runtime estimate, and exits non-zero if >10% of generated answers are empty.
"""

import argparse
import hashlib
import json
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure repo root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from scripts.make_manifest import write_manifest
from src.corruptions.image import black_image, gaussian_blend, patch_mask, swap_blend, swapped
from src.io_utils import write_raw
from src.metrics.open_ended import (
    compute_sequence_reliance,
    is_answer_changed,
    normalize_answer,
    sequence_jsd,
    vqa_soft_accuracy,
)
from src.vlm.adapters import get_adapter
from src.vlm.data import load_aokvqa_validation, sample_main_disjoint_ids


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run open-ended VLM reliance-utility benchmark on A-OKVQA")
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2-VL-2B-Instruct",
        help="HuggingFace model identifier (e.g. Qwen/Qwen2-VL-2B-Instruct, HuggingFaceTB/SmolVLM-Instruct, llava-hf/llava-1.5-7b-hf, Qwen/Qwen2-VL-7B-Instruct)",
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
        "--pairing_seed",
        type=int,
        default=0,
        help="Random seed for swapped donor derangement (default: 0 -> seed+100)",
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
        default="results/raw/vlm_open",
        help="Output directory for raw evaluation shards (e.g. results/raw/vlm_open)",
    )
    parser.add_argument(
        "--prompt_suffix",
        type=str,
        default="Answer the question with a short phrase.",
        help="Open-ended instruction appended to question (default: 'Answer the question with a short phrase.')",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=8,
        help="Maximum generated tokens for open-ended answer (default: 8)",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Run 10 questions validation check and verify fraction empty <= 10 percent",
    )
    parser.add_argument(
        "--assistant_prefix",
        type=str,
        default=None,
        choices=["answer_colon", "the_answer_is"],
        help="SmolVLM / LLaVA named assistant pre-fill variant.",
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


def get_git_info() -> Dict[str, Optional[str]]:
    info: Dict[str, Optional[str]] = {"git_commit": None, "git_tag": None}
    try:
        res = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=str(REPO_ROOT))
        if res.returncode == 0:
            info["git_commit"] = res.stdout.strip() or None
    except Exception:
        pass
    try:
        res = subprocess.run(["git", "describe", "--tags", "--exact-match", "HEAD"], capture_output=True, text=True, cwd=str(REPO_ROOT))
        if res.returncode == 0:
            info["git_tag"] = res.stdout.strip() or None
    except Exception:
        pass
    return info


def get_lib_versions() -> Dict[str, str]:
    versions: Dict[str, str] = {}
    for lib in ("torch", "transformers", "bitsandbytes"):
        try:
            mod = __import__(lib)
            versions[lib] = getattr(mod, "__version__", "unknown")
        except ImportError:
            versions[lib] = "n/a"
    return versions


def get_gpu_name() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except Exception:
        pass
    return "n/a"


def run_check_mode(adapter: Any, examples: List[Dict[str, Any]], prompt_suffix: str, max_new_tokens: int):
    """Run 10 questions open-ended check mode.

    # VERIFY-ON-GPU
    Evaluates:
    - Generated answers and lengths.
    - Fraction empty.
    - Sanity check R=0 between identical conditions.
    - Time per question and estimated total runtime.
    - Exits non-zero if fraction empty > 10%.
    """
    check_items = examples[:10]
    print(f"\n==================================================")
    print(f"RUNNING OPEN-ENDED MODEL CHECK ON {len(check_items)} ITEMS: {adapter.model_id}")
    print(f"Prompt instruction: '{prompt_suffix}' (greedy, max {max_new_tokens} tokens)")
    print(f"==================================================")

    generated_answers = []
    answer_word_lengths = []
    identical_rs = []
    start_time = time.time()

    for i, ex in enumerate(check_items):
        t0 = time.time()
        q = ex["question"]
        img = ex["image"]
        direct_answers = ex.get("direct_answers", [])

        # 1. Generate clean answer A*
        ans_clean = adapter.generate_open_answer(
            img, q, prompt_suffix=prompt_suffix, max_new_tokens=max_new_tokens
        )
        generated_answers.append(ans_clean)
        norm_ans = normalize_answer(ans_clean)
        words = norm_ans.split()
        answer_word_lengths.append(len(words))

        # Utility calculation against human direct_answers
        u_clean = vqa_soft_accuracy(ans_clean, direct_answers)

        # 2. Teacher-forced distribution sanity check (clean vs clean must yield R=0)
        p_clean_1 = adapter.teacher_forced_answer_probs(img, q, ans_clean, prompt_suffix=prompt_suffix)
        p_clean_2 = adapter.teacher_forced_answer_probs(img, q, ans_clean, prompt_suffix=prompt_suffix)
        m_jsd, _ = sequence_jsd(p_clean_1, p_clean_2)
        identical_rs.append(m_jsd)

        dt = time.time() - t0
        print(
            f"  Item {i+1:2d} ({dt:.2f}s): "
            f"A* = {repr(ans_clean):25s} | "
            f"Normalized = {repr(norm_ans):20s} | "
            f"VQA Acc = {u_clean:.2f} | "
            f"Sanity R(clean, clean) = {m_jsd:.6f}"
        )

    elapsed = time.time() - start_time
    time_per_q = elapsed / len(check_items)
    n_empty = sum(1 for a in generated_answers if not normalize_answer(a))
    frac_empty = n_empty / len(check_items)
    mean_words = float(np.mean(answer_word_lengths))
    max_identical_r = float(np.max(identical_rs))

    # Total runtime estimate: 500 questions x (3 corruptions x 9 grid + 2 nulls) forward passes
    # Approx 30 model passes per question
    est_total_hours = (time_per_q * 30 * 500) / 3600.0

    print("-" * 50)
    print(f"Check items evaluated        : {len(check_items)}")
    print(f"Mean answer length (words)   : {mean_words:.2f}")
    print(f"Empty answers                : {n_empty}/{len(check_items)} ({frac_empty:.1%})")
    print(f"Max identical condition R    : {max_identical_r:.8f} (Sanity check: must be < 1e-5)")
    print(f"Time per question (check)    : {time_per_q:.2f} s")
    print(f"Estimated runtime (500 items): {est_total_hours:.1f} hours")
    print("-" * 50)

    # Sanity check R == 0
    if max_identical_r > 1e-4:
        print(f"\n[FAIL] Sanity check failed! Identical conditions produced non-zero reliance: {max_identical_r:.6f}")
        sys.exit(1)

    # Gate: fraction empty <= 10%
    if frac_empty > 0.10:
        print(f"\n[FAIL] Fraction of empty answers ({frac_empty:.1%}) exceeds 10% threshold!")
        print("Model generated degenerate empty outputs under the open-ended prompt.")
        sys.exit(1)

    print(f"\n[PASS] Open-ended model check successful! Fraction empty <= 10%, R sanity passed.")


def run_open_benchmark():
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    git_info = get_git_info()
    lib_versions = get_lib_versions()
    gpu_name = get_gpu_name()

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

    try:
        data_ids_hash = hashlib.sha256(data_ids_path.read_bytes()).hexdigest()
    except Exception:
        data_ids_hash = "unknown"

    # 2. Model adapter initialization
    adapter = get_adapter(args.model, assistant_prefix=args.assistant_prefix)
    model_slug = args.model.split("/")[-1]

    # Pre-compute derangement permutation for swapped corruption
    swap_seed = args.pairing_seed if args.pairing_seed != 0 else (args.seed + 100)
    swap_permutation = get_derangement(n_total, seed=swap_seed)

    # 3. Check mode execution
    if args.check:
        try:
            # VERIFY-ON-GPU
            adapter.load(device="cuda", dtype="float16")
            run_check_mode(adapter, all_examples, prompt_suffix=args.prompt_suffix, max_new_tokens=args.max_new_tokens)
            return
        except SystemExit as se:
            raise se
        except Exception as e:
            print(f"\n[ERROR] Open-ended --check failed with exception: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

    # 4. Filter needed shard tasks
    n_shards = (n_total + args.shard_size - 1) // args.shard_size
    pair_str = f"_pair{args.pairing_seed}" if args.pairing_seed != 0 else ""
    needed_tasks = []
    for corruption in corruptions_list:
        for shard_idx in range(n_shards):
            shard_name = f"open_{model_slug}_{corruption}{pair_str}_seed{args.seed}_shard{shard_idx}.json"
            shard_file = out_dir / shard_name
            if not shard_file.exists():
                needed_tasks.append((corruption, shard_idx, shard_file))

    if not needed_tasks:
        print(f"\nAll {len(corruptions_list) * n_shards} open-ended shards already exist in {out_dir}. Nothing to do!")
        return

    print(f"\nTarget Model: {args.model} [OPEN-ENDED GENERATION]")
    print(f"Total Questions: {n_total} across {n_shards} shards")
    print(f"Corruptions: {corruptions_list}")
    print(f"Pending tasks: {len(needed_tasks)} shards")
    print(f"Git commit: {git_info['git_commit']}, tag: {git_info['git_tag']}")

    # Load model onto GPU
    adapter.load(device="cuda", dtype="float16")  # VERIFY-ON-GPU

    try:
        model_hf_revision = getattr(adapter.model.config, "_commit_hash", "unknown") or "unknown"
    except Exception:
        model_hf_revision = "unknown"

    start_time = time.time()
    completed_items = 0
    total_items_to_process = len(needed_tasks) * args.shard_size

    for corruption, shard_idx, shard_file in needed_tasks:
        start_idx = shard_idx * args.shard_size
        end_idx = min(start_idx + args.shard_size, n_total)
        shard_examples = all_examples[start_idx:end_idx]

        print(
            f"\n>>> Shard {shard_idx + 1}/{n_shards} "
            f"[{corruption}] (Questions {start_idx}..{end_idx - 1}) <<<",
            flush=True,
        )

        items_records = []
        for q_local_idx, ex in enumerate(shard_examples):
            q_global_idx = start_idx + q_local_idx
            q_id = ex["question_id"]
            question = ex["question"]
            direct_answers = ex.get("direct_answers", [])
            orig_image = ex["image"]

            # 1. Clean reference generation A* (S=0)
            # VERIFY-ON-GPU
            ans_clean = adapter.generate_open_answer(
                orig_image, question, prompt_suffix=args.prompt_suffix, max_new_tokens=args.max_new_tokens
            )
            u_clean = vqa_soft_accuracy(ans_clean, direct_answers)

            # Clean teacher-forced distributions for A*
            p_clean = adapter.teacher_forced_answer_probs(
                orig_image, question, ans_clean, prompt_suffix=args.prompt_suffix
            )

            # 2. Null conditions (removed, black)
            # Removed (text-only)
            ans_removed = adapter.generate_open_answer(
                None, question, prompt_suffix=args.prompt_suffix, max_new_tokens=args.max_new_tokens
            )
            u_removed = vqa_soft_accuracy(ans_removed, direct_answers)
            p_removed = adapter.teacher_forced_answer_probs(
                None, question, ans_clean, prompt_suffix=args.prompt_suffix
            )

            # Black
            blk_img = black_image(orig_image.size)
            ans_black = adapter.generate_open_answer(
                blk_img, question, prompt_suffix=args.prompt_suffix, max_new_tokens=args.max_new_tokens
            )
            u_black = vqa_soft_accuracy(ans_black, direct_answers)
            p_black = adapter.teacher_forced_answer_probs(
                blk_img, question, ans_clean, prompt_suffix=args.prompt_suffix
            )

            # Baseline reliance R(0) against nulls
            r0_removed, _ = sequence_jsd(p_clean, p_removed)
            r0_black, _ = sequence_jsd(p_clean, p_black)

            # Swapped donor image
            donor_idx = int(swap_permutation[q_global_idx])
            donor_image = all_examples[donor_idx]["image"]

            # 3. Probs and answers across severity grid S
            answers_per_s = []
            u_per_s = []
            ans_changed_per_s = []
            r_removed_per_s = []
            r_black_per_s = []
            r_clean_per_s = []

            for s_idx, S in enumerate(S_grid):
                item_seed_s = args.seed * 10000 + q_global_idx * 100 + int(round(S * 100))
                if corruption == "gaussian":
                    corrupted_img = gaussian_blend(orig_image, S=S, seed=item_seed_s)
                elif corruption == "patch":
                    corrupted_img = patch_mask(orig_image, S=S, seed=item_seed_s, patch=28)
                elif corruption == "swapblend":
                    corrupted_img = swap_blend(orig_image, donor_image, S=S)
                else:
                    raise ValueError(f"Unknown corruption kind: {corruption}")

                # Generate greedy answer under degraded image
                ans_s = adapter.generate_open_answer(
                    corrupted_img, question, prompt_suffix=args.prompt_suffix, max_new_tokens=args.max_new_tokens
                )
                u_s = vqa_soft_accuracy(ans_s, direct_answers)
                changed_s = is_answer_changed(ans_s, ans_clean)

                # Teacher force reference answer A* under degraded image
                p_s = adapter.teacher_forced_answer_probs(
                    corrupted_img, question, ans_clean, prompt_suffix=args.prompt_suffix
                )

                r_rem_s, _ = sequence_jsd(p_s, p_removed)
                r_blk_s, _ = sequence_jsd(p_s, p_black)
                r_cln_s, _ = sequence_jsd(p_s, p_clean)

                answers_per_s.append(ans_s)
                u_per_s.append(u_s)
                ans_changed_per_s.append(changed_s)
                r_removed_per_s.append(r_rem_s)
                r_black_per_s.append(r_blk_s)
                r_clean_per_s.append(r_cln_s)

            item_dict = {
                "question_id": q_id,
                "clean_answer": ans_clean,
                "clean_utility": u_clean,
                "null_answers": {
                    "removed": ans_removed,
                    "black": ans_black,
                },
                "null_utility": {
                    "removed": u_removed,
                    "black": u_black,
                },
                "R0": {
                    "removed": r0_removed,
                    "black": r0_black,
                },
                "answers_per_S": answers_per_s,
                "utility_per_S": u_per_s,
                "answer_changed_per_S": ans_changed_per_s,
                "R_removed_per_S": r_removed_per_s,
                "R_black_per_S": r_black_per_s,
                "R_clean_per_S": r_clean_per_s,
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
                    f"ETA: {format_eta(remaining_sec)}",
                    flush=True,
                )

        shard_data: Dict[str, Any] = {
            "mode": "open_ended",
            "model_name": args.model,
            "corruption": corruption,
            "shard_idx": shard_idx,
            "seed": args.seed,
            "pairing_seed": args.pairing_seed,
            "start_idx": start_idx,
            "end_idx": end_idx,
            "n_questions": len(items_records),
            "S_grid": S_grid,
            "items": items_records,
        }

        if not shard_file.exists():
            write_raw(shard_file, shard_data)
            print(f"  -> Successfully saved shard: {shard_file.name}", flush=True)

        config_dict = {
            "mode": "open_ended",
            "model": args.model,
            "corruption": corruption,
            "shard_idx": shard_idx,
            "seed": args.seed,
            "pairing_seed": args.pairing_seed,
            "n_items": len(items_records),
            "grid_points": len(S_grid),
        }
        manifest_dir = out_dir / f"manifest_open_{model_slug}_{corruption}{pair_str}_seed{args.seed}_shard{shard_idx}"
        write_manifest(
            manifest_dir,
            config_dict,
            seed=args.seed,
            model_name=args.model,
            git_commit=git_info.get("git_commit"),
            git_tag=git_info.get("git_tag"),
            model_hf_revision=model_hf_revision,
            assistant_prefix=args.assistant_prefix,
            data_ids_hash=data_ids_hash,
            lib_versions=lib_versions,
            gpu_name=gpu_name,
        )

    print(f"\nAll open-ended benchmark tasks completed in {format_eta(time.time() - start_time)}.")


if __name__ == "__main__":
    run_open_benchmark()
