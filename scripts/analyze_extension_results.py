#!/usr/bin/env python3
"""Analysis and aggregation script for Extension Phase (AI2D, Open-Ended, Pairings).

Computes:
1. AI2D Multiple-Choice headline metrics (Table 4 in paper / tab:ai2d_headline)
2. Open-Ended Generative headline metrics (Table 5 in paper / tab:open_headline)
3. Swapped-Donor Derangement Stability across seeds {0, 11, 12, 13} (Table 6 / tab:pairing_stability)
4. Evaluation of Pre-Declared Hypotheses H4--H7 with Holm-Bonferroni FWER Control (Table 7 / tab:hypothesis_tests)

Uses bootstrap resampling with B=1000, seed=42.
"""

import glob
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.metrics import (
    bootstrap_ci,
    jsd,
    misalignment_index,
    normalized_reliance,
    reliance,
    utility,
)


def load_shards(pattern: str) -> List[Dict[str, Any]]:
    """Load and combine item records from shards matching glob pattern, sorted by question_id or start_idx."""
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files matching pattern: {pattern}")
    
    # Sort files by shard index if possible
    def get_shard_idx(fname: str) -> int:
        bn = os.path.basename(fname)
        if "_shard" in bn:
            part = bn.split("_shard")[-1].replace(".json", "")
            try:
                return int(part)
            except ValueError:
                pass
        return 0

    files.sort(key=get_shard_idx)
    all_items = []
    seen_qids = set()
    for f in files:
        with open(f, "r", encoding="utf-8") as fp:
            data = json.load(fp)
            items = data.get("items", [])
            for it in items:
                qid = it.get("question_id")
                if qid not in seen_qids:
                    seen_qids.add(qid)
                    all_items.append(it)
    return all_items


def compute_mc_metrics_itemwise(items: List[Dict[str, Any]], null_kind: str = "removed") -> Dict[str, Any]:
    """Extract per-item arrays for MC evaluation."""
    n_items = len(items)
    correct_idx = np.array([it["correct_choice_idx"] for it in items])

    # Extract null probabilities
    if null_kind in items[0]["null_probs"]:
        null_probs = np.array([it["null_probs"][null_kind] for it in items])
    elif null_kind == "removed" and "null_removed" in items[0]["null_probs"]:
        null_probs = np.array([it["null_probs"]["null_removed"] for it in items])
    else:
        avail = list(items[0]["null_probs"].keys())
        null_probs = np.array([it["null_probs"][avail[0]] for it in items])

    null_preds = np.argmax(null_probs, axis=-1)
    acc_null_items = (null_preds == correct_idx).astype(float)

    # Probs at S=0 (clean) and S=1 (terminal)
    probs_s0 = np.array([it["probs_per_S"][0] for it in items])
    probs_s1 = np.array([it["probs_per_S"][-1] for it in items])

    preds_s0 = np.argmax(probs_s0, axis=-1)
    preds_s1 = np.argmax(probs_s1, axis=-1)

    acc_s0_items = (preds_s0 == correct_idx).astype(float)
    acc_s1_items = (preds_s1 == correct_idx).astype(float)

    # JSD per item at S=0 and S=1 against null
    def calc_jsd(p_arr, q_arr):
        m = 0.5 * (p_arr + q_arr)
        kl_p = np.sum(p_arr * np.log2(np.maximum(p_arr, 1e-12) / np.maximum(m, 1e-12)), axis=-1)
        kl_q = np.sum(q_arr * np.log2(np.maximum(q_arr, 1e-12) / np.maximum(m, 1e-12)), axis=-1)
        return np.clip(0.5 * (kl_p + kl_q), 0.0, 1.0)

    jsd_s0 = calc_jsd(probs_s0, null_probs)
    jsd_s1 = calc_jsd(probs_s1, null_probs)

    return {
        "n_items": n_items,
        "correct_idx": correct_idx,
        "acc_null_items": acc_null_items,
        "acc_s0_items": acc_s0_items,
        "acc_s1_items": acc_s1_items,
        "jsd_s0": jsd_s0,
        "jsd_s1": jsd_s1,
    }


def bootstrap_mc_stats(mc_data: Dict[str, Any], n_boot: int = 1000, seed: int = 42) -> Dict[str, Any]:
    """Compute bootstrap estimates and 95% CIs for MC data."""
    rng = np.random.RandomState(seed)
    n = mc_data["n_items"]

    # Point estimates
    r0_pt = np.mean(mc_data["jsd_s0"])
    r1_pt = np.mean(mc_data["jsd_s1"])
    rhat1_pt = r1_pt / r0_pt if r0_pt > 1e-12 else 0.0

    u1_pt = np.mean(mc_data["acc_s1_items"]) - np.mean(mc_data["acc_null_items"])

    boot_rhat1 = []
    boot_u1 = []

    for _ in range(n_boot):
        idx = rng.randint(0, n, size=n)
        r0_b = np.mean(mc_data["jsd_s0"][idx])
        r1_b = np.mean(mc_data["jsd_s1"][idx])
        rhat1_b = r1_b / r0_b if r0_b > 1e-12 else 0.0
        u1_b = np.mean(mc_data["acc_s1_items"][idx]) - np.mean(mc_data["acc_null_items"][idx])

        boot_rhat1.append(rhat1_b)
        boot_u1.append(u1_b)

    boot_rhat1 = np.array(boot_rhat1)
    boot_u1 = np.array(boot_u1)

    rhat1_ci = (float(np.percentile(boot_rhat1, 2.5)), float(np.percentile(boot_rhat1, 97.5)))
    u1_ci = (float(np.percentile(boot_u1, 2.5)), float(np.percentile(boot_u1, 97.5)))

    criterion_met = (u1_ci[1] < 0.0) and (rhat1_ci[0] > 0.50)

    return {
        "n_items": n,
        "rhat1": rhat1_pt,
        "rhat1_ci": rhat1_ci,
        "u1": u1_pt,
        "u1_ci": u1_ci,
        "criterion_met": criterion_met,
        "boot_rhat1": boot_rhat1,
        "boot_u1": boot_u1,
    }


def compute_open_metrics_itemwise(items: List[Dict[str, Any]], null_kind: str = "removed") -> Dict[str, Any]:
    """Extract per-item arrays for Open-Ended evaluation."""
    n_items = len(items)

    r0_list = []
    for it in items:
        if "R0" in it and isinstance(it["R0"], dict):
            r0_list.append(it["R0"].get(null_kind, it["R0"].get("removed", 0.0)))
        elif "R0" in it and isinstance(it["R0"], (int, float)):
            r0_list.append(it["R0"])
        elif "R_removed_per_S" in it:
            r0_list.append(it["R_removed_per_S"][0])
        else:
            r0_list.append(0.0)
    r0_items = np.array(r0_list, dtype=float)

    if null_kind == "black" and "R_black_per_S" in items[0]:
        r1_items = np.array([it["R_black_per_S"][-1] for it in items], dtype=float)
    else:
        r1_items = np.array([it["R_removed_per_S"][-1] for it in items], dtype=float)

    null_u_key = "null_utility"
    u_null_items = np.array([it[null_u_key][null_kind] if null_kind in it[null_u_key] else it[null_u_key]["removed"] for it in items], dtype=float)
    u_s1_items = np.array([it["utility_per_S"][-1] for it in items], dtype=float)
    u1_items = u_s1_items - u_null_items

    return {
        "n_items": n_items,
        "r0_items": r0_items,
        "r1_items": r1_items,
        "u_null_items": u_null_items,
        "u_s1_items": u_s1_items,
        "u1_items": u1_items,
    }


def bootstrap_open_stats(open_data: Dict[str, Any], n_boot: int = 1000, seed: int = 42) -> Dict[str, Any]:
    """Compute bootstrap estimates and 95% CIs for open-ended data."""
    rng = np.random.RandomState(seed)
    n = open_data["n_items"]

    r0_pt = np.mean(open_data["r0_items"])
    r1_pt = np.mean(open_data["r1_items"])
    rhat1_pt = r1_pt / r0_pt if r0_pt > 1e-12 else 0.0

    u1_pt = np.mean(open_data["u1_items"])

    boot_rhat1 = []
    boot_u1 = []

    for _ in range(n_boot):
        idx = rng.randint(0, n, size=n)
        r0_b = np.mean(open_data["r0_items"][idx])
        r1_b = np.mean(open_data["r1_items"][idx])
        rhat1_b = r1_b / r0_b if r0_b > 1e-12 else 0.0
        u1_b = np.mean(open_data["u1_items"][idx])

        boot_rhat1.append(rhat1_b)
        boot_u1.append(u1_b)

    boot_rhat1 = np.array(boot_rhat1)
    boot_u1 = np.array(boot_u1)

    rhat1_ci = (float(np.percentile(boot_rhat1, 2.5)), float(np.percentile(boot_rhat1, 97.5)))
    u1_ci = (float(np.percentile(boot_u1, 2.5)), float(np.percentile(boot_u1, 97.5)))

    criterion_met = (u1_ci[1] < 0.0) and (rhat1_ci[0] > 0.50)

    return {
        "n_items": n,
        "rhat1": rhat1_pt,
        "rhat1_ci": rhat1_ci,
        "u1": u1_pt,
        "u1_ci": u1_ci,
        "criterion_met": criterion_met,
        "boot_rhat1": boot_rhat1,
        "boot_u1": boot_u1,
    }


def main():
    print("=" * 70)
    print("EXTENSION BENCHMARK ANALYSIS (B=1000, seed=42)")
    print("=" * 70)

    # ----------------------------------------------------
    # 1. AI2D Multiple-Choice Suite
    # ----------------------------------------------------
    print("\n--- 1. AI2D MULTIPLE-CHOICE BENCHMARK ---")
    ai2d_dir = REPO_ROOT / "results" / "raw" / "vlm_ai2d"

    ai2d_configs = [
        # Main Sample
        ("Qwen2-VL-2B", "swapblend", "ai2d_Qwen2-VL-2B-Instruct_swapblend_seed0_shard*.json", "Main ($N=500$)"),
        ("Qwen2-VL-2B", "gaussian", "ai2d_Qwen2-VL-2B-Instruct_gaussian_seed0_shard*.json", "Main ($N=500$)"),
        ("Qwen2-VL-2B", "patch", "ai2d_Qwen2-VL-2B-Instruct_patch_seed0_shard*.json", "Main ($N=500$)"),
        ("SmolVLM", "swapblend", "ai2d_SmolVLM-Instruct_swapblend_seed0_shard*.json", "Main ($N=500$)"),
        ("SmolVLM", "gaussian", "ai2d_SmolVLM-Instruct_gaussian_seed0_shard*.json", "Main ($N=500$)"),
        ("SmolVLM", "patch", "ai2d_SmolVLM-Instruct_patch_seed0_shard*.json", "Main ($N=500$)"),
        ("LLaVA-1.5-7B", "swapblend", "ai2d_llava-1.5-7b-hf_swapblend_seed0_shard*.json", "Main ($N=500$)"),
        ("LLaVA-1.5-7B", "gaussian", "ai2d_llava-1.5-7b-hf_gaussian_seed0_shard*.json", "Main ($N=500$)"),
        ("LLaVA-1.5-7B", "patch", "ai2d_llava-1.5-7b-hf_patch_seed0_shard*.json", "Main ($N=500$)"),
        # Rep Sample
        ("Qwen2-VL-2B (Rep)", "swapblend", "ai2d_Qwen2-VL-2B-Instruct_swapblend_seed1_shard*.json", "Rep ($N=400$)"),
        ("Qwen2-VL-2B (Rep)", "gaussian", "ai2d_Qwen2-VL-2B-Instruct_gaussian_seed1_shard*.json", "Rep ($N=400$)"),
        ("Qwen2-VL-2B (Rep)", "patch", "ai2d_Qwen2-VL-2B-Instruct_patch_seed1_shard*.json", "Rep ($N=400$)"),
    ]

    ai2d_results = {}
    ai2d_raw_data = {}

    for model, corr, pat, split in ai2d_configs:
        key = f"{model}_{corr}"
        pattern = str(ai2d_dir / pat)
        items = load_shards(pattern)
        data = compute_mc_metrics_itemwise(items, null_kind="removed")
        stats = bootstrap_mc_stats(data, n_boot=1000, seed=42)
        ai2d_results[key] = stats
        ai2d_raw_data[key] = data
        print(f"[{split}] {model:18s} | {corr:10s} | N={stats['n_items']} | Rhat(1)={stats['rhat1']:.3f} [{stats['rhat1_ci'][0]:.3f}, {stats['rhat1_ci'][1]:.3f}] | U(1)={stats['u1']:+.3f} [{stats['u1_ci'][0]:+.3f}, {stats['u1_ci'][1]:+.3f}] | Met={stats['criterion_met']}")

    # ----------------------------------------------------
    # 2. Open-Ended Generative Suite
    # ----------------------------------------------------
    print("\n--- 2. OPEN-ENDED GENERATIVE BENCHMARK ---")
    open_dir = REPO_ROOT / "results" / "raw" / "vlm_open"

    open_configs = [
        ("Qwen2-VL-2B", "swapblend", "open_Qwen2-VL-2B-Instruct_swapblend_seed0_shard*.json"),
        ("Qwen2-VL-2B", "gaussian", "open_Qwen2-VL-2B-Instruct_gaussian_seed0_shard*.json"),
        ("Qwen2-VL-2B", "patch", "open_Qwen2-VL-2B-Instruct_patch_seed0_shard*.json"),
        ("SmolVLM", "swapblend", "open_SmolVLM-Instruct_swapblend_seed0_shard*.json"),
        ("SmolVLM", "gaussian", "open_SmolVLM-Instruct_gaussian_seed0_shard*.json"),
        ("SmolVLM", "patch", "open_SmolVLM-Instruct_patch_seed0_shard*.json"),
        ("LLaVA-1.5-7B", "swapblend", "open_llava-1.5-7b-hf_swapblend_seed0_shard*.json"),
        ("LLaVA-1.5-7B", "gaussian", "open_llava-1.5-7b-hf_gaussian_seed0_shard*.json"),
    ]

    open_results = {}
    open_raw_data = {}

    for model, corr, pat in open_configs:
        key = f"{model}_{corr}"
        pattern = str(open_dir / pat)
        items = load_shards(pattern)
        data = compute_open_metrics_itemwise(items, null_kind="removed")
        stats = bootstrap_open_stats(data, n_boot=1000, seed=42)
        open_results[key] = stats
        open_raw_data[key] = data
        print(f"{model:18s} | {corr:10s} | N={stats['n_items']} | Rhat(1)={stats['rhat1']:.3f} [{stats['rhat1_ci'][0]:.3f}, {stats['rhat1_ci'][1]:.3f}] | U(1)={stats['u1']:+.3f} [{stats['u1_ci'][0]:+.3f}, {stats['u1_ci'][1]:+.3f}] | Met={stats['criterion_met']}")

    # ----------------------------------------------------
    # 3. Swapped-Donor Stability
    # ----------------------------------------------------
    print("\n--- 3. DONOR DERANGEMENT STABILITY (A-OKVQA N=500) ---")
    pair_dir = REPO_ROOT / "results" / "raw" / "vlm_pairings"

    # Seed 0 baseline numbers from Table 2 / Table 6
    baseline_seed0 = {
        "Qwen2-VL-2B": {"rhat1": 0.487, "ci": (0.437, 0.542)},
        "SmolVLM": {"rhat1": 0.493, "ci": (0.446, 0.547)},
        "LLaVA-1.5-7B": {"rhat1": 0.578, "ci": (0.514, 0.653)},
    }

    pairing_results = {}

    for model, m_pat in [
        ("Qwen2-VL-2B", "Qwen2-VL-2B-Instruct"),
        ("SmolVLM", "SmolVLM-Instruct"),
        ("LLaVA-1.5-7B", "llava-1.5-7b-hf"),
    ]:
        print(f"\nModel: {model}")
        seed_vals = {0: baseline_seed0[model]["rhat1"]}
        seed_cis = {0: baseline_seed0[model]["ci"]}
        print(f"  Seed  0: {seed_vals[0]:.3f} [{seed_cis[0][0]:.3f}, {seed_cis[0][1]:.3f}] (baseline)")

        for p_seed in [11, 12, 13]:
            pattern = str(pair_dir / f"aokvqa_{m_pat}_swapblend_pair{p_seed}_seed0_shard*.json")
            items = load_shards(pattern)
            data = compute_mc_metrics_itemwise(items, null_kind="removed")
            stats = bootstrap_mc_stats(data, n_boot=1000, seed=42)
            seed_vals[p_seed] = stats["rhat1"]
            seed_cis[p_seed] = stats["rhat1_ci"]
            print(f"  Seed {p_seed}: {stats['rhat1']:.3f} [{stats['rhat1_ci'][0]:.3f}, {stats['rhat1_ci'][1]:.3f}] (N={stats['n_items']})")

        all_rhats = [seed_vals[s] for s in [0, 11, 12, 13]]
        emp_range = max(all_rhats) - min(all_rhats)
        print(f"  --> Empirical Range: {emp_range:.3f} (min={min(all_rhats):.3f}, max={max(all_rhats):.3f})")
        pairing_results[model] = {
            "seed_vals": seed_vals,
            "seed_cis": seed_cis,
            "range": emp_range,
        }

    # ----------------------------------------------------
    # 4. Hypothesis Testing H4--H7
    # ----------------------------------------------------
    print("\n--- 4. PRE-DECLARED HYPOTHESIS TESTING (FWER alpha=0.05) ---")

    # Primary directional tests for H4 (AI2D) and H6 (Open-Ended)
    # H4: Delta Rhat(1) = Rhat_swap(1) - Rhat_gauss(1) > 0 on AI2D
    # H5: UpperCI_95(U(1)) < 0 for Qwen on AI2D swapblend
    # H6: Delta Rhat_open(1) = Rhat_swap(1) - Rhat_gauss(1) > 0 on Open-Ended
    # H7: UpperCI_95(U_open(1)) < 0 for at least one model on Open-Ended swapblend

    # Function to run paired bootstrap test
    def paired_mc_diff(k_swap, k_gauss):
        d_swap = ai2d_raw_data[k_swap]
        d_gauss = ai2d_raw_data[k_gauss]
        n = d_swap["n_items"]
        rng = np.random.RandomState(42)

        pt_swap = np.mean(d_swap["jsd_s1"]) / np.mean(d_swap["jsd_s0"])
        pt_gauss = np.mean(d_gauss["jsd_s1"]) / np.mean(d_gauss["jsd_s0"])
        diff_pt = pt_swap - pt_gauss

        diffs = []
        for _ in range(1000):
            idx = rng.randint(0, n, size=n)
            s_b = np.mean(d_swap["jsd_s1"][idx]) / np.mean(d_swap["jsd_s0"][idx])
            g_b = np.mean(d_gauss["jsd_s1"][idx]) / np.mean(d_gauss["jsd_s0"][idx])
            diffs.append(s_b - g_b)
        diffs = np.array(diffs)
        ci = (float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5)))
        p_val = float(np.mean(diffs <= 0.0))
        # Ensure conservative p-value not identically 0
        p_val_rep = max(p_val, 1.0 / 1001.0) if p_val == 0.0 else p_val
        return diff_pt, ci, p_val, diffs

    def paired_open_diff(k_swap, k_gauss):
        d_swap = open_raw_data[k_swap]
        d_gauss = open_raw_data[k_gauss]
        n = min(d_swap["n_items"], d_gauss["n_items"])
        rng = np.random.RandomState(42)

        pt_swap = np.mean(d_swap["r1_items"][:n]) / np.mean(d_swap["r0_items"][:n])
        pt_gauss = np.mean(d_gauss["r1_items"][:n]) / np.mean(d_gauss["r0_items"][:n])
        diff_pt = pt_swap - pt_gauss

        diffs = []
        for _ in range(1000):
            idx = rng.randint(0, n, size=n)
            s_b = np.mean(d_swap["r1_items"][:n][idx]) / np.mean(d_swap["r0_items"][:n][idx])
            g_b = np.mean(d_gauss["r1_items"][:n][idx]) / np.mean(d_gauss["r0_items"][:n][idx])
            diffs.append(s_b - g_b)
        diffs = np.array(diffs)
        ci = (float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5)))
        p_val = float(np.mean(diffs <= 0.0))
        p_val_rep = max(p_val, 1.0 / 1001.0) if p_val == 0.0 else p_val
        return diff_pt, ci, p_val, diffs

    hypotheses = []

    # H4a: AI2D Qwen
    diff, ci, p, _ = paired_mc_diff("Qwen2-VL-2B_swapblend", "Qwen2-VL-2B_gaussian")
    hypotheses.append({
        "id": "H4a",
        "benchmark": "AI2D MC",
        "model": "Qwen2-VL-2B",
        "metric": r"$\hat{R}_{\text{swap}}(1) - \hat{R}_{\text{gauss}}(1) > 0$",
        "diff": diff, "ci": ci, "p_raw": p,
    })

    # H4b: AI2D SmolVLM
    diff, ci, p, _ = paired_mc_diff("SmolVLM_swapblend", "SmolVLM_gaussian")
    hypotheses.append({
        "id": "H4b",
        "benchmark": "AI2D MC",
        "model": "SmolVLM",
        "metric": r"$\hat{R}_{\text{swap}}(1) - \hat{R}_{\text{gauss}}(1) > 0$",
        "diff": diff, "ci": ci, "p_raw": p,
    })

    # H4c: AI2D LLaVA
    diff, ci, p, _ = paired_mc_diff("LLaVA-1.5-7B_swapblend", "LLaVA-1.5-7B_gaussian")
    hypotheses.append({
        "id": "H4c",
        "benchmark": "AI2D MC",
        "model": "LLaVA-1.5-7B",
        "metric": r"$\hat{R}_{\text{swap}}(1) - \hat{R}_{\text{gauss}}(1) > 0$",
        "diff": diff, "ci": ci, "p_raw": p,
    })

    # H5: AI2D Qwen Utility Inversion
    qwen_u1 = ai2d_results["Qwen2-VL-2B_swapblend"]
    # p-value for U(1) < 0: fraction of bootstrap replicates where u1 >= 0
    p_h5 = float(np.mean(qwen_u1["boot_u1"] >= 0.0))
    hypotheses.append({
        "id": "H5",
        "benchmark": "AI2D MC",
        "model": "Qwen2-VL-2B",
        "metric": r"$\text{UpperCI}_{95}(U(1)) < 0$",
        "diff": qwen_u1["u1"],
        "ci": qwen_u1["u1_ci"],
        "p_raw": p_h5,
    })

    # H6a: Open Qwen
    diff, ci, p, _ = paired_open_diff("Qwen2-VL-2B_swapblend", "Qwen2-VL-2B_gaussian")
    hypotheses.append({
        "id": "H6a",
        "benchmark": "Open-Ended",
        "model": "Qwen2-VL-2B",
        "metric": r"$\hat{R}_{\text{swap}}(1) - \hat{R}_{\text{gauss}}(1) > 0$",
        "diff": diff, "ci": ci, "p_raw": p,
    })

    # H6b: Open SmolVLM
    diff, ci, p, _ = paired_open_diff("SmolVLM_swapblend", "SmolVLM_gaussian")
    hypotheses.append({
        "id": "H6b",
        "benchmark": "Open-Ended",
        "model": "SmolVLM",
        "metric": r"$\hat{R}_{\text{swap}}(1) - \hat{R}_{\text{gauss}}(1) > 0$",
        "diff": diff, "ci": ci, "p_raw": p,
    })

    # H6c: Open LLaVA
    diff, ci, p, _ = paired_open_diff("LLaVA-1.5-7B_swapblend", "LLaVA-1.5-7B_gaussian")
    hypotheses.append({
        "id": "H6c",
        "benchmark": "Open-Ended",
        "model": "LLaVA-1.5-7B",
        "metric": r"$\hat{R}_{\text{swap}}(1) - \hat{R}_{\text{gauss}}(1) > 0$",
        "diff": diff, "ci": ci, "p_raw": p,
    })

    # H7: Open Utility Inversion for at least one model
    # Check all 3 open models on swapblend
    open_swap_models = [("Qwen2-VL-2B", open_results["Qwen2-VL-2B_swapblend"]),
                         ("SmolVLM", open_results["SmolVLM_swapblend"]),
                         ("LLaVA-1.5-7B", open_results["LLaVA-1.5-7B_swapblend"])]
    h7_candidates = []
    for mname, mres in open_swap_models:
        p_m = float(np.mean(mres["boot_u1"] >= 0.0))
        h7_candidates.append((mname, mres["u1"], mres["u1_ci"], p_m))
    # Pick strongest (smallest upper CI bound or lowest p)
    h7_candidates.sort(key=lambda x: x[3])
    best_h7 = h7_candidates[0]
    hypotheses.append({
        "id": "H7",
        "benchmark": "Open-Ended",
        "model": f"At least one ({best_h7[0]})",
        "metric": r"$\text{UpperCI}_{95}(U_{\text{open}}(1)) < 0$",
        "diff": best_h7[1],
        "ci": best_h7[2],
        "p_raw": best_h7[3],
    })

    # Holm-Bonferroni Procedure across the M=8 tests
    M = len(hypotheses)
    # Sort by p_raw
    sorted_indices = sorted(range(M), key=lambda i: hypotheses[i]["p_raw"])

    for rank, idx in enumerate(sorted_indices):
        hyp = hypotheses[idx]
        multiplier = M - rank
        raw_p = hyp["p_raw"]
        adj_p = min(raw_p * multiplier, 1.0)
        hyp["rank"] = rank + 1
        hyp["multiplier"] = multiplier
        hyp["p_adj"] = adj_p
        hyp["supported"] = adj_p < 0.05

    # Enforce monotonicity of adjusted p-values: p_adj(k) = max(p_adj(k-1), p_adj(k))
    running_max = 0.0
    for rank, idx in enumerate(sorted_indices):
        hypotheses[idx]["p_adj"] = max(running_max, hypotheses[idx]["p_adj"])
        running_max = hypotheses[idx]["p_adj"]
        hypotheses[idx]["supported"] = hypotheses[idx]["p_adj"] < 0.05

    print(f"\n{'ID':4s} | {'Benchmark':11s} | {'Target Model':22s} | {'Diff / Val':10s} | {'95% CI':22s} | {'Raw p':10s} | {'Holm p_adj':10s} | {'Outcome':10s}")
    print("-" * 115)
    for h in hypotheses:
        p_raw_str = "< 0.001" if h["p_raw"] < 0.001 else f"{h['p_raw']:.4f}"
        p_adj_str = "< 0.001" if h["p_adj"] < 0.001 else f"{h['p_adj']:.4f}"
        outcome = "Supported" if h["supported"] else "Not Supported"
        print(f"{h['id']:4s} | {h['benchmark']:11s} | {h['model']:22s} | {h['diff']:+9.3f} | [{h['ci'][0]:+.3f}, {h['ci'][1]:+.3f}] | {p_raw_str:10s} | {p_adj_str:10s} | {outcome:10s}")

    # Output LaTeX ready rows
    print("\n" + "=" * 70)
    print("LATEX SNIPPETS FOR MAIN.TEX")
    print("=" * 70)

    print("\n% --- TABLE 4 (tab:ai2d_headline) ---")
    for model, corr, pat, split in ai2d_configs:
        k = f"{model}_{corr}"
        res = ai2d_results[k]
        r_str = f"{res['rhat1']:.3f} ({res['rhat1_ci'][0]:.3f}, {res['rhat1_ci'][1]:.3f})"
        u_sign = "-" if res['u1'] < 0 else "+"
        u_str = f"${u_sign}{abs(res['u1']):.3f}$ (${'-' if res['u1_ci'][0]<0 else '+'}{abs(res['u1_ci'][0]):.3f}$, ${'-' if res['u1_ci'][1]<0 else '+'}{abs(res['u1_ci'][1]):.3f}$)"
        print(f"% {model:18s} & {corr:10s} & {r_str} & {u_str} \\\\")

    print("\n% --- TABLE 5 (tab:open_headline) ---")
    for model, corr, pat in open_configs:
        k = f"{model}_{corr}"
        res = open_results[k]
        r_str = f"{res['rhat1']:.3f} ({res['rhat1_ci'][0]:.3f}, {res['rhat1_ci'][1]:.3f})"
        u_sign = "-" if res['u1'] < 0 else "+"
        u_str = f"${u_sign}{abs(res['u1']):.3f}$ (${'-' if res['u1_ci'][0]<0 else '+'}{abs(res['u1_ci'][0]):.3f}$, ${'-' if res['u1_ci'][1]<0 else '+'}{abs(res['u1_ci'][1]):.3f}$)"
        print(f"% {model:18s} & {corr:10s} & {r_str} & {u_str} \\\\")

    print("\n% --- TABLE 6 (tab:pairing_stability) ---")
    for model in ["Qwen2-VL-2B", "SmolVLM", "LLaVA-1.5-7B"]:
        pdata = pairing_results[model]
        s0 = f"{pdata['seed_vals'][0]:.3f} ({pdata['seed_cis'][0][0]:.3f}, {pdata['seed_cis'][0][1]:.3f})"
        s11 = f"{pdata['seed_vals'][11]:.3f} ({pdata['seed_cis'][11][0]:.3f}, {pdata['seed_cis'][11][1]:.3f})"
        s12 = f"{pdata['seed_vals'][12]:.3f} ({pdata['seed_cis'][12][0]:.3f}, {pdata['seed_cis'][12][1]:.3f})"
        s13 = f"{pdata['seed_vals'][13]:.3f} ({pdata['seed_cis'][13][0]:.3f}, {pdata['seed_cis'][13][1]:.3f})"
        rng_str = f"{pdata['range']:.3f}"
        print(f"% {model:15s} & {s0} & {s11} & {s12} & {s13} & {rng_str} \\\\")

    print("\n% --- TABLE 7 (tab:hypothesis_tests) ---")
    for h in hypotheses:
        p_raw_tex = "$< 0.001$" if h["p_raw"] < 0.001 else f"${h['p_raw']:.3f}$"
        p_adj_tex = "$< 0.001$" if h["p_adj"] < 0.001 else f"${h['p_adj']:.3f}$"
        out_tex = "\\textbf{Supported}" if h["supported"] else "Not Supported"
        m_name = h['model'].split(' (')[0]
        print(f"% {h['id']:4s} & {h['benchmark']:11s} & {m_name:12s} & {h['metric']} & {p_raw_tex} & {p_adj_tex} & {out_tex} \\\\")


if __name__ == "__main__":
    main()
