"""Post Hoc Baseline Visual Reliance Metrics and Spearman Correlation Analysis.

POST HOC ANALYSIS:
This script performs a post hoc investigation into static baseline visual
sensitivity and reliance indicators across vision-language models (VLMs) on
the A-OKVQA benchmark. It does not alter metric functions or raw data.

Metrics evaluated per model (main and replication separately):
  1. VRS (Visual Reliance Sensitivity):
     VRS = acc(clean) - acc(swapped)
  2. Blank Drop:
     Blank Drop = acc(clean) - acc(black)
  3. Image Sensitivity:
     P[argmax(clean) != argmax(swapped)]
  4. Visual Benefit Rate:
     P[clean correct & swapped wrong]
  5. Visual Harm Rate:
     P[clean wrong & swapped correct]

All metrics are computed with a 95% bootstrap confidence interval
(B=1000 resamples, random seed 42).

In addition, the script evaluates the Spearman rank correlation across models
and corruption conditions between these static baseline metrics and the
dynamic terminal normalized reliance R_hat(1).
"""

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple

# Ensure repo root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
from scipy import stats

from scripts.analyze_vlm_all import load_shards_grouped


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Post hoc baseline visual reliance metrics and correlation analysis"
    )
    parser.add_argument(
        "--main_raw_dir",
        type=str,
        default="results/raw/vlm_main",
        help="Path to main benchmark raw shards directory",
    )
    parser.add_argument(
        "--rep_raw_dir",
        type=str,
        default="results/raw/vlm_rep",
        help="Path to replication benchmark raw shards directory",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="results/derived",
        help="Directory to save output summary CSVs",
    )
    parser.add_argument(
        "--n_boot",
        type=int,
        default=1000,
        help="Number of bootstrap resamples (default: 1000)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for bootstrap reproducibility (default: 42)",
    )
    return parser.parse_args()


def compute_baseline_metrics_for_items(
    items: List[Dict[str, Any]],
    n_boot: int = 1000,
    seed: int = 42,
) -> Tuple[Dict[str, float], Dict[str, Tuple[float, float]]]:
    """Compute baseline metrics with 95% bootstrap CIs from a list of item dictionaries."""
    n = len(items)
    if n == 0:
        raise ValueError("Cannot compute metrics on empty item list.")

    y = np.array([it["correct_choice_idx"] for it in items])
    p_clean = np.array([it["probs_per_S"][0] for it in items])
    p_swapped = np.array([it["swapped_probs"] for it in items])
    p_black = np.array([it["null_probs"]["black"] for it in items])

    y_clean = np.argmax(p_clean, axis=-1)
    y_swapped = np.argmax(p_swapped, axis=-1)
    y_black = np.argmax(p_black, axis=-1)

    c_clean = (y_clean == y).astype(float)
    c_swapped = (y_swapped == y).astype(float)
    c_black = (y_black == y).astype(float)

    vrs = c_clean - c_swapped
    blank_drop = c_clean - c_black
    img_sens = (y_clean != y_swapped).astype(float)
    benefit = ((c_clean == 1.0) & (c_swapped == 0.0)).astype(float)
    harm = ((c_clean == 0.0) & (c_swapped == 1.0)).astype(float)

    point_estimates = {
        "acc_clean": float(np.mean(c_clean)),
        "acc_swapped": float(np.mean(c_swapped)),
        "acc_black": float(np.mean(c_black)),
        "VRS": float(np.mean(vrs)),
        "Blank_Drop": float(np.mean(blank_drop)),
        "Image_Sensitivity": float(np.mean(img_sens)),
        "Visual_Benefit_Rate": float(np.mean(benefit)),
        "Visual_Harm_Rate": float(np.mean(harm)),
    }

    # Bootstrap resampling
    rng = np.random.default_rng(seed)
    keys_to_boot = ["VRS", "Blank_Drop", "Image_Sensitivity", "Visual_Benefit_Rate", "Visual_Harm_Rate"]
    boot_samples = {k: np.zeros(n_boot, dtype=float) for k in keys_to_boot}

    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot_samples["VRS"][b] = np.mean(vrs[idx])
        boot_samples["Blank_Drop"][b] = np.mean(blank_drop[idx])
        boot_samples["Image_Sensitivity"][b] = np.mean(img_sens[idx])
        boot_samples["Visual_Benefit_Rate"][b] = np.mean(benefit[idx])
        boot_samples["Visual_Harm_Rate"][b] = np.mean(harm[idx])

    ci_estimates = {}
    for k in keys_to_boot:
        lo = float(np.percentile(boot_samples[k], 2.5))
        hi = float(np.percentile(boot_samples[k], 97.5))
        ci_estimates[k] = (lo, hi)

    return point_estimates, ci_estimates


def run_baseline_metrics(
    main_raw_dir: Path,
    rep_raw_dir: Path,
    out_dir: Path,
    n_boot: int = 1000,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Execute end-to-end post hoc baseline metric evaluation and correlation checks."""
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("POST HOC BASELINE VISUAL RELIANCE ANALYSIS")
    print(f"Bootstrap settings: B={n_boot}, seed={seed}")
    print("=" * 80)

    # 1. Load main shards (scans sibling directories vlm_llava, vlm_smol automatically)
    groups_main = load_shards_grouped(main_raw_dir, no_siblings=False)
    # 2. Load replication shards (in isolation)
    groups_rep = load_shards_grouped(rep_raw_dir, no_siblings=True)

    models_ordered = [
        "Qwen/Qwen2-VL-2B-Instruct",
        "HuggingFaceTB/SmolVLM-Instruct",
        "llava-hf/llava-1.5-7b-hf",
    ]

    summary_records = []
    model_metrics_map = {}

    for split_name, groups, split_seed in [("main", groups_main, 0), ("replication", groups_rep, 1)]:
        print(f"\n--- Split: {split_name.upper()} (seed={split_seed}) ---")
        for m in models_ordered:
            # Pick any available corruption group for this model to extract its clean/swapped/black items
            avail_key = next((k for k in groups if k[0] == m and k[2] == split_seed), None)
            if avail_key is None:
                print(f"[WARN] No shards found for model {m} in split {split_name}.")
                continue

            items = groups[avail_key]["items"]
            n_items = len(items)
            mets, cis = compute_baseline_metrics_for_items(items, n_boot=n_boot, seed=seed)
            model_metrics_map[(split_name, m)] = (mets, cis)

            print(f"\nModel: {m} (N={n_items})")
            print(f"  Clean Accuracy   : {mets['acc_clean']:.4f}")
            print(f"  Swapped Accuracy : {mets['acc_swapped']:.4f}")
            print(f"  Black Accuracy   : {mets['acc_black']:.4f}")
            print(f"  VRS              : {mets['VRS']:.4f}  95% CI [{cis['VRS'][0]:.4f}, {cis['VRS'][1]:.4f}]")
            print(f"  Blank Drop       : {mets['Blank_Drop']:.4f}  95% CI [{cis['Blank_Drop'][0]:.4f}, {cis['Blank_Drop'][1]:.4f}]")
            print(f"  Image Sensitivity: {mets['Image_Sensitivity']:.4f}  95% CI [{cis['Image_Sensitivity'][0]:.4f}, {cis['Image_Sensitivity'][1]:.4f}]")
            print(f"  Visual Benefit   : {mets['Visual_Benefit_Rate']:.4f}  95% CI [{cis['Visual_Benefit_Rate'][0]:.4f}, {cis['Visual_Benefit_Rate'][1]:.4f}]")
            print(f"  Visual Harm      : {mets['Visual_Harm_Rate']:.4f}  95% CI [{cis['Visual_Harm_Rate'][0]:.4f}, {cis['Visual_Harm_Rate'][1]:.4f}]")

            summary_records.append({
                "split": split_name,
                "model_name": m,
                "n_items": n_items,
                "acc_clean": mets["acc_clean"],
                "acc_swapped": mets["acc_swapped"],
                "acc_black": mets["acc_black"],
                "VRS": mets["VRS"],
                "VRS_ci_lo": cis["VRS"][0],
                "VRS_ci_hi": cis["VRS"][1],
                "Blank_Drop": mets["Blank_Drop"],
                "Blank_Drop_ci_lo": cis["Blank_Drop"][0],
                "Blank_Drop_ci_hi": cis["Blank_Drop"][1],
                "Image_Sensitivity": mets["Image_Sensitivity"],
                "Image_Sensitivity_ci_lo": cis["Image_Sensitivity"][0],
                "Image_Sensitivity_ci_hi": cis["Image_Sensitivity"][1],
                "Visual_Benefit_Rate": mets["Visual_Benefit_Rate"],
                "Visual_Benefit_Rate_ci_lo": cis["Visual_Benefit_Rate"][0],
                "Visual_Benefit_Rate_ci_hi": cis["Visual_Benefit_Rate"][1],
                "Visual_Harm_Rate": mets["Visual_Harm_Rate"],
                "Visual_Harm_Rate_ci_lo": cis["Visual_Harm_Rate"][0],
                "Visual_Harm_Rate_ci_hi": cis["Visual_Harm_Rate"][1],
            })

    metrics_df = pd.DataFrame(summary_records)
    metrics_csv = out_dir / "baseline_metrics_summary.csv"
    metrics_df.to_csv(metrics_csv, index=False)
    print(f"\nSaved baseline summary: {metrics_csv}")

    # 3. Load pre-computed R_hat(1) from derived summary CSVs
    main_summary_csv = out_dir / "vlm_main_summary.csv"
    rep_summary_csv = out_dir / "vlm_rep" / "vlm_main_summary.csv"

    cond_records = []
    if main_summary_csv.exists():
        df_ms = pd.read_csv(main_summary_csv)
        for _, row in df_ms.iterrows():
            m = row["model_name"]
            c = row["corruption"]
            r1 = float(row["R_hat_s1"])
            if ("main", m) in model_metrics_map:
                mets, _ = model_metrics_map[("main", m)]
                cond_records.append({
                    "split": "main",
                    "model_name": m,
                    "corruption": c,
                    "R_hat_1": r1,
                    "VRS": mets["VRS"],
                    "Blank_Drop": mets["Blank_Drop"],
                    "Image_Sensitivity": mets["Image_Sensitivity"],
                    "Visual_Benefit_Rate": mets["Visual_Benefit_Rate"],
                    "Visual_Harm_Rate": mets["Visual_Harm_Rate"],
                })

    if rep_summary_csv.exists():
        df_rs = pd.read_csv(rep_summary_csv)
        for _, row in df_rs.iterrows():
            m = row["model_name"]
            c = row["corruption"]
            r1 = float(row["R_hat_s1"])
            if ("replication", m) in model_metrics_map:
                mets, _ = model_metrics_map[("replication", m)]
                cond_records.append({
                    "split": "replication",
                    "model_name": m,
                    "corruption": c,
                    "R_hat_1": r1,
                    "VRS": mets["VRS"],
                    "Blank_Drop": mets["Blank_Drop"],
                    "Image_Sensitivity": mets["Image_Sensitivity"],
                    "Visual_Benefit_Rate": mets["Visual_Benefit_Rate"],
                    "Visual_Harm_Rate": mets["Visual_Harm_Rate"],
                })

    cond_df = pd.DataFrame(cond_records)

    # 4. Compute Spearman Correlations
    print("\n" + "=" * 80)
    print("SPEARMAN CORRELATIONS WITH DYNAMIC TERMINAL RELIANCE R_hat(1)")
    print("=" * 80)

    corr_cols = ["VRS", "Blank_Drop", "Image_Sensitivity", "Visual_Benefit_Rate", "Visual_Harm_Rate"]
    corr_results = []

    def evaluate_and_print_correlations(sub_df: pd.DataFrame, label: str):
        print(f"\n--- Spearman Rank Correlation: {label} (N={len(sub_df)}) ---")
        header = f"{'Metric':<22} {'Spearman rho':>14} {'p-value':>12}"
        print(header)
        print("-" * len(header))
        for col in corr_cols:
            if len(sub_df) < 3 or sub_df[col].nunique() <= 1 or sub_df["R_hat_1"].nunique() <= 1:
                r, p = float("nan"), float("nan")
            else:
                res = stats.spearmanr(sub_df[col], sub_df["R_hat_1"])
                r, p = float(res.statistic), float(res.pvalue)

            print(f"{col:<22} {r:>+14.4f} {p:>12.4f}")
            corr_results.append({
                "subset": label,
                "n_samples": len(sub_df),
                "metric": col,
                "spearman_rho": r,
                "p_value": p,
            })

    if not cond_df.empty:
        evaluate_and_print_correlations(cond_df[cond_df["split"] == "main"], "Main Benchmark (all 9 conditions)")
        evaluate_and_print_correlations(cond_df[cond_df["split"] == "replication"], "Replication Benchmark (all 6 conditions)")
        evaluate_and_print_correlations(cond_df, "Pooled Main + Replication (all 15 conditions)")

        # Per-corruption breakdowns across models
        for corr in ["swapblend", "gaussian", "patch"]:
            sub = cond_df[cond_df["corruption"] == corr]
            if len(sub) >= 3:
                evaluate_and_print_correlations(sub, f"Per-Corruption: {corr} across models & splits")

        corr_df = pd.DataFrame(corr_results)
        corr_csv = out_dir / "baseline_correlations.csv"
        corr_df.to_csv(corr_csv, index=False)
        print(f"\nSaved correlation summary: {corr_csv}")
    else:
        corr_df = pd.DataFrame()
        print("[WARN] Condition dataframe is empty; could not compute correlations.")

    return metrics_df, corr_df


def main():
    args = parse_args()
    run_baseline_metrics(
        main_raw_dir=Path(args.main_raw_dir),
        rep_raw_dir=Path(args.rep_raw_dir),
        out_dir=Path(args.out_dir),
        n_boot=args.n_boot,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
