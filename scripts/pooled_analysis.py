"""Secondary Pooled Main + Replication Analysis (N=945).

SECONDARY ANALYSIS:
This script pools the main benchmark sample (N=500, seed 0) and held-out
replication sample (N=445, seed 1) into a single N=945 evaluation set per
model across shared corruptions (swapblend and gaussian).

Pre-registration status: Secondary / exploratory synthesis.
Outputs are stored in a dedicated folder (results/derived/pooled/) and
kept strictly decoupled from primary pre-registered tables.

Metrics computed with 95% bootstrap CIs (B=1000, seed 42):
  - Clean accuracy and unimodal removed null accuracy
  - Terminal utility U(1) = acc(S=1) - acc(null) with 95% bootstrap CI
  - Terminal normalized reliance R_hat(1) with 95% bootstrap CI
  - Empirical utility crossing S* and significant crossing S*_sig
"""

import argparse
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple

# Ensure repo root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd

from scripts.analyze_vlm_all import load_shards_grouped, compute_metrics_for_items
from src.metrics import reliance, normalized_reliance, utility, utility_crossing


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Secondary pooled main + replication analysis (N=945)"
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
        default="results/derived/pooled",
        help="Dedicated separate output directory for pooled analysis",
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


def run_pooled_analysis(
    main_raw_dir: Path,
    rep_raw_dir: Path,
    out_dir: Path,
    n_boot: int = 1000,
    seed: int = 42,
) -> pd.DataFrame:
    """Run secondary pooled analysis (N=945) across models and shared corruptions."""
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 90)
    print("SECONDARY POOLED MAIN + REPLICATION ANALYSIS (N=945)")
    print(f"Bootstrap settings: B={n_boot}, seed={seed}")
    print(f"Output directory: {out_dir} (strictly decoupled from primary tables)")
    print("=" * 90)

    groups_main = load_shards_grouped(main_raw_dir, no_siblings=False)
    groups_rep = load_shards_grouped(rep_raw_dir, no_siblings=True)

    models_ordered = [
        "Qwen/Qwen2-VL-2B-Instruct",
        "HuggingFaceTB/SmolVLM-Instruct",
        "llava-hf/llava-1.5-7b-hf",
    ]
    shared_corruptions = ["swapblend", "gaussian"]

    records = []

    for m in models_ordered:
        m_short = m.split('/')[-1]
        print(f"\nModel: {m_short}")
        for c in shared_corruptions:
            key_main = (m, c, 0)
            key_rep = (m, c, 1)

            if key_main not in groups_main or key_rep not in groups_rep:
                print(f"[WARN] Missing shards for {m} | {c}; skipping.")
                continue

            items_main = groups_main[key_main]["items"]
            items_rep = groups_rep[key_rep]["items"]
            items_pooled = items_main + items_rep
            N = len(items_pooled)
            S_grid = groups_main[key_main]["S_grid"]
            n_S = len(S_grid)

            core = compute_metrics_for_items(items_pooled, S_grid, null_kind="removed")
            u1_pt = float(core["U_S"][-1])
            rhat1_pt = float(core["R_hat_S"][-1])
            s_star = core["S_star_raw"]

            acc_S = core["acc_S_matrix"]
            acc_null = core["acc_null_per_item"]
            jsd_mat = core["jsd_matrix"]

            # Bootstrap CIs (B=1000, seed=42)
            rng = np.random.default_rng(seed)
            boot_u_matrix = np.zeros((n_boot, n_S), dtype=float)
            boot_rhat1 = []

            for b in range(n_boot):
                idx = rng.integers(0, N, size=N)
                b_acc_deg = np.mean(acc_S[:, idx], axis=1)
                b_acc_null = float(np.mean(acc_null[idx]))
                b_u = utility(b_acc_deg, b_acc_null)
                boot_u_matrix[b] = b_u

                b_D = np.mean(jsd_mat[:, idx], axis=1)
                b_R = reliance(b_D)
                try:
                    b_R_hat = normalized_reliance(b_R)
                    boot_rhat1.append(float(b_R_hat[-1]))
                except Exception:
                    pass

            u1_ci = (
                float(np.percentile(boot_u_matrix[:, -1], 2.5)),
                float(np.percentile(boot_u_matrix[:, -1], 97.5)),
            )

            if len(boot_rhat1) > 0:
                rhat1_ci = (
                    float(np.percentile(boot_rhat1, 2.5)),
                    float(np.percentile(boot_rhat1, 97.5)),
                )
            else:
                rhat1_ci = (np.nan, np.nan)

            # Significant crossing: first S where upper bootstrap CI of U(S) < 0
            u_ci_upper = np.percentile(boot_u_matrix, 97.5, axis=0)
            sig_cross_idx = np.where(u_ci_upper < 0)[0]
            sig_s_star = float(S_grid[sig_cross_idx[0]]) if len(sig_cross_idx) > 0 else None

            u1_sig_drop = bool(u1_ci[1] < 0.0)
            rhat1_above_half = bool(rhat1_ci[0] > 0.50)
            rhat1_straddles_half = bool(rhat1_ci[0] <= 0.50 <= rhat1_ci[1])

            print(f"  Corruption: {c:<10} (N={N})")
            print(f"    Clean Acc : {core['clean_acc']:.4f} | Null Acc (removed): {core['acc_null']:.4f}")
            print(f"    S*        : {s_star} | S*_sig: {sig_s_star}")
            print(f"    U(1)      : {u1_pt:+.4f}  95% CI [{u1_ci[0]:+.4f}, {u1_ci[1]:+.4f}] (Upper < 0: {u1_sig_drop})")
            print(f"    R_hat(1)  : {rhat1_pt:.4f}  95% CI [{rhat1_ci[0]:.4f}, {rhat1_ci[1]:.4f}] (Lower > 0.5: {rhat1_above_half}, Straddles 0.5: {rhat1_straddles_half})")

            records.append({
                "model_name": m,
                "corruption": c,
                "n_samples": N,
                "clean_acc": core["clean_acc"],
                "null_removed_acc": core["acc_null"],
                "S_star": s_star,
                "sig_S_star": sig_s_star,
                "U_1": u1_pt,
                "U_1_ci_lo": u1_ci[0],
                "U_1_ci_hi": u1_ci[1],
                "U_1_upper_below_zero": u1_sig_drop,
                "R_hat_1": rhat1_pt,
                "R_hat_1_ci_lo": rhat1_ci[0],
                "R_hat_1_ci_hi": rhat1_ci[1],
                "R_hat_1_lower_above_half": rhat1_above_half,
                "R_hat_1_straddles_half": rhat1_straddles_half,
            })

    df = pd.DataFrame(records)
    csv_path = out_dir / "vlm_pooled_summary.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved secondary pooled summary: {csv_path}")

    # Also output a clean separate LaTeX table inside the separate output folder
    tex_path = out_dir / "vlm_pooled_summary.tex"
    tex_lines = [
        "% Secondary synthesis: Pooled Main + Replication (N=945)",
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Secondary Synthesis: Pooled Main and Replication Performance ($N=945$).}",
        "\\label{tab:vlm_pooled_summary}",
        "\\small",
        df[["model_name", "corruption", "n_samples", "clean_acc", "null_removed_acc", "U_1", "R_hat_1"]].to_latex(index=False),
        "\\end{table}",
    ]
    tex_path.write_text("\n".join(tex_lines), encoding="utf-8")
    print(f"Saved separate pooled LaTeX table: {tex_path}")

    return df


def main():
    args = parse_args()
    run_pooled_analysis(
        main_raw_dir=Path(args.main_raw_dir),
        rep_raw_dir=Path(args.rep_raw_dir),
        out_dir=Path(args.out_dir),
        n_boot=args.n_boot,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
