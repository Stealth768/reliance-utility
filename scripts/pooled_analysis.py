"""Secondary Pooled Main + Replication Analysis (N=945) under Removed and Black Nulls.

POST HOC / SECONDARY ANALYSIS:
This script pools the main benchmark sample (N=500, seed 0) and held-out
replication sample (N=445, seed 1) into a single N=945 evaluation set per
model across shared corruptions (swapblend and gaussian).

Pre-registration status: Secondary / exploratory post hoc synthesis.
Outputs are stored in a dedicated folder (results/derived/pooled/) and
kept strictly decoupled from primary pre-registered tables.

Evaluates under both:
  1. Primary removed null (B_removed, text-only prompt)
  2. Black null (B_black, all-zero visual input)

Metrics computed with 95% bootstrap CIs (B=1000, seed 42):
  - Clean accuracy and unimodal null accuracy
  - Terminal utility U(1) = acc(S=1) - acc(null) with 95% bootstrap CI
  - Terminal normalized reliance R_hat(1) with 95% bootstrap CI
  - Empirical utility crossing S* and significant crossing S*_sig

Side-by-side comparison flags:
  - U(1) sign flip
  - R_hat(1) CI 0.5 status change (contains 0.5 vs strictly > 0.5)
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
        description="Secondary pooled main + replication analysis (N=945) under removed and black nulls"
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


def evaluate_pooled_condition(
    items: List[Dict[str, Any]],
    S_grid: np.ndarray,
    null_kind: str,
    n_boot: int = 1000,
    seed: int = 42,
) -> Dict[str, Any]:
    """Compute pooled metrics under given null baseline with bootstrap CIs."""
    N = len(items)
    n_S = len(S_grid)
    core = compute_metrics_for_items(items, S_grid, null_kind=null_kind)

    acc_S = core["acc_S_matrix"]
    acc_null = core["acc_null_per_item"]
    jsd_mat = core["jsd_matrix"]

    u1_pt = float(core["U_S"][-1])
    rhat1_pt = float(core["R_hat_S"][-1]) if core["R_hat_S"] is not None else np.nan
    s_star = core["S_star_raw"]

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

    # Significant crossing
    u_ci_upper = np.percentile(boot_u_matrix, 97.5, axis=0)
    sig_cross_idx = np.where(u_ci_upper < 0)[0]
    sig_s_star = float(S_grid[sig_cross_idx[0]]) if len(sig_cross_idx) > 0 else None

    return {
        "null_kind": null_kind,
        "clean_acc": core["clean_acc"],
        "null_acc": core["acc_null"],
        "S_star": s_star,
        "sig_S_star": sig_s_star,
        "U_1": u1_pt,
        "U_1_ci_lo": u1_ci[0],
        "U_1_ci_hi": u1_ci[1],
        "R_hat_1": rhat1_pt,
        "R_hat_1_ci_lo": rhat1_ci[0],
        "R_hat_1_ci_hi": rhat1_ci[1],
        "U_S": core["U_S"],
        "R_S": core["R_S"],
        "R_hat_S": core["R_hat_S"],
    }


def run_pooled_analysis(
    main_raw_dir: Path,
    rep_raw_dir: Path,
    out_dir: Path,
    n_boot: int = 1000,
    seed: int = 42,
) -> pd.DataFrame:
    """Run secondary pooled analysis (N=945) across models under removed and black nulls."""
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 100)
    print("POST HOC SECONDARY POOLED ANALYSIS (N=945): REMOVED VS BLACK NULL")
    print(f"Bootstrap settings: B={n_boot}, seed={seed}")
    print(f"Output directory: {out_dir} (strictly decoupled from primary tables)")
    print("=" * 100)

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
        print(f"\n{'='*30} Model: {m_short} {'='*30}")
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

            rem = evaluate_pooled_condition(items_pooled, S_grid, "removed", n_boot=n_boot, seed=seed)
            blk = evaluate_pooled_condition(items_pooled, S_grid, "black", n_boot=n_boot, seed=seed)

            # Flags
            sign_rem = np.sign(rem["U_1"])
            sign_blk = np.sign(blk["U_1"])
            u1_sign_flip = bool(sign_rem != sign_blk and sign_rem != 0 and sign_blk != 0)

            ci_rem_has_half = bool(rem["R_hat_1_ci_lo"] <= 0.5 <= rem["R_hat_1_ci_hi"])
            ci_blk_has_half = bool(blk["R_hat_1_ci_lo"] <= 0.5 <= blk["R_hat_1_ci_hi"])
            ci_half_change = bool(ci_rem_has_half != ci_blk_has_half)

            flags = []
            if u1_sign_flip:
                flags.append(f"U(1) SIGN FLIP ({rem['U_1']:+.4f} -> {blk['U_1']:+.4f})")
            if ci_half_change:
                flags.append(f"R_hat(1) CI 0.5 STATUS CHANGED (rem={ci_rem_has_half} -> blk={ci_blk_has_half})")

            flag_str = " | ".join(flags) if flags else "None"

            print(f"\n>>> Corruption: {c:<10} (N={N})")
            print(f"  REMOVED NULL : null_acc={rem['null_acc']:.4f} | S*={rem['S_star']} | sig_S*={rem['sig_S_star']}")
            print(f"                 U(1)={rem['U_1']:+.4f} [{rem['U_1_ci_lo']:+.4f}, {rem['U_1_ci_hi']:+.4f}] (Upper < 0: {rem['U_1_ci_hi'] < 0})")
            print(f"                 R_hat(1)={rem['R_hat_1']:.4f} [{rem['R_hat_1_ci_lo']:.4f}, {rem['R_hat_1_ci_hi']:.4f}] (Contains 0.5: {ci_rem_has_half})")
            print(f"  BLACK NULL   : null_acc={blk['null_acc']:.4f} | S*={blk['S_star']} | sig_S*={blk['sig_S_star']}")
            print(f"                 U(1)={blk['U_1']:+.4f} [{blk['U_1_ci_lo']:+.4f}, {blk['U_1_ci_hi']:+.4f}] (Upper < 0: {blk['U_1_ci_hi'] < 0})")
            print(f"                 R_hat(1)={blk['R_hat_1']:.4f} [{blk['R_hat_1_ci_lo']:.4f}, {blk['R_hat_1_ci_hi']:.4f}] (Contains 0.5: {ci_blk_has_half})")
            if flags:
                print(f"  *** FLAG     : {flag_str} ***")

            # Store removed record
            records.append({
                "model_name": m,
                "corruption": c,
                "null_baseline": "removed",
                "n_samples": N,
                "clean_acc": rem["clean_acc"],
                "null_acc": rem["null_acc"],
                "S_star": rem["S_star"],
                "sig_S_star": rem["sig_S_star"],
                "U_1": rem["U_1"],
                "U_1_ci_lo": rem["U_1_ci_lo"],
                "U_1_ci_hi": rem["U_1_ci_hi"],
                "U_1_upper_below_zero": bool(rem["U_1_ci_hi"] < 0),
                "R_hat_1": rem["R_hat_1"],
                "R_hat_1_ci_lo": rem["R_hat_1_ci_lo"],
                "R_hat_1_ci_hi": rem["R_hat_1_ci_hi"],
                "R_hat_1_lower_above_half": bool(rem["R_hat_1_ci_lo"] > 0.50),
                "R_hat_1_contains_half": ci_rem_has_half,
                "flag_u1_sign_flip": u1_sign_flip,
                "flag_ci_half_change": ci_half_change,
            })

            # Store black record
            records.append({
                "model_name": m,
                "corruption": c,
                "null_baseline": "black",
                "n_samples": N,
                "clean_acc": blk["clean_acc"],
                "null_acc": blk["null_acc"],
                "S_star": blk["S_star"],
                "sig_S_star": blk["sig_S_star"],
                "U_1": blk["U_1"],
                "U_1_ci_lo": blk["U_1_ci_lo"],
                "U_1_ci_hi": blk["U_1_ci_hi"],
                "U_1_upper_below_zero": bool(blk["U_1_ci_hi"] < 0),
                "R_hat_1": blk["R_hat_1"],
                "R_hat_1_ci_lo": blk["R_hat_1_ci_lo"],
                "R_hat_1_ci_hi": blk["R_hat_1_ci_hi"],
                "R_hat_1_lower_above_half": bool(blk["R_hat_1_ci_lo"] > 0.50),
                "R_hat_1_contains_half": ci_blk_has_half,
                "flag_u1_sign_flip": u1_sign_flip,
                "flag_ci_half_change": ci_half_change,
            })

    df = pd.DataFrame(records)
    csv_path = out_dir / "vlm_pooled_summary.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved secondary pooled summary: {csv_path}")

    # LaTeX summary in separate folder
    tex_path = out_dir / "vlm_pooled_summary.tex"
    tex_lines = [
        "% Secondary synthesis: Pooled Main + Replication (N=945) under Removed and Black Nulls",
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Secondary Synthesis: Pooled Main and Replication Performance ($N=945$) across Null Baselines.}",
        "\\label{tab:vlm_pooled_summary}",
        "\\small",
        df[["model_name", "corruption", "null_baseline", "clean_acc", "null_acc", "U_1", "R_hat_1"]].to_latex(index=False),
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
