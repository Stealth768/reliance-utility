"""Post Hoc Black-Null Recomputation and Comparison Analysis.

POST HOC ANALYSIS:
Recomputes reliance-utility misalignment metrics on existing VLM shards using
the black null baseline (B_black, all-zero visual input) instead of the
primary removed null (B_removed, text-only prompt).

Evaluates across all models and corruptions in:
  - Main benchmark (N=500, seed 0): 3 models x 3 corruptions = 9 conditions
  - Replication benchmark (N=445, seed 1): 3 models x 2 corruptions = 6 conditions

Metrics computed under both null baselines:
  - U(S), R(S), R_hat(S), S*, significant crossing
  - U(1) point estimate and 95% bootstrap CI (B=1000, seed 42)
  - R_hat(1) point estimate and 95% bootstrap CI (B=1000, seed 42)

Side-by-side comparison flags:
  1. Any change in the sign of terminal utility U(1).
  2. Any change in whether the 95% CI of R_hat(1) contains the 0.50 threshold.
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
        description="Post hoc black-null recomputation and sensitivity comparison"
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
        help="Directory to save output comparison CSV",
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


def evaluate_condition_under_null(
    items: List[Dict[str, Any]],
    S_grid: np.ndarray,
    null_kind: str,
    n_boot: int = 1000,
    seed: int = 42,
) -> Dict[str, Any]:
    """Compute U(S), R(S), R_hat(S), S*, significant crossing, U(1) CI, R_hat(1) CI."""
    core = compute_metrics_for_items(items, S_grid, null_kind=null_kind)
    n_items = len(items)
    n_S = len(S_grid)

    acc_S_matrix = core["acc_S_matrix"]
    acc_null_per_item = core["acc_null_per_item"]
    jsd_matrix = core["jsd_matrix"]

    u1_point = float(core["U_S"][-1])
    rhat1_point = float(core["R_hat_S"][-1]) if core["R_hat_S"] is not None else np.nan
    s_star = core["S_star_raw"]

    # Bootstrap
    rng = np.random.default_rng(seed)
    boot_u_matrix = np.zeros((n_boot, n_S), dtype=float)
    boot_rhat1 = []

    for b in range(n_boot):
        idx = rng.integers(0, n_items, size=n_items)
        b_acc_deg = np.mean(acc_S_matrix[:, idx], axis=1)
        b_acc_null = float(np.mean(acc_null_per_item[idx]))
        b_u = utility(b_acc_deg, b_acc_null)
        boot_u_matrix[b] = b_u

        b_D = np.mean(jsd_matrix[:, idx], axis=1)
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

    return {
        "null_kind": null_kind,
        "null_acc": core["acc_null"],
        "S_star": s_star,
        "sig_S_star": sig_s_star,
        "U_1": u1_point,
        "U_1_ci_lo": u1_ci[0],
        "U_1_ci_hi": u1_ci[1],
        "R_hat_1": rhat1_point,
        "R_hat_1_ci_lo": rhat1_ci[0],
        "R_hat_1_ci_hi": rhat1_ci[1],
        "U_S": core["U_S"],
        "R_S": core["R_S"],
        "R_hat_S": core["R_hat_S"],
    }


def run_black_null_comparison(
    main_raw_dir: Path,
    rep_raw_dir: Path,
    out_dir: Path,
    n_boot: int = 1000,
    seed: int = 42,
) -> pd.DataFrame:
    """Run full black vs removed null sensitivity evaluation."""
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 100)
    print("POST HOC BLACK-NULL RECOMPUTATION & SIDE-BY-SIDE SENSITIVITY COMPARISON")
    print(f"Bootstrap settings: B={n_boot}, seed={seed}")
    print("=" * 100)

    groups_main = load_shards_grouped(main_raw_dir, no_siblings=False)
    groups_rep = load_shards_grouped(rep_raw_dir, no_siblings=True)

    models_ordered = [
        "Qwen/Qwen2-VL-2B-Instruct",
        "HuggingFaceTB/SmolVLM-Instruct",
        "llava-hf/llava-1.5-7b-hf",
    ]

    records = []

    for split_name, groups, split_seed in [("main", groups_main, 0), ("replication", groups_rep, 1)]:
        print(f"\n{'='*40} SPLIT: {split_name.upper()} (seed={split_seed}) {'='*40}")
        for m in models_ordered:
            for c in ["swapblend", "gaussian", "patch"]:
                key = (m, c, split_seed)
                if key not in groups:
                    continue

                items = groups[key]["items"]
                S_grid = groups[key]["S_grid"]
                n_items = len(items)

                res_rem = evaluate_condition_under_null(items, S_grid, "removed", n_boot=n_boot, seed=seed)
                res_blk = evaluate_condition_under_null(items, S_grid, "black", n_boot=n_boot, seed=seed)

                # Flag checks
                # 1. Sign change in U(1)
                sign_rem = np.sign(res_rem["U_1"])
                sign_blk = np.sign(res_blk["U_1"])
                u1_sign_flip = bool(sign_rem != sign_blk and sign_rem != 0 and sign_blk != 0)

                # 2. R_hat(1) CI contains 0.5
                ci_rem_has_half = bool(res_rem["R_hat_1_ci_lo"] <= 0.5 <= res_rem["R_hat_1_ci_hi"])
                ci_blk_has_half = bool(res_blk["R_hat_1_ci_lo"] <= 0.5 <= res_blk["R_hat_1_ci_hi"])
                ci_half_change = bool(ci_rem_has_half != ci_blk_has_half)

                flags = []
                if u1_sign_flip:
                    flags.append(f"U(1) SIGN FLIP ({res_rem['U_1']:+.4f} -> {res_blk['U_1']:+.4f})")
                if ci_half_change:
                    flags.append(f"R_hat(1) CI 0.5 STATUS CHANGED (rem={ci_rem_has_half} -> blk={ci_blk_has_half})")

                flag_str = " | ".join(flags) if flags else "None"

                m_short = m.split('/')[-1]
                print(f"\n>>> {m_short} | {c} (N={n_items})")
                print(f"  REMOVED NULL : null_acc={res_rem['null_acc']:.4f} | S*={res_rem['S_star']} | sig_S*={res_rem['sig_S_star']}")
                print(f"                 U(1)={res_rem['U_1']:+.4f} [{res_rem['U_1_ci_lo']:+.4f}, {res_rem['U_1_ci_hi']:+.4f}]")
                print(f"                 R_hat(1)={res_rem['R_hat_1']:.4f} [{res_rem['R_hat_1_ci_lo']:.4f}, {res_rem['R_hat_1_ci_hi']:.4f}] (CI contains 0.5: {ci_rem_has_half})")
                print(f"  BLACK NULL   : null_acc={res_blk['null_acc']:.4f} | S*={res_blk['S_star']} | sig_S*={res_blk['sig_S_star']}")
                print(f"                 U(1)={res_blk['U_1']:+.4f} [{res_blk['U_1_ci_lo']:+.4f}, {res_blk['U_1_ci_hi']:+.4f}]")
                print(f"                 R_hat(1)={res_blk['R_hat_1']:.4f} [{res_blk['R_hat_1_ci_lo']:.4f}, {res_blk['R_hat_1_ci_hi']:.4f}] (CI contains 0.5: {ci_blk_has_half})")
                if flags:
                    print(f"  *** FLAG     : {flag_str} ***")

                records.append({
                    "split": split_name,
                    "model_name": m,
                    "corruption": c,
                    "n_items": n_items,
                    # Removed null
                    "rem_null_acc": res_rem["null_acc"],
                    "rem_S_star": res_rem["S_star"],
                    "rem_sig_S_star": res_rem["sig_S_star"],
                    "rem_U_1": res_rem["U_1"],
                    "rem_U_1_ci_lo": res_rem["U_1_ci_lo"],
                    "rem_U_1_ci_hi": res_rem["U_1_ci_hi"],
                    "rem_R_hat_1": res_rem["R_hat_1"],
                    "rem_R_hat_1_ci_lo": res_rem["R_hat_1_ci_lo"],
                    "rem_R_hat_1_ci_hi": res_rem["R_hat_1_ci_hi"],
                    "rem_ci_contains_half": ci_rem_has_half,
                    # Black null
                    "blk_null_acc": res_blk["null_acc"],
                    "blk_S_star": res_blk["S_star"],
                    "blk_sig_S_star": res_blk["sig_S_star"],
                    "blk_U_1": res_blk["U_1"],
                    "blk_U_1_ci_lo": res_blk["U_1_ci_lo"],
                    "blk_U_1_ci_hi": res_blk["U_1_ci_hi"],
                    "blk_R_hat_1": res_blk["R_hat_1"],
                    "blk_R_hat_1_ci_lo": res_blk["R_hat_1_ci_lo"],
                    "blk_R_hat_1_ci_hi": res_blk["R_hat_1_ci_hi"],
                    "blk_ci_contains_half": ci_blk_has_half,
                    # Flags
                    "flag_u1_sign_flip": u1_sign_flip,
                    "flag_ci_half_change": ci_half_change,
                    "flags_summary": flag_str,
                })

    df = pd.DataFrame(records)
    csv_path = out_dir / "black_null_comparison.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved black null comparison: {csv_path}")

    # Summary table of flags
    flags_df = df[df["flag_u1_sign_flip"] | df["flag_ci_half_change"]]
    print("\n" + "=" * 100)
    print("SUMMARY OF SENSITIVITY FLAGS (REMOVED VS BLACK NULL)")
    print("=" * 100)
    if not flags_df.empty:
        for _, row in flags_df.iterrows():
            m_s = row["model_name"].split('/')[-1]
            print(f"- [{row['split'].upper()}] {m_s} | {row['corruption']}: {row['flags_summary']}")
    else:
        print("No conditions exhibited a U(1) sign flip or R_hat(1) CI 0.5 status change.")

    return df


def main():
    args = parse_args()
    run_black_null_comparison(
        main_raw_dir=Path(args.main_raw_dir),
        rep_raw_dir=Path(args.rep_raw_dir),
        out_dir=Path(args.out_dir),
        n_boot=args.n_boot,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
