"""Master Criterion Status by Null Baseline and Sample Size Table Generator.

POST HOC SENSITIVITY TABLE:
Generates the comprehensive summary table:
  "Criterion Status by Null and Sample"
Evaluating for each:
  model x corruption x sample {main, replication, pooled} x null {removed, black}
the pre-registered criteria:
  1. U(1) significance: Upper 95% bootstrap CI < 0 (negative terminal utility)
  2. R_hat(1) vs 0.50: Lower 95% bootstrap CI > 0.50 (terminal reliance threshold)
  3. Pre-registered criterion met: Both criteria 1 and 2 met simultaneously.

No thresholds are altered.
Outputs:
  - results/derived/criterion_status_by_null_and_sample.csv
  - tables/criterion_status_by_null_and_sample.tex
"""

import argparse
from pathlib import Path
import re
import sys
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd

from scripts.analyze_vlm_all import load_shards_grouped, compute_metrics_for_items
from src.metrics import reliance, normalized_reliance, utility


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Criterion Status by Null and Sample table"
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
        help="Directory to save output CSV",
    )
    parser.add_argument(
        "--table_dir",
        type=str,
        default="tables",
        help="Directory to save LaTeX table",
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


def evaluate_sample_null(
    items: List[Dict[str, Any]],
    S_grid: np.ndarray,
    null_kind: str,
    n_boot: int = 1000,
    seed: int = 42,
) -> Dict[str, Any]:
    N = len(items)
    core = compute_metrics_for_items(items, S_grid, null_kind=null_kind)

    acc_S = core["acc_S_matrix"]
    acc_null = core["acc_null_per_item"]
    jsd_mat = core["jsd_matrix"]

    u1_pt = float(core["U_S"][-1])
    rhat1_pt = float(core["R_hat_S"][-1]) if core["R_hat_S"] is not None else np.nan

    rng = np.random.default_rng(seed)
    boot_u1 = np.zeros(n_boot, dtype=float)
    boot_rhat1 = []

    for b in range(n_boot):
        idx = rng.integers(0, N, size=N)
        b_u1 = np.mean(acc_S[-1, idx]) - np.mean(acc_null[idx])
        boot_u1[b] = b_u1

        b_D = np.mean(jsd_mat[:, idx], axis=1)
        b_R = reliance(b_D)
        try:
            b_R_hat = normalized_reliance(b_R)
            boot_rhat1.append(float(b_R_hat[-1]))
        except Exception:
            pass

    u1_ci = (
        float(np.percentile(boot_u1, 2.5)),
        float(np.percentile(boot_u1, 97.5)),
    )
    rhat1_ci = (
        float(np.percentile(boot_rhat1, 2.5)),
        float(np.percentile(boot_rhat1, 97.5)),
    ) if boot_rhat1 else (np.nan, np.nan)

    u1_sig = bool(u1_ci[1] < 0.0)

    if rhat1_ci[0] > 0.50:
        rhat_status = "Strictly > 0.50"
    elif rhat1_ci[1] < 0.50:
        rhat_status = "Strictly < 0.50"
    else:
        rhat_status = "Straddles 0.50"

    criterion_met = bool(u1_sig and rhat_status == "Strictly > 0.50")

    return {
        "clean_acc": core["clean_acc"],
        "null_acc": core["acc_null"],
        "U_1": u1_pt,
        "U_1_ci_lo": u1_ci[0],
        "U_1_ci_hi": u1_ci[1],
        "U_1_ci_str": f"[{u1_ci[0]:+.4f}, {u1_ci[1]:+.4f}]",
        "u1_significant": u1_sig,
        "R_hat_1": rhat1_pt,
        "R_hat_1_ci_lo": rhat1_ci[0],
        "R_hat_1_ci_hi": rhat1_ci[1],
        "R_hat_1_ci_str": f"[{rhat1_ci[0]:.4f}, {rhat1_ci[1]:.4f}]",
        "R_hat_1_status": rhat_status,
        "criterion_met": criterion_met,
    }


def build_criterion_status_table(
    main_raw_dir: Path,
    rep_raw_dir: Path,
    out_dir: Path,
    table_dir: Path,
    n_boot: int = 1000,
    seed: int = 42,
) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 105)
    print("CRITERION STATUS BY NULL AND SAMPLE (MAIN, REPLICATION, POOLED)")
    print(f"Bootstrap settings: B={n_boot}, seed={seed}")
    print("=" * 105)

    groups_main = load_shards_grouped(main_raw_dir, no_siblings=False)
    groups_rep = load_shards_grouped(rep_raw_dir, no_siblings=True)

    models_ordered = [
        "Qwen/Qwen2-VL-2B-Instruct",
        "HuggingFaceTB/SmolVLM-Instruct",
        "llava-hf/llava-1.5-7b-hf",
    ]

    records = []

    # 1. Main conditions
    for m in models_ordered:
        for c in ["swapblend", "gaussian", "patch"]:
            key = (m, c, 0)
            if key not in groups_main:
                continue
            items = groups_main[key]["items"]
            S_grid = groups_main[key]["S_grid"]
            for null_kind in ["removed", "black"]:
                res = evaluate_sample_null(items, S_grid, null_kind, n_boot=n_boot, seed=seed)
                records.append({
                    "model": m.split('/')[-1],
                    "corruption": c,
                    "sample": "main",
                    "n_items": len(items),
                    "null_baseline": null_kind,
                    **res,
                })

    # 2. Replication conditions
    for m in models_ordered:
        for c in ["swapblend", "gaussian"]:
            key = (m, c, 1)
            if key not in groups_rep:
                continue
            items = groups_rep[key]["items"]
            S_grid = groups_rep[key]["S_grid"]
            for null_kind in ["removed", "black"]:
                res = evaluate_sample_null(items, S_grid, null_kind, n_boot=n_boot, seed=seed)
                records.append({
                    "model": m.split('/')[-1],
                    "corruption": c,
                    "sample": "replication",
                    "n_items": len(items),
                    "null_baseline": null_kind,
                    **res,
                })

    # 3. Pooled conditions
    for m in models_ordered:
        for c in ["swapblend", "gaussian"]:
            key_m = (m, c, 0)
            key_r = (m, c, 1)
            if key_m not in groups_main or key_r not in groups_rep:
                continue
            items = groups_main[key_m]["items"] + groups_rep[key_r]["items"]
            S_grid = groups_main[key_m]["S_grid"]
            for null_kind in ["removed", "black"]:
                res = evaluate_sample_null(items, S_grid, null_kind, n_boot=n_boot, seed=seed)
                records.append({
                    "model": m.split('/')[-1],
                    "corruption": c,
                    "sample": "pooled",
                    "n_items": len(items),
                    "null_baseline": null_kind,
                    **res,
                })

    df = pd.DataFrame(records)

    # Console display
    header = (
        f"{'Model':<18} {'Corruption':<10} {'Sample':<12} {'Null':<8} "
        f"{'U(1)':>8} {'U(1) 95% CI':<24} {'U(1)<0':>7} "
        f"{'R_hat(1)':>9} {'R_hat 95% CI':<20} {'R_hat vs 0.5':<16} {'Met?':>5}"
    )
    print("\n" + header)
    print("-" * len(header))
    for _, r in df.iterrows():
        met_str = "YES" if r["criterion_met"] else "NO"
        sig_str = "YES" if r["u1_significant"] else "NO"
        print(
            f"{r['model']:<18} {r['corruption']:<10} {r['sample']:<12} {r['null_baseline']:<8} "
            f"{r['U_1']:>+8.4f} {r['U_1_ci_str']:<24} {sig_str:>7} "
            f"{r['R_hat_1']:>9.4f} {r['R_hat_1_ci_str']:<20} {r['R_hat_1_status']:<16} {met_str:>5}"
        )

    # Save CSV
    csv_path = out_dir / "criterion_status_by_null_and_sample.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved CSV: {csv_path}")

    # Save LaTeX table
    tex_path = table_dir / "criterion_status_by_null_and_sample.tex"
    tex_lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Pre-Declared Criterion Status by Null Baseline and Sample Size ($B=1000$, seed 42).}",
        r"\label{tab:criterion-status-by-null-and-sample}",
        r"\fittable{%",
        r"\begin{tabular}{llllcccc}",
        r"\toprule",
        r"\textbf{Model} & \textbf{Corruption} & \textbf{Sample} & \textbf{Null} & $U(1)$ & $U(1)<0$ & $\hat{R}(1)$ & \textbf{Met?} \\",
        r"\midrule",
    ]
    from scripts.generate_paper_tables import format_model_name
    for _, r in df.iterrows():
        m_disp = format_model_name(str(r["model"]))
        corr_disp = str(r["corruption"]).replace("_", r"\_")
        sample_disp = str(r["sample"]).replace("_", r"\_")
        null_disp = str(r["null_baseline"]).replace("_", r"\_")

        # Parse CI
        u1_m = re.search(r"\[([+-]?\d+\.?\d*),\s*([+-]?\d+\.?\d*)\]", str(r["U_1_ci_str"]))
        if u1_m:
            u1_lo, u1_hi = float(u1_m.group(1)), float(u1_m.group(2))
            u1_cell = f"{r['U_1']:+.3f} ({u1_lo:+.3f}, {u1_hi:+.3f})"
        else:
            u1_cell = f"{r['U_1']:+.3f}"

        u1_sig = "True" if r["u1_significant"] else "False"

        rhat_m = re.search(r"\[([+-]?\d+\.?\d*),\s*([+-]?\d+\.?\d*)\]", str(r["R_hat_1_ci_str"]))
        if rhat_m:
            rhat_lo, rhat_hi = float(rhat_m.group(1)), float(rhat_m.group(2))
            rhat_cell = f"{r['R_hat_1']:.3f} ({rhat_lo:.3f}, {rhat_hi:.3f})"
        else:
            rhat_cell = f"{r['R_hat_1']:.3f}"

        met_str = r"\textbf{True}" if r["criterion_met"] else "False"

        tex_lines.append(f"{m_disp} & {corr_disp} & {sample_disp} & {null_disp} & {u1_cell} & {u1_sig} & {rhat_cell} & {met_str} \\\\")

    tex_lines.extend([
        r"\bottomrule",
        r"\end{tabular}}",
        r"\end{table}",
    ])
    tex_path.write_text("\n".join(tex_lines) + "\n", encoding="utf-8")
    print(f"Saved LaTeX: {tex_path}")

    return df


def main():
    args = parse_args()
    build_criterion_status_table(
        main_raw_dir=Path(args.main_raw_dir),
        rep_raw_dir=Path(args.rep_raw_dir),
        out_dir=Path(args.out_dir),
        table_dir=Path(args.table_dir),
        n_boot=args.n_boot,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
