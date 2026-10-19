"""Analysis and visualization of Camera-LiDAR BEV fusion experiments."""

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Tuple

# Ensure repo root on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd

from src.metrics import (
    jsd,
    misalignment_index,
    normalized_reliance,
    reliance,
    utility,
    utility_crossing,
)

CB_COLORS = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#D55E00", "#56B4E9"]


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze BEV fusion experiments")
    parser.add_argument(
        "--raw_dir",
        type=str,
        default="results/raw/bev",
        help="Directory with raw prediction files",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="results/derived",
        help="Directory for derived CSV summaries",
    )
    parser.add_argument(
        "--table_dir",
        type=str,
        default="tables",
        help="Directory for LaTeX tables",
    )
    parser.add_argument(
        "--fig_dir",
        type=str,
        default="figures",
        help="Directory for vector PDF figures",
    )
    parser.add_argument(
        "--n_boot",
        type=int,
        default=1000,
        help="Bootstrap resamples count",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    return parser.parse_args()


def load_bev_predictions(raw_dir: Path) -> Dict[Tuple[str, str, float], List[Dict[str, Any]]]:
    """Load predictions grouped by (model, corruption, dropout_p)."""
    groups: Dict[Tuple[str, str, float], List[Dict[str, Any]]] = {}
    for pred_file in sorted(raw_dir.glob("**/predictions.json")):
        try:
            with open(pred_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            m = data["model"]
            c = data["corruption"]
            p = float(data["dropout_p"])
            groups.setdefault((m, c, p), []).append(data)
        except Exception as e:
            print(f"[WARN] Failed to load {pred_file}: {e}")
    return groups


def analyze_condition(
    model_name: str,
    corruption: str,
    dropout_p: float,
    seed_runs: List[Dict[str, Any]],
    n_boot: int = 1000,
    seed: int = 42,
) -> Dict[str, Any]:
    """Analyze one (model, corruption, p) condition across seeds."""
    S_grid = np.array(seed_runs[0]["S_grid"])
    n_S = len(S_grid)

    all_items = []
    seed_item_lists = []
    for run in seed_runs:
        all_items.extend(run["items"])
        seed_item_lists.append(run["items"])

    n_items_total = len(all_items)
    correct_idx = np.array([it["correct_choice_idx"] for it in all_items])
    null_probs = np.array([it["null_probs"]["removed"] for it in all_items])

    null_preds = np.argmax(null_probs, axis=-1)
    acc_null = float(np.mean(null_preds == correct_idx))

    probs_S = np.array([[it["probs_per_S"][s] for it in all_items] for s in range(n_S)])
    preds_S = np.argmax(probs_S, axis=-1)
    acc_deg = np.mean(preds_S == correct_idx[None, :], axis=1)

    # JSD per item
    jsd_matrix = np.zeros((n_S, n_items_total), dtype=float)
    for s in range(n_S):
        ps = probs_S[s]
        pn = null_probs
        m = 0.5 * (ps + pn)
        kl_p = np.sum(ps * np.log2(np.maximum(ps, 1e-12) / np.maximum(m, 1e-12)), axis=-1)
        kl_n = np.sum(pn * np.log2(np.maximum(pn, 1e-12) / np.maximum(m, 1e-12)), axis=-1)
        jsd_matrix[s] = np.clip(0.5 * (kl_p + kl_n), 0.0, 1.0)

    D_S = np.mean(jsd_matrix, axis=1)
    U_S = utility(acc_deg, acc_null)
    R_S = reliance(D_S)
    try:
        R_hat_S = normalized_reliance(R_S)
        S_star = utility_crossing(U_S, S_grid, window=3)
        MI = misalignment_index(R_hat_S, S_grid, S_star)
    except Exception:
        R_hat_S = np.zeros_like(R_S)
        S_star = None
        MI = None

    # Hierarchical bootstrap (seeds then items)
    rng = np.random.default_rng(seed)
    n_seeds = len(seed_runs)
    boot_mi = []
    boot_r1 = []

    for b in range(n_boot):
        b_seed_idx = rng.integers(0, n_seeds, size=n_seeds)
        b_items = []
        for s_idx in b_seed_idx:
            sub = seed_item_lists[s_idx]
            sub_idx = rng.integers(0, len(sub), size=len(sub))
            for i in sub_idx:
                b_items.append(sub[i])

        b_corr = np.array([it["correct_choice_idx"] for it in b_items])
        b_null = np.array([it["null_probs"]["removed"] for it in b_items])
        b_acc_null = float(np.mean(np.argmax(b_null, axis=-1) == b_corr))

        b_probs_S = np.array([[it["probs_per_S"][s] for it in b_items] for s in range(n_S)])
        b_preds_S = np.argmax(b_probs_S, axis=-1)
        b_acc_deg = np.mean(b_preds_S == b_corr[None, :], axis=1)

        b_u = utility(b_acc_deg, b_acc_null)
        b_s = utility_crossing(b_u, S_grid, window=3)

        b_jsd = np.zeros((n_S, len(b_items)), dtype=float)
        for s in range(n_S):
            ps = b_probs_S[s]
            pn = b_null
            m = 0.5 * (ps + pn)
            kl_p = np.sum(ps * np.log2(np.maximum(ps, 1e-12) / np.maximum(m, 1e-12)), axis=-1)
            kl_n = np.sum(pn * np.log2(np.maximum(pn, 1e-12) / np.maximum(m, 1e-12)), axis=-1)
            b_jsd[s] = np.clip(0.5 * (kl_p + kl_n), 0.0, 1.0)
        b_D = np.mean(b_jsd, axis=1)
        b_R = reliance(b_D)
        try:
            b_R_hat = normalized_reliance(b_R)
            boot_r1.append(float(b_R_hat[-1]))
            if b_s is not None:
                val_mi = misalignment_index(b_R_hat, S_grid, b_s)
                boot_mi.append(val_mi if val_mi is not None else np.nan)
            else:
                boot_mi.append(np.nan)
        except Exception:
            boot_r1.append(np.nan)
            boot_mi.append(np.nan)

    valid_mi = [m for m in boot_mi if not np.isnan(m)]
    if len(valid_mi) > 0:
        mi_ci_low, mi_ci_high = float(np.percentile(valid_mi, 2.5)), float(np.percentile(valid_mi, 97.5))
    else:
        mi_ci_low, mi_ci_high = np.nan, np.nan

    valid_r1 = [r for r in boot_r1 if not np.isnan(r)]
    if len(valid_r1) > 0:
        r1_ci_low, r1_ci_high = float(np.percentile(valid_r1, 2.5)), float(np.percentile(valid_r1, 97.5))
    else:
        r1_ci_low, r1_ci_high = np.nan, np.nan

    return {
        "model": model_name,
        "corruption": corruption,
        "dropout_p": dropout_p,
        "n_seeds": n_seeds,
        "clean_acc": float(acc_deg[0]),
        "acc_null_removed": acc_null,
        "R_hat_1": float(R_hat_S[-1]) if len(R_hat_S) > 0 else np.nan,
        "R_hat_1_ci_low": r1_ci_low,
        "R_hat_1_ci_high": r1_ci_high,
        "S_star": S_star,
        "MI": MI,
        "MI_ci_low": mi_ci_low,
        "MI_ci_high": mi_ci_high,
    }


def run_full_bev_analysis(
    raw_dir: Path,
    out_dir: Path,
    table_dir: Path,
    fig_dir: Path,
    n_boot: int = 1000,
    seed: int = 42,
):
    """Run end-to-end analysis on BEV outputs."""
    out_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    grouped = load_bev_predictions(raw_dir)
    if not grouped:
        print(f"[WARN] No BEV predictions found in {raw_dir}.")
        return

    records = []
    for (m, c, p), runs in sorted(grouped.items()):
        print(f"Analyzing BEV: {m} | {c} | p={p} across {len(runs)} seeds...")
        res = analyze_condition(m, c, p, runs, n_boot=n_boot, seed=seed)
        records.append(res)

    df = pd.DataFrame(records)
    csv_path = out_dir / "bev_summary.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")

    # LaTeX Table
    tex_path = table_dir / "bev_summary.tex"
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Camera-LiDAR BEV Sensor Fusion: Reliance-Utility Metrics Under Camera Degradation.}",
        "\\label{tab:bev_summary}",
        "\\small",
        df.to_latex(index=False, escape=False).strip(),
        "\\end{table}",
    ]
    tex_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved: {tex_path}")

    # Plot figure if matplotlib available
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7, 4.5))
        for idx, (m, c) in enumerate(df[["model", "corruption"]].drop_duplicates().values):
            sub = df[(df["model"] == m) & (df["corruption"] == c)].sort_values("dropout_p")
            color = CB_COLORS[idx % len(CB_COLORS)]
            marker = "o" if m == "fixed" else "s"
            linestyle = "-" if m == "fixed" else "--"
            label = f"{m.capitalize()} ({c})"
            ax.plot(sub["dropout_p"], sub["R_hat_1"], marker=marker, linestyle=linestyle, color=color, label=label, linewidth=1.8)

        ax.set_xlabel("Camera Dropout Probability $p$", fontsize=11)
        ax.set_ylabel(r"Terminal Normalized Reliance $\hat{R}(1)$", fontsize=11)
        ax.set_title("BEV Sensor Fusion: Reliance Discounting Across Architectures", fontsize=12, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(frameon=True, fontsize=9)
        plt.tight_layout()

        fig_path = fig_dir / "bev_reliance_vs_p.pdf"
        fig.savefig(fig_path, format="pdf", bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {fig_path}")
    except Exception as e:
        print(f"[WARN] Failed to plot BEV figure: {e}")


def main():
    args = parse_args()
    run_full_bev_analysis(
        raw_dir=Path(args.raw_dir),
        out_dir=Path(args.out_dir),
        table_dir=Path(args.table_dir),
        fig_dir=Path(args.fig_dir),
        n_boot=args.n_boot,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
