"""Analysis and visualization of modality-dropout fusion case study."""

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

CB_COLORS = ["#0072B2", "#E69F00", "#009E73", "#CC79A7"]


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze modality-dropout fusion experiments")
    parser.add_argument(
        "--raw_dir",
        type=str,
        default="results/raw/fusion",
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


def load_fusion_predictions(raw_dir: Path) -> Dict[Tuple[str, float], List[Dict[str, Any]]]:
    """Load predictions grouped by (dataset, dropout_p)."""
    groups: Dict[Tuple[str, float], List[Dict[str, Any]]] = {}
    for pred_file in sorted(raw_dir.glob("**/predictions.json")):
        try:
            with open(pred_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            dset = data["dataset"]
            p = float(data["dropout_p"])
            groups.setdefault((dset, p), []).append(data)
        except Exception as e:
            print(f"[WARN] Failed to load {pred_file}: {e}")
    return groups


def analyze_dataset_p_condition(
    dataset: str,
    dropout_p: float,
    seed_runs: List[Dict[str, Any]],
    n_boot: int = 1000,
    seed: int = 42,
) -> Dict[str, Any]:
    """Analyze one (dataset, p) condition across its seeds."""
    S_grid = np.array(seed_runs[0]["S_grid"])
    n_S = len(S_grid)

    # Flatten items across seeds
    all_items = []
    seed_item_lists = []
    for run in seed_runs:
        all_items.extend(run["items"])
        seed_item_lists.append(run["items"])

    n_items_total = len(all_items)
    correct_idx = np.array([it["correct_choice_idx"] for it in all_items])
    null_probs = np.array([it["null_probs"]["removed"] for it in all_items])

    null_preds = np.argmax(null_probs, axis=-1)
    acc_null_per_item = (null_preds == correct_idx).astype(float)
    acc_null = float(np.mean(acc_null_per_item))

    probs_S = np.array([[it["probs_per_S"][s] for it in all_items] for s in range(n_S)])
    preds_S = np.argmax(probs_S, axis=-1)
    acc_S_matrix = (preds_S == correct_idx[None, :]).astype(float)
    acc_deg = np.mean(acc_S_matrix, axis=1)

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
        b_acc_deg = np.mean((b_preds_S == b_corr[None, :]).astype(float), axis=1)

        b_u = utility(b_acc_deg, b_acc_null)
        b_s = utility_crossing(b_u, S_grid, window=3)

        if b_s is not None:
            # JSD
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
                val_mi = misalignment_index(b_R_hat, S_grid, b_s)
                boot_mi.append(val_mi if val_mi is not None else np.nan)
            except Exception:
                boot_mi.append(np.nan)
        else:
            boot_mi.append(np.nan)

    valid_mi = [m for m in boot_mi if not np.isnan(m)]
    if len(valid_mi) > 0:
        ci_low, ci_high = float(np.percentile(valid_mi, 2.5)), float(np.percentile(valid_mi, 97.5))
    else:
        ci_low, ci_high = np.nan, np.nan

    return {
        "dataset": dataset,
        "dropout_p": dropout_p,
        "n_seeds": n_seeds,
        "clean_acc": float(acc_deg[0]),
        "acc_null_removed": acc_null,
        "S_star": S_star,
        "MI": MI,
        "MI_ci_low": ci_low,
        "MI_ci_high": ci_high,
    }


def run_full_fusion_analysis(
    raw_dir: Path,
    out_dir: Path,
    table_dir: Path,
    fig_dir: Path,
    n_boot: int = 1000,
    seed: int = 42,
):
    """Run end-to-end analysis on fusion outputs."""
    out_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    grouped = load_fusion_predictions(raw_dir)
    if not grouped:
        print(f"[WARN] No fusion predictions found in {raw_dir}.")
        return

    records = []
    for (dset, p), runs in sorted(grouped.items()):
        print(f"Analyzing fusion: {dset} (p={p}) across {len(runs)} seed(s)...")
        res = analyze_dataset_p_condition(dset, p, runs, n_boot=n_boot, seed=seed)
        records.append(res)

    df = pd.DataFrame(records)
    csv_path = out_dir / "fusion_summary.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")

    # LaTeX Table
    tex_path = table_dir / "fusion_summary.tex"
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Modality-Dropout Impact on Misalignment Index (MI) and Clean Accuracy.}",
        "\\label{tab:fusion_summary}",
        "\\small",
        df.to_latex(index=False, escape=False).strip(),
        "\\end{table}",
    ]
    tex_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved: {tex_path}")

    # PDF Figure: MI vs p
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 4))
        for idx, dset in enumerate(df["dataset"].unique()):
            sub = df[df["dataset"] == dset].sort_values("dropout_p")
            color = CB_COLORS[idx % len(CB_COLORS)]
            yerr_low = np.maximum(0, sub["MI"] - sub["MI_ci_low"])
            yerr_high = np.maximum(0, sub["MI_ci_high"] - sub["MI"])
            ax.errorbar(
                sub["dropout_p"],
                sub["MI"],
                yerr=[yerr_low, yerr_high],
                label=dset,
                fmt="o-",
                color=color,
                capsize=4,
                linewidth=1.8,
            )

        ax.set_xlabel("Modality Dropout Probability $p$", fontsize=11)
        ax.set_ylabel("Misalignment Index (MI)", fontsize=11)
        ax.set_title("Modality Dropout Mitigates Reliance Misalignment", fontsize=12, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(frameon=True)
        plt.tight_layout()

        fig_path = fig_dir / "fusion_mi_vs_p.pdf"
        fig.savefig(fig_path, format="pdf", bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {fig_path}")
    except Exception as e:
        print(f"[WARN] Failed to plot fusion figure: {e}")


def main():
    args = parse_args()
    run_full_fusion_analysis(
        raw_dir=Path(args.raw_dir),
        out_dir=Path(args.out_dir),
        table_dir=Path(args.table_dir),
        fig_dir=Path(args.fig_dir),
        n_boot=args.n_boot,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
