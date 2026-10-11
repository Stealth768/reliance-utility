"""Main study analysis and visualization of VLM reliance-utility benchmark shards.

Analyzes raw shards from results/raw/vlm_main/ (or results/raw/vlm_pilot/):
- Computes U(S), R(S), R_hat(S), S*, MI with pooled bootstrap CI (n_boot=1000)
- Accuracies for nulls (removed, black) and swapped control
- Formats S* at grid edge (last grid point) as 'edge, no usable crossing'
- Computes significant crossing: first S where upper bootstrap CI of U(S) < 0
- Table of mean letter mass per model
- Publication-quality vector PDF figures (colorblind-safe, distinct line styles)
  for U(S), R_hat(S), and reliance-utility trajectory per model and corruption.
"""

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Ensure repo root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.metrics import (
    jsd,
    misalignment_index,
    normalized_reliance,
    reliance,
    smooth,
    utility,
    utility_crossing,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze VLM main study evaluation shards")
    parser.add_argument(
        "--raw_dir",
        type=str,
        default="results/raw/vlm_main",
        help="Directory containing raw JSON shards (default: results/raw/vlm_main)",
    )
    parser.add_argument(
        "--fig_dir",
        type=str,
        default="figures",
        help="Output directory for generated vector figures (default: figures)",
    )
    parser.add_argument(
        "--out_csv",
        type=str,
        default="results/derived/vlm_main_summary.csv",
        help="Path to save output CSV summary (default: results/derived/vlm_main_summary.csv)",
    )
    parser.add_argument(
        "--n_boot",
        type=int,
        default=1000,
        help="Number of pooled bootstrap resamples for CI estimation",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for bootstrap resampling",
    )
    return parser.parse_args()


def load_all_shards(raw_dir: Path) -> Dict[Tuple[str, str], Dict[str, Any]]:
    """Load and merge all shards grouped by (model_name, corruption)."""
    json_files = sorted(raw_dir.glob("*.json"))
    bundles: Dict[Tuple[str, str], Dict[str, Any]] = {}

    for jf in json_files:
        if "manifest" in jf.name:
            continue
        try:
            with open(jf, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue

        if "corruption" not in data or "items" not in data:
            continue

        model_name = data.get("model_name", "VLM")
        corruption = data["corruption"]
        key = (model_name, corruption)

        if key not in bundles:
            bundles[key] = {
                "model_name": model_name,
                "corruption": corruption,
                "S_grid": np.array(data["S_grid"]),
                "items": {},
            }

        for item in data["items"]:
            bundles[key]["items"][item["question_id"]] = item

    result = {}
    for key, b in bundles.items():
        result[key] = {
            "model_name": b["model_name"],
            "corruption": b["corruption"],
            "S_grid": b["S_grid"],
            "items": list(b["items"].values()),
        }
    return result


def analyze_model_corruption(
    bundle: Dict[str, Any],
    null_kind: str = "removed",
    n_boot: int = 1000,
    seed: int = 42,
) -> Dict[str, Any]:
    """Compute point estimates, bootstrap CIs, and significant crossing for one (model, corruption)."""
    items = bundle["items"]
    S_grid = bundle["S_grid"]
    n_items = len(items)
    n_S = len(S_grid)

    if n_items == 0:
        return {}

    correct_idx = np.array([it["correct_choice_idx"] for it in items])
    null_probs = np.array([it["null_probs"][null_kind] for it in items])
    null_preds = np.argmax(null_probs, axis=-1)
    acc_null_per_item = (null_preds == correct_idx).astype(float)
    acc_null = float(np.mean(acc_null_per_item))

    # Probs matrix across severity grid: (n_S, N, 4)
    probs_S = np.array([
        [it["probs_per_S"][s] for it in items]
        for s in range(n_S)
    ])
    preds_S = np.argmax(probs_S, axis=-1)
    acc_S_matrix = (preds_S == correct_idx[None, :]).astype(float)
    acc_deg = np.mean(acc_S_matrix, axis=1)

    # JSD matrix per item: (n_S, N)
    jsd_matrix = np.zeros((n_S, n_items), dtype=float)
    for s in range(n_S):
        ps = probs_S[s]
        pn = null_probs
        m = 0.5 * (ps + pn)
        kl_p = np.sum(ps * np.log2(np.maximum(ps, 1e-12) / np.maximum(m, 1e-12)), axis=-1)
        kl_n = np.sum(pn * np.log2(np.maximum(pn, 1e-12) / np.maximum(m, 1e-12)), axis=-1)
        jsd_matrix[s] = np.clip(0.5 * (kl_p + kl_n), 0.0, 1.0)

    # Point estimates
    D_S = np.mean(jsd_matrix, axis=1)
    U_S = utility(acc_deg, acc_null)
    R_S = reliance(D_S)
    R_hat_S = normalized_reliance(R_S)
    S_star_raw = utility_crossing(U_S, S_grid, window=3)
    MI = misalignment_index(R_hat_S, S_grid, S_star_raw)

    clean_acc = float(acc_deg[0])
    black_probs = np.array([it["null_probs"]["black"] for it in items])
    black_acc = float(np.mean(np.argmax(black_probs, axis=-1) == correct_idx))
    swapped_probs = np.array([it["swapped_probs"] for it in items])
    swapped_acc = float(np.mean(np.argmax(swapped_probs, axis=-1) == correct_idx))

    # Mean letter mass
    all_masses = []
    for it in items:
        all_masses.append(it["null_mass"]["removed"])
        all_masses.append(it["null_mass"]["black"])
        all_masses.append(it["swapped_mass"])
        all_masses.extend(it["mass_per_S"])
    mean_letter_mass = float(np.mean(all_masses))

    # Pooled bootstrap resampling
    rng = np.random.default_rng(seed)
    boot_s = []
    boot_mi = []
    boot_u_matrix = np.zeros((n_boot, n_S), dtype=float)
    none_count = 0

    for b in range(n_boot):
        idx = rng.integers(0, n_items, size=n_items)
        b_acc_deg = np.mean(acc_S_matrix[:, idx], axis=1)
        b_acc_null = float(np.mean(acc_null_per_item[idx]))
        b_u = utility(b_acc_deg, b_acc_null)
        boot_u_matrix[b] = b_u
        b_s = utility_crossing(b_u, S_grid, window=3)

        if b_s is None:
            none_count += 1
        else:
            boot_s.append(b_s)
            b_d = np.mean(jsd_matrix[:, idx], axis=1)
            b_r = reliance(b_d)
            b_rh = normalized_reliance(b_r)
            b_mi = misalignment_index(b_rh, S_grid, b_s)
            if b_mi is not None:
                boot_mi.append(b_mi)

    no_crossing_fraction = none_count / n_boot
    s_lo = float(np.percentile(boot_s, 2.5)) if boot_s else float("nan")
    s_hi = float(np.percentile(boot_s, 97.5)) if boot_s else float("nan")
    mi_lo = float(np.percentile(boot_mi, 2.5)) if boot_mi else float("nan")
    mi_hi = float(np.percentile(boot_mi, 97.5)) if boot_mi else float("nan")

    # Significant crossing: first S where bootstrap upper CI of U(S) < 0
    u_upper_ci = np.percentile(boot_u_matrix, 97.5, axis=0)  # shape (n_S,)
    significant_crossing: Optional[float] = None
    for s_idx, S_val in enumerate(S_grid):
        if u_upper_ci[s_idx] < 0.0:
            significant_crossing = float(S_val)
            break

    # Format S* display
    last_grid_point = float(S_grid[-1])
    if S_star_raw is None:
        s_star_display = "no crossing"
    elif np.isclose(S_star_raw, last_grid_point, atol=1e-12) or S_star_raw >= last_grid_point:
        s_star_display = "edge, no usable crossing"
    else:
        s_star_display = f"{S_star_raw:.3f}"

    sig_crossing_display = (
        f"{significant_crossing:.3f}" if significant_crossing is not None else "none"
    )

    return {
        "model_name": bundle["model_name"],
        "corruption": bundle["corruption"],
        "n_questions": n_items,
        "clean_acc": clean_acc,
        "null_acc_removed": acc_null,
        "null_acc_black": black_acc,
        "swapped_acc": swapped_acc,
        "mean_letter_mass": mean_letter_mass,
        "R_0": float(R_S[0]),
        "S_star_raw": S_star_raw,
        "S_star_display": s_star_display,
        "S_star_lo": s_lo,
        "S_star_hi": s_hi,
        "significant_crossing": significant_crossing,
        "significant_crossing_display": sig_crossing_display,
        "MI": MI,
        "MI_lo": mi_lo,
        "MI_hi": mi_hi,
        "no_crossing_fraction": no_crossing_fraction,
        "S_grid": S_grid,
        "U_curve": U_S,
        "U_smooth": smooth(U_S, window=3),
        "R_curve": R_S,
        "R_hat_curve": R_hat_S,
    }


def print_and_export_results(all_stats: List[Dict[str, Any]], out_csv_path: Path):
    """Print clean summary tables and export CSV."""
    rows = []
    print("\n" + "=" * 115)
    print("VLM RELIANCE-UTILITY MAIN BENCHMARK SUMMARY TABLE")
    print("=" * 115)
    print(
        f"{'Model':<22} | {'Corruption':<10} | {'R(0)':>6} | {'Clean':>5} | {'Rem':>5} | "
        f"{'Blk':>5} | {'Swp':>5} | {'S* (Point)':>26} | {'Sig Cross':>9} | {'MI [95% CI]':>17}"
    )
    print("-" * 130)

    for st in all_stats:
        mname = st["model_name"].split("/")[-1]
        corr = st["corruption"]
        r0 = st["R_0"]
        c_acc = st["clean_acc"]
        rem_acc = st["null_acc_removed"]
        blk_acc = st["null_acc_black"]
        swp_acc = st["swapped_acc"]
        s_star_str = st["S_star_display"]
        sig_str = st["significant_crossing_display"]
        mi = st["MI"]
        mi_str = (
            f"{mi:.3f} [{st['MI_lo']:.3f}, {st['MI_hi']:.3f}]"
            if mi is not None and not math.isnan(mi)
            else "no crossing"
        )

        print(
            f"{mname:<22} | {corr:<10} | {r0:>6.3f} | {c_acc:>5.3f} | {rem_acc:>5.3f} | "
            f"{blk_acc:>5.3f} | {swp_acc:>5.3f} | {s_star_str:>26} | {sig_str:>9} | {mi_str:>17}"
        )

        rows.append({
            "model": st["model_name"],
            "corruption": st["corruption"],
            "n_questions": st["n_questions"],
            "clean_acc": c_acc,
            "null_acc_removed": rem_acc,
            "null_acc_black": blk_acc,
            "swapped_acc": swp_acc,
            "mean_letter_mass": st["mean_letter_mass"],
            "R_0": r0,
            "S_star_display": s_star_str,
            "S_star_lo": st["S_star_lo"],
            "S_star_hi": st["S_star_hi"],
            "significant_crossing": sig_str,
            "MI": mi if mi is not None else float("nan"),
            "MI_lo": st["MI_lo"],
            "MI_hi": st["MI_hi"],
            "no_crossing_fraction": st["no_crossing_fraction"],
        })

    print("-" * 130)

    # Table of Mean Letter Mass per Model
    model_mass_dict: Dict[str, List[float]] = {}
    for st in all_stats:
        model_mass_dict.setdefault(st["model_name"], []).append(st["mean_letter_mass"])

    print("\n" + "=" * 50)
    print("MEAN LETTER PROBABILITY MASS PER MODEL")
    print("=" * 50)
    for model_name, masses in model_mass_dict.items():
        print(f"  {model_name:<35} : {np.mean(masses):.4f}")
    print("=" * 50)

    # Save to CSV
    out_csv_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out_csv_path, index=False)
    print(f"\nSaved benchmark summary CSV to: {out_csv_path}")


def plot_all_figures(all_stats: List[Dict[str, Any]], fig_dir: Path):
    """Generate colorblind-safe publication-quality vector figures per model and corruption."""
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Colorblind-safe styling: Wong palette
    corr_styles = {
        "gaussian":  {"color": "#E69F00", "label": "Gaussian Blend", "ls": "-",  "marker": "o"},
        "patch":     {"color": "#56B4E9", "label": "Patch Mask",     "ls": "--", "marker": "s"},
        "swapblend": {"color": "#D55E00", "label": "Swap Blend",     "ls": "-.", "marker": "^"},
    }

    # Group by model
    by_model: Dict[str, List[Dict[str, Any]]] = {}
    for st in all_stats:
        by_model.setdefault(st["model_name"], []).append(st)

    cmap = plt.get_cmap("viridis")

    for model_name, stats_list in by_model.items():
        slug = model_name.split("/")[-1]

        # 1. Utility curves
        fig1, ax1 = plt.subplots(figsize=(6.5, 4.2))
        for st in stats_list:
            corr = st["corruption"]
            cfg = corr_styles.get(corr, {"color": "black", "label": corr, "ls": "-", "marker": "o"})
            sg = st["S_grid"]
            u_sm = st["U_smooth"]
            ax1.plot(sg, u_sm, color=cfg["color"], ls=cfg["ls"], marker=cfg["marker"], label=cfg["label"], lw=2.0)
            if st["significant_crossing"] is not None:
                ax1.axvline(
                    st["significant_crossing"],
                    color=cfg["color"],
                    ls=":",
                    alpha=0.6,
                    label=f"Sig Cross ({corr}): {st['significant_crossing']:.2f}",
                )

        ax1.axhline(0.0, color="gray", ls="--", lw=1.0)
        ax1.set_xlabel("Degradation Severity S", fontsize=11)
        ax1.set_ylabel("Utility U(S) = Acc(S) - Acc(Null)", fontsize=11)
        ax1.set_title(f"Utility vs Severity ({slug})", fontsize=11)
        ax1.legend(fontsize=8, loc="upper right")
        ax1.set_xlim(-0.02, 1.02)
        fig1.tight_layout()
        f1_path = fig_dir / f"vlm_{slug}_utility_curves.pdf"
        fig1.savefig(f1_path)
        plt.close(fig1)

        # 2. Normalized Reliance curves
        fig2, ax2 = plt.subplots(figsize=(6.5, 4.2))
        for st in stats_list:
            corr = st["corruption"]
            cfg = corr_styles.get(corr, {"color": "black", "label": corr, "ls": "-", "marker": "o"})
            sg = st["S_grid"]
            rh = st["R_hat_curve"]
            ax2.plot(sg, rh, color=cfg["color"], ls=cfg["ls"], marker=cfg["marker"], label=cfg["label"], lw=2.0)

        ax2.axhline(1.0, color="gray", ls="--", lw=1.0, label="R_hat = 1.0 (Clean Baseline)")
        ax2.set_xlabel("Degradation Severity S", fontsize=11)
        ax2.set_ylabel("Normalized Reliance R_hat(S)", fontsize=11)
        ax2.set_title(f"Normalized Reliance vs Severity ({slug})", fontsize=11)
        ax2.legend(fontsize=8, loc="lower left")
        ax2.set_xlim(-0.02, 1.02)
        fig2.tight_layout()
        f2_path = fig_dir / f"vlm_{slug}_reliance_curves.pdf"
        fig2.savefig(f2_path)
        plt.close(fig2)

        # 3. Trajectory plot (U vs R_hat colored by S)
        fig3, ax3 = plt.subplots(figsize=(6.5, 4.8))
        for st in stats_list:
            corr = st["corruption"]
            cfg = corr_styles.get(corr, {"color": "black", "label": corr, "ls": "-", "marker": "o"})
            sg = st["S_grid"]
            u_vals = st["U_smooth"]
            rh_vals = st["R_hat_curve"]
            ax3.plot(rh_vals, u_vals, color=cfg["color"], ls=cfg["ls"], lw=1.5, alpha=0.6, label=f"Trajectory ({cfg['label']})")
            sc = ax3.scatter(rh_vals, u_vals, c=sg, cmap=cmap, vmin=0.0, vmax=1.0, s=45, edgecolors="black", linewidths=0.5, zorder=3)

        cbar = fig3.colorbar(sc, ax=ax3)
        cbar.set_label("Severity S", fontsize=10)
        ax3.axhline(0.0, color="gray", ls="--", lw=1.0, label="U = 0")
        ax3.set_xlabel("Normalized Reliance R_hat(S)", fontsize=11)
        ax3.set_ylabel("Utility U(S)", fontsize=11)
        ax3.set_title(f"Reliance-Utility Trajectory ({slug})", fontsize=11)
        ax3.legend(fontsize=8, loc="upper right")
        fig3.tight_layout()
        f3_path = fig_dir / f"vlm_{slug}_trajectory.pdf"
        fig3.savefig(f3_path)
        plt.close(fig3)

    print(f"\nAll publication figures saved to: {fig_dir}/")


def main():
    args = parse_args()
    raw_dir = Path(args.raw_dir)
    fig_dir = Path(args.fig_dir)
    out_csv = Path(args.out_csv)

    if not raw_dir.exists():
        print(f"Raw results directory '{raw_dir}' does not exist.")
        return

    bundles = load_all_shards(raw_dir)
    if not bundles:
        print(f"No valid evaluation shards found in '{raw_dir}'.")
        return

    all_stats = []
    for key, bundle in sorted(bundles.items()):
        st = analyze_model_corruption(
            bundle,
            null_kind="removed",
            n_boot=args.n_boot,
            seed=args.seed,
        )
        if st:
            all_stats.append(st)

    if not all_stats:
        print("No evaluations could be analyzed.")
        return

    print_and_export_results(all_stats, out_csv)
    plot_all_figures(all_stats, fig_dir)


if __name__ == "__main__":
    main()
