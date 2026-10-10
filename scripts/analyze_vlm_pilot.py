"""Analysis and visualization of VLM reliance-utility pilot evaluation shards (CPU only).

Loads raw shards from results/raw/vlm_pilot/, computes:
- Accuracy P = argmax(p) == correct
- Utility U(S) = acc(S) - acc(null)
- Reliance R(S) = mean JSD(probs(S), probs(null)) using metrics.jsd
- Normalized reliance R_hat(S), utility crossing S*, misalignment index MI
- Pooled bootstrap CI over questions (n_boot=1000) reporting no_crossing_fraction
- Gate criteria evaluation (1)-(3)
- Vector PDF plots (colorblind-safe): U(S), R_hat(S), and Trajectory (U vs R_hat colored by S).
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
    parser = argparse.ArgumentParser(description="Analyze VLM pilot evaluation shards")
    parser.add_argument(
        "--raw_dir",
        type=str,
        default="results/raw/vlm_pilot",
        help="Directory containing raw JSON shards",
    )
    parser.add_argument(
        "--fig_dir",
        type=str,
        default="figures",
        help="Output directory for generated vector figures",
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


def load_shards_by_corruption(raw_dir: Path) -> Dict[str, List[Dict[str, Any]]]:
    """Load and merge all items across shards grouped by corruption kind."""
    json_files = sorted(raw_dir.glob("*.json"))
    corruptions: Dict[str, Dict[str, Any]] = {}

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

        corr = data["corruption"]
        if corr not in corruptions:
            corruptions[corr] = {
                "model_name": data.get("model_name", "Qwen2-VL"),
                "S_grid": np.array(data["S_grid"]),
                "items": {},
            }

        # Deduplicate items by question_id
        for item in data["items"]:
            corruptions[corr]["items"][item["question_id"]] = item

    # Flatten items into sorted list
    result = {}
    for corr, bundle in corruptions.items():
        items_list = list(bundle["items"].values())
        result[corr] = {
            "model_name": bundle["model_name"],
            "S_grid": bundle["S_grid"],
            "items": items_list,
        }
    return result


def analyze_corruption_data(
    bundle: Dict[str, Any],
    null_kind: str = "removed",
    n_boot: int = 1000,
    seed: int = 42,
) -> Dict[str, Any]:
    """Compute point estimates and bootstrap CIs for a single corruption."""
    items = bundle["items"]
    S_grid = bundle["S_grid"]
    n_items = len(items)
    n_S = len(S_grid)

    if n_items == 0:
        return {}

    # Extract per-item correctness and probability vectors
    # correct_choice_idx: (N,)
    correct_idx = np.array([it["correct_choice_idx"] for it in items])

    # null probs: (N, 4)
    null_probs = np.array([it["null_probs"][null_kind] for it in items])
    null_preds = np.argmax(null_probs, axis=-1)
    acc_null_per_item = (null_preds == correct_idx).astype(float)
    acc_null = float(np.mean(acc_null_per_item))

    # probs across S: (n_S, N, 4)
    probs_S = np.array([
        [it["probs_per_S"][s] for it in items]
        for s in range(n_S)
    ])  # shape: (n_S, N, 4)

    # accuracy across S: (n_S, N)
    preds_S = np.argmax(probs_S, axis=-1)
    acc_S_matrix = (preds_S == correct_idx[None, :]).astype(float)  # shape: (n_S, N)
    acc_deg = np.mean(acc_S_matrix, axis=1)  # shape: (n_S,)

    # JSD per item across S: (n_S, N)
    jsd_matrix = np.zeros((n_S, n_items), dtype=float)
    for s in range(n_S):
        ps = probs_S[s]
        pn = null_probs
        m = 0.5 * (ps + pn)
        kl_p = np.sum(ps * np.log2(np.maximum(ps, 1e-12) / np.maximum(m, 1e-12)), axis=-1)
        kl_n = np.sum(pn * np.log2(np.maximum(pn, 1e-12) / np.maximum(m, 1e-12)), axis=-1)
        jsd_matrix[s] = np.clip(0.5 * (kl_p + kl_n), 0.0, 1.0)

    # Point estimates
    D_S = np.mean(jsd_matrix, axis=1)  # shape: (n_S,)
    U_S = utility(acc_deg, acc_null)
    R_S = reliance(D_S)
    R_hat_S = normalized_reliance(R_S)
    S_star = utility_crossing(U_S, S_grid, window=3)
    MI = misalignment_index(R_hat_S, S_grid, S_star)

    # Clean accuracy (S=0)
    clean_acc = float(acc_deg[0])

    # Other nulls and controls
    black_probs = np.array([it["null_probs"]["black"] for it in items])
    black_acc = float(np.mean(np.argmax(black_probs, axis=-1) == correct_idx))

    swapped_probs = np.array([it["swapped_probs"] for it in items])
    swapped_acc = float(np.mean(np.argmax(swapped_probs, axis=-1) == correct_idx))

    # Mean mass on letters across all evaluations
    all_masses = []
    for it in items:
        all_masses.append(it["null_mass"]["removed"])
        all_masses.append(it["null_mass"]["black"])
        all_masses.append(it["swapped_mass"])
        all_masses.extend(it["mass_per_S"])
    mean_letter_mass = float(np.mean(all_masses))

    # Bootstrap CI over questions
    rng = np.random.default_rng(seed)
    boot_s = []
    boot_mi = []
    none_count = 0

    for _ in range(n_boot):
        idx = rng.integers(0, n_items, size=n_items)
        b_acc_deg = np.mean(acc_S_matrix[:, idx], axis=1)
        b_acc_null = float(np.mean(acc_null_per_item[idx]))
        b_u = utility(b_acc_deg, b_acc_null)
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

    return {
        "n_questions": n_items,
        "clean_acc": clean_acc,
        "null_acc_removed": acc_null,
        "null_acc_black": black_acc,
        "swapped_acc": swapped_acc,
        "mean_letter_mass": mean_letter_mass,
        "R_0": float(R_S[0]),
        "S_star": S_star,
        "S_star_lo": s_lo,
        "S_star_hi": s_hi,
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


def evaluate_gate_criteria(stats_by_corruption: Dict[str, Dict[str, Any]]):
    """Evaluate pre-declared gate criteria (1)-(3) and print diagnostic table."""
    print("\n" + "=" * 90)
    print("VLM PILOT GATE CRITERIA EVALUATION (Qwen2-VL-2B on A-OKVQA)")
    print("=" * 90)

    # 1. Gate summary table
    print(
        f"{'Corruption':<12} | {'R(0)':>7} | {'Clean Acc':>9} | {'Null (Rem)':>10} | "
        f"{'Null (Blk)':>10} | {'Swapped':>8} | {'U Cross?':>9} | {'S*':>6} | {'MI (95% CI)':>17} | {'Mass':>6}"
    )
    print("-" * 105)

    u_crossed_any = False
    all_r0_positive = True
    text_acc_in_range_all = True

    for corr, st in stats_by_corruption.items():
        r0 = st["R_0"]
        clean_a = st["clean_acc"]
        null_rem = st["null_acc_removed"]
        null_blk = st["null_acc_black"]
        swp = st["swapped_acc"]
        s_star = st["S_star"]
        u_crossed = s_star is not None
        if u_crossed:
            u_crossed_any = True
        mi = st["MI"]
        mi_str = (
            f"{mi:.3f} [{st['MI_lo']:.3f}, {st['MI_hi']:.3f}]"
            if mi is not None and not math.isnan(mi)
            else "no crossing"
        )
        s_star_str = f"{s_star:.3f}" if s_star is not None else "None"
        mass = st["mean_letter_mass"]

        if r0 <= 0.02:
            all_r0_positive = False

        # Text-only accuracy condition: > 0.25 and < clean_accuracy - 0.1
        if not (0.25 < null_rem < (clean_a - 0.10)):
            text_acc_in_range_all = False

        print(
            f"{corr:<12} | {r0:>7.4f} | {clean_a:>9.3f} | {null_rem:>10.3f} | "
            f"{null_blk:>10.3f} | {swp:>8.3f} | {'YES' if u_crossed else 'NO':>9} | "
            f"{s_star_str:>6} | {mi_str:>17} | {mass:>6.3f}"
        )

    print("-" * 105)

    # 2. Gate verdicts
    pass_1 = all_r0_positive
    pass_2 = u_crossed_any
    pass_3 = text_acc_in_range_all

    print("\n--- Gate Criterion Verdicts ---")
    print(f"(1) R(0) > 0.02 for tested corruptions:                {'PASS' if pass_1 else 'FAIL'}")
    print(f"(2) U(S) crosses zero for at least one corruption:      {'PASS' if pass_2 else 'FAIL'}")
    print(f"(3) Text-only accuracy > 0.25 and < clean_acc - 0.10:    {'PASS' if pass_3 else 'FAIL'}")

    all_pass = pass_1 and pass_2 and pass_3
    print(f"\nOverall Pilot Gate Status: {'PASS' if all_pass else 'FAIL'}")
    if not all_pass:
        print("Diagnosis: Criteria failed; check if visual modality signal is sufficiently strong or if corruption is adequate.")


def plot_vlm_figures(stats_by_corruption: Dict[str, Dict[str, Any]], fig_dir: Path):
    """Generate publication-quality vector PDF figures (colorblind-safe)."""
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Colorblind-safe styling: Wong palette
    # Orange (#E69F00) for Gaussian, Sky Blue (#56B4E9) for Patch
    styles = {
        "gaussian": {"color": "#E69F00", "label": "Gaussian Blend", "ls": "-", "marker": "o"},
        "patch":    {"color": "#56B4E9", "label": "Patch Mask",     "ls": "--", "marker": "s"},
    }

    # 1. Figure 1: Utility curve U(S)
    fig1, ax1 = plt.subplots(figsize=(6.5, 4.2))
    for corr, st in stats_by_corruption.items():
        cfg = styles.get(corr, {"color": "black", "label": corr, "ls": "-", "marker": "o"})
        sg = st["S_grid"]
        u_sm = st["U_smooth"]
        ax1.plot(sg, u_sm, color=cfg["color"], ls=cfg["ls"], marker=cfg["marker"], label=cfg["label"], lw=2.0)
        if st["S_star"] is not None:
            ax1.axvline(st["S_star"], color=cfg["color"], ls=":", alpha=0.7, label=f"S* ({corr}) = {st['S_star']:.2f}")

    ax1.axhline(0.0, color="gray", ls="--", lw=1.0)
    ax1.set_xlabel("Degradation Severity S", fontsize=11)
    ax1.set_ylabel("Utility U(S) = Acc(S) - Acc(Null)", fontsize=11)
    ax1.set_title("VLM Pilot: Modality Utility vs Severity (Qwen2-VL on A-OKVQA)", fontsize=11)
    ax1.legend(fontsize=9, loc="upper right")
    ax1.set_xlim(-0.02, 1.02)
    fig1.tight_layout()
    f1_path = fig_dir / "vlm_pilot_utility_curves.pdf"
    fig1.savefig(f1_path)
    plt.close(fig1)
    print(f"Saved figure: {f1_path.name}")

    # 2. Figure 2: Normalized Reliance curve R_hat(S)
    fig2, ax2 = plt.subplots(figsize=(6.5, 4.2))
    for corr, st in stats_by_corruption.items():
        cfg = styles.get(corr, {"color": "black", "label": corr, "ls": "-", "marker": "o"})
        sg = st["S_grid"]
        rh = st["R_hat_curve"]
        ax2.plot(sg, rh, color=cfg["color"], ls=cfg["ls"], marker=cfg["marker"], label=cfg["label"], lw=2.0)

    ax2.axhline(1.0, color="gray", ls="--", lw=1.0, label="R_hat = 1.0 (Clean Baseline)")
    ax2.set_xlabel("Degradation Severity S", fontsize=11)
    ax2.set_ylabel("Normalized Reliance R_hat(S)", fontsize=11)
    ax2.set_title("VLM Pilot: Normalized Reliance vs Severity", fontsize=11)
    ax2.legend(fontsize=9, loc="lower left")
    ax2.set_xlim(-0.02, 1.02)
    fig2.tight_layout()
    f2_path = fig_dir / "vlm_pilot_reliance_curves.pdf"
    fig2.savefig(f2_path)
    plt.close(fig2)
    print(f"Saved figure: {f2_path.name}")

    # 3. Figure 3: Reliance-Utility Trajectory (U vs R_hat colored by S)
    fig3, ax3 = plt.subplots(figsize=(6.5, 4.8))
    cmap = plt.get_cmap("viridis")

    for corr, st in stats_by_corruption.items():
        cfg = styles.get(corr, {"color": "black", "label": corr, "ls": "-", "marker": "o"})
        sg = st["S_grid"]
        u_vals = st["U_smooth"]
        rh_vals = st["R_hat_curve"]

        # Plot trajectory line
        ax3.plot(rh_vals, u_vals, color=cfg["color"], ls=cfg["ls"], lw=1.5, alpha=0.6, label=f"Trajectory ({cfg['label']})")
        # Scatter points colored by S
        sc = ax3.scatter(rh_vals, u_vals, c=sg, cmap=cmap, vmin=0.0, vmax=1.0, s=50, edgecolors="black", linewidths=0.5, zorder=3)

    cbar = fig3.colorbar(sc, ax=ax3)
    cbar.set_label("Severity S", fontsize=10)
    ax3.axhline(0.0, color="gray", ls="--", lw=1.0, label="U = 0")
    ax3.set_xlabel("Normalized Reliance R_hat(S)", fontsize=11)
    ax3.set_ylabel("Utility U(S)", fontsize=11)
    ax3.set_title("VLM Pilot: Reliance-Utility Trajectory (Colored by S)", fontsize=11)
    ax3.legend(fontsize=8, loc="upper right")
    fig3.tight_layout()
    f3_path = fig_dir / "vlm_pilot_trajectory.pdf"
    fig3.savefig(f3_path)
    plt.close(fig3)
    print(f"Saved figure: {f3_path.name}")


def main():
    args = parse_args()
    raw_dir = Path(args.raw_dir)
    fig_dir = Path(args.fig_dir)

    if not raw_dir.exists():
        print(f"Directory {raw_dir} does not exist.")
        return

    corruptions_data = load_shards_by_corruption(raw_dir)
    if not corruptions_data:
        print(f"No valid evaluation shards found in {raw_dir}.")
        return

    stats = {}
    for corr, bundle in corruptions_data.items():
        st = analyze_corruption_data(
            bundle,
            null_kind="removed",
            n_boot=args.n_boot,
            seed=args.seed,
        )
        if st:
            stats[corr] = st

    if not stats:
        print("No corruption stats could be computed.")
        return

    evaluate_gate_criteria(stats)
    plot_vlm_figures(stats, fig_dir)


if __name__ == "__main__":
    main()
