"""Multi-model analysis and paper artifacts generator for VLM reliance-utility benchmark.

Processes all shards in results/raw/vlm_main/:
- Groups by (model, corruption, seed)
- Computes U(S), R(S), R_hat(S), S*, significant S*, MI
- Pooled bootstrap CIs (n_boot=1000) and hierarchical bootstrap CIs (seeds then items)
- No-crossing fraction, control accuracies (removed, black, swapped, extra nulls if present)
- Mean letter mass
- Null sensitivity table (MI and S* across null variants)
- Correlation check: Pearson correlation between MI and plain accuracy drop per model
- Exports derived CSVs, LaTeX booktabs tables, and colorblind-safe vector PDF figures.
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

import numpy as np
import pandas as pd
from scipy import stats

from src.metrics import (
    jsd,
    misalignment_index,
    normalized_reliance,
    reliance,
    smooth,
    utility,
    utility_crossing,
)

# Wong (2011) colorblind-safe palette
CB_COLORS = [
    "#0072B2",  # Blue
    "#E69F00",  # Orange
    "#009E73",  # Green
    "#CC79A7",  # Pink
    "#56B4E9",  # Sky Blue
    "#D55E00",  # Vermilion
    "#F0E442",  # Yellow
]

CORRUPTION_STYLES = {
    "gaussian": {"linestyle": "-", "marker": "o"},
    "patch": {"linestyle": "--", "marker": "s"},
    "swapblend": {"linestyle": "-.", "marker": "^"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Multi-model analysis and paper artifact generator")
    parser.add_argument(
        "--raw_dir",
        type=str,
        default="results/raw/vlm_main",
        help="Directory containing raw JSON shards",
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
        help="Directory for generated LaTeX tables",
    )
    parser.add_argument(
        "--fig_dir",
        type=str,
        default="figures",
        help="Directory for generated vector PDF figures",
    )
    parser.add_argument(
        "--n_boot",
        type=int,
        default=1000,
        help="Number of bootstrap resamples",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for bootstrap resampling",
    )
    return parser.parse_args()


def load_shards_grouped(raw_dir: Path) -> Dict[Tuple[str, str, int], Dict[str, Any]]:
    """Load all JSON shards and group by (model_name, corruption, seed).

    Scans `raw_dir` and any sibling/sub `vlm_*` directories under `results/raw/`
    so both legacy Qwen shards (in `vlm_main`) and new LLaVA/SmolVLM shards
    (in `vlm_llava`, `vlm_smolvlm`, etc.) are processed seamlessly.
    """
    raw_path = Path(raw_dir)
    target_dirs = []
    if raw_path.name.startswith("vlm_") and raw_path.parent.exists():
        # Scan raw_dir and any sibling vlm_* directories in the same parent (e.g. results/raw/vlm_*)
        target_dirs = sorted(list(raw_path.parent.glob("vlm_*")))
    elif any(raw_path.glob("vlm_*")):
        # If passed results/raw directly
        target_dirs = sorted(list(raw_path.glob("vlm_*")))
    else:
        target_dirs = [raw_path]

    json_files = []
    for d in target_dirs:
        if d.is_dir():
            json_files.extend(sorted(d.glob("*.json")))

    groups: Dict[Tuple[str, str, int], Dict[str, Any]] = {}

    for jf in json_files:
        if "manifest" in jf.name:
            continue
        try:
            with open(jf, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue

        if not isinstance(data, dict) or "corruption" not in data or "items" not in data:
            continue

        model_name = data.get("model_name", "unknown")
        corruption = data.get("corruption", "unknown")
        raw_seed = data.get("seed")
        if raw_seed is not None:
            seed = int(raw_seed)
        else:
            import re
            m = re.search(r"_seed(\d+)_", jf.name)
            seed = int(m.group(1)) if m else 0
        key = (model_name, corruption, seed)

        if key not in groups:
            groups[key] = {
                "model_name": model_name,
                "corruption": corruption,
                "seed": seed,
                "S_grid": np.array(data.get("S_grid", np.linspace(0.0, 1.0, 9))),
                "items": {},
            }

        for item in data.get("items", []):
            qid = str(item.get("question_id", len(groups[key]["items"])))
            groups[key]["items"][qid] = item

    # Convert items dict to list sorted by question_id or insertion
    result = {}
    for key, g in groups.items():
        result[key] = {
            "model_name": g["model_name"],
            "corruption": g["corruption"],
            "seed": g["seed"],
            "S_grid": g["S_grid"],
            "items": list(g["items"].values()),
        }
    return result


def compute_metrics_for_items(
    items: List[Dict[str, Any]],
    S_grid: np.ndarray,
    null_kind: str = "removed",
) -> Dict[str, Any]:
    """Compute core vectors and scalars for a list of items under specified null."""
    n_items = len(items)
    n_S = len(S_grid)
    if n_items == 0:
        return {}

    correct_idx = np.array([it["correct_choice_idx"] for it in items])

    # Check null availability
    if null_kind in items[0]["null_probs"]:
        null_probs = np.array([it["null_probs"][null_kind] for it in items])
    elif null_kind == "removed" and "null_removed" in items[0]["null_probs"]:
        null_probs = np.array([it["null_probs"]["null_removed"] for it in items])
    else:
        # Fallback to first available null
        avail = list(items[0]["null_probs"].keys())
        null_probs = np.array([it["null_probs"][avail[0]] for it in items])

    null_preds = np.argmax(null_probs, axis=-1)
    acc_null_per_item = (null_preds == correct_idx).astype(float)
    acc_null = float(np.mean(acc_null_per_item))

    # Probs across S
    probs_S = np.array([[it["probs_per_S"][s] for it in items] for s in range(n_S)])
    preds_S = np.argmax(probs_S, axis=-1)
    acc_S_matrix = (preds_S == correct_idx[None, :]).astype(float)
    acc_deg = np.mean(acc_S_matrix, axis=1)

    # JSD per item
    jsd_matrix = np.zeros((n_S, n_items), dtype=float)
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
        S_star_raw = utility_crossing(U_S, S_grid, window=3)
        MI = misalignment_index(R_hat_S, S_grid, S_star_raw)
    except Exception:
        R_hat_S = np.zeros_like(R_S)
        S_star_raw = None
        MI = None

    clean_acc = float(acc_deg[0])
    acc_s1 = float(acc_deg[-1])

    return {
        "n_items": n_items,
        "acc_null": acc_null,
        "acc_null_per_item": acc_null_per_item,
        "acc_S_matrix": acc_S_matrix,
        "acc_deg": acc_deg,
        "clean_acc": clean_acc,
        "acc_s1": acc_s1,
        "D_S": D_S,
        "U_S": U_S,
        "R_S": R_S,
        "R_hat_S": R_hat_S,
        "S_star_raw": S_star_raw,
        "MI": MI,
        "jsd_matrix": jsd_matrix,
    }


def analyze_model_corruption_all_seeds(
    model_name: str,
    corruption: str,
    seed_groups: List[Dict[str, Any]],
    n_boot: int = 1000,
    seed: int = 42,
) -> Dict[str, Any]:
    """Run full analysis across all seeds for a (model, corruption) pair."""
    # Combine items across seeds
    all_items = []
    seeds = []
    for sg in seed_groups:
        all_items.extend(sg["items"])
        seeds.append(sg["seed"])
    S_grid = seed_groups[0]["S_grid"]
    n_S = len(S_grid)

    # Core computation under 'removed'
    core = compute_metrics_for_items(all_items, S_grid, null_kind="removed")
    if not core:
        return {}

    acc_S_matrix = core["acc_S_matrix"]
    acc_null_per_item = core["acc_null_per_item"]
    n_items_total = core["n_items"]

    # Other controls
    correct_idx = np.array([it["correct_choice_idx"] for it in all_items])
    black_acc = None
    if "black" in all_items[0]["null_probs"]:
        b_probs = np.array([it["null_probs"]["black"] for it in all_items])
        black_acc = float(np.mean(np.argmax(b_probs, axis=-1) == correct_idx))

    swapped_acc = None
    if "swapped_probs" in all_items[0]:
        swp_probs = np.array([it["swapped_probs"] for it in all_items])
        swapped_acc = float(np.mean(np.argmax(swp_probs, axis=-1) == correct_idx))

    # Mean letter mass
    all_masses = []
    for it in all_items:
        if "null_mass" in it and isinstance(it["null_mass"], dict):
            for mval in it["null_mass"].values():
                if isinstance(mval, (int, float)):
                    all_masses.append(float(mval))
        if "swapped_mass" in it and isinstance(it["swapped_mass"], (int, float)):
            all_masses.append(float(it["swapped_mass"]))
        if "mass_per_S" in it and isinstance(it["mass_per_S"], list):
            for mval in it["mass_per_S"]:
                if isinstance(mval, (int, float)):
                    all_masses.append(float(mval))
    mean_letter_mass = float(np.mean(all_masses)) if all_masses else 1.0

    # Pooled bootstrap — track dropped resamples by cause
    rng = np.random.default_rng(seed)
    boot_s_list = []
    boot_mi_list = []
    boot_rhat_s1_list: List[float] = []   # R_hat at S=1 (last grid) per valid resample
    boot_u_matrix = np.zeros((n_boot, n_S), dtype=float)
    no_crossing_count = 0
    # Cause sub-counters (sum = n_resamples_dropped)
    n_dropped_no_crossing = 0    # S_star is None → no usable crossing
    n_dropped_edge_crossing = 0  # S_star == last grid point → degenerate interval, MI=None
    n_dropped_r0_zero = 0        # normalized_reliance raises ValueError (R(0) ~ 0)
    n_dropped_other = 0          # any other exception in the MI computation
    n_resamples_dropped = 0

    _last_grid_val = float(S_grid[-1])

    for b in range(n_boot):
        idx = rng.integers(0, n_items_total, size=n_items_total)
        b_acc_deg = np.mean(acc_S_matrix[:, idx], axis=1)
        b_acc_null = float(np.mean(acc_null_per_item[idx]))
        b_u = utility(b_acc_deg, b_acc_null)
        boot_u_matrix[b] = b_u
        b_s = utility_crossing(b_u, S_grid, window=3)

        if b_s is None:
            no_crossing_count += 1
            n_dropped_no_crossing += 1
            n_resamples_dropped += 1
            boot_s_list.append(np.nan)
            boot_mi_list.append(np.nan)
        else:
            boot_s_list.append(b_s)
            b_D = np.mean(core["jsd_matrix"][:, idx], axis=1)
            b_R = reliance(b_D)
            try:
                b_R_hat = normalized_reliance(b_R)
                boot_rhat_s1_list.append(float(b_R_hat[-1]))   # R_hat at S=1
                b_mi = misalignment_index(b_R_hat, S_grid, b_s)
                if b_mi is None or np.isnan(b_mi):
                    # MI is None when S_star is at the grid edge (degenerate interval)
                    n_dropped_edge_crossing += 1
                    n_resamples_dropped += 1
                    boot_mi_list.append(np.nan)
                else:
                    boot_mi_list.append(b_mi)
            except ValueError:
                # normalized_reliance raises ValueError when R(0) ~ 0
                n_dropped_r0_zero += 1
                n_resamples_dropped += 1
                boot_mi_list.append(np.nan)
            except Exception:
                n_dropped_other += 1
                n_resamples_dropped += 1
                boot_mi_list.append(np.nan)

    no_crossing_frac = no_crossing_count / n_boot
    frac_resamples_dropped = n_resamples_dropped / n_boot
    valid_mi = [m for m in boot_mi_list if not np.isnan(m)]
    if len(valid_mi) > 0:
        mi_ci_pooled = (float(np.percentile(valid_mi, 2.5)), float(np.percentile(valid_mi, 97.5)))
    else:
        mi_ci_pooled = (np.nan, np.nan)

    # R_hat at S=1 point estimate and 95% CI
    R_hat_s1_point = float(core["R_hat_S"][-1]) if core["R_hat_S"] is not None else np.nan
    if len(boot_rhat_s1_list) >= 2:
        rhat_s1_ci = (
            float(np.percentile(boot_rhat_s1_list, 2.5)),
            float(np.percentile(boot_rhat_s1_list, 97.5)),
        )
    else:
        rhat_s1_ci = (np.nan, np.nan)

    # Significant crossing: first S where upper bootstrap CI of U(S) < 0
    u_ci_upper = np.percentile(boot_u_matrix, 97.5, axis=0)
    sig_cross_idx = np.where(u_ci_upper < 0)[0]
    sig_s_star = float(S_grid[sig_cross_idx[0]]) if len(sig_cross_idx) > 0 else None

    # Hierarchical bootstrap (resample seeds, then items within seeds)
    hier_mi_list = []
    n_seeds = len(seed_groups)
    seed_item_lists = [sg["items"] for sg in seed_groups]

    for b in range(n_boot):
        b_seed_indices = rng.integers(0, n_seeds, size=n_seeds)
        b_items = []
        for s_idx in b_seed_indices:
            s_items = seed_item_lists[s_idx]
            sub_idx = rng.integers(0, len(s_items), size=len(s_items))
            for i in sub_idx:
                b_items.append(s_items[i])

        # Evaluate on b_items
        b_sub = compute_metrics_for_items(b_items, S_grid, null_kind="removed")
        if b_sub and b_sub["S_star_raw"] is not None and b_sub["MI"] is not None:
            hier_mi_list.append(b_sub["MI"])
        else:
            hier_mi_list.append(np.nan)

    valid_hier_mi = [m for m in hier_mi_list if not np.isnan(m)]
    if len(valid_hier_mi) > 0:
        mi_ci_hier = (float(np.percentile(valid_hier_mi, 2.5)), float(np.percentile(valid_hier_mi, 97.5)))
    else:
        mi_ci_hier = (np.nan, np.nan)

    # Formatting S* edge rule
    s_star_val = core["S_star_raw"]
    s_star_display = f"{s_star_val:.3f}" if s_star_val is not None else "None"
    if s_star_val is not None and math.isclose(s_star_val, S_grid[-1], abs_tol=1e-5):
        s_star_display = f"{s_star_val:.3f} (edge)"

    sig_s_star_display = f"{sig_s_star:.3f}" if sig_s_star is not None else "None"

    return {
        "model_name": model_name,
        "corruption": corruption,
        "n_seeds": n_seeds,
        "seeds": seeds,
        "n_items_total": n_items_total,
        "S_grid": S_grid,
        "U_S": core["U_S"],
        "R_S": core["R_S"],
        "R_hat_S": core["R_hat_S"],
        "clean_acc": core["clean_acc"],
        "acc_s1": core["acc_s1"],
        "null_removed_acc": core["acc_null"],
        "null_black_acc": black_acc,
        "swapped_acc": swapped_acc,
        "mean_letter_mass": mean_letter_mass,
        "S_star": s_star_val,
        "S_star_display": s_star_display,
        "sig_S_star": sig_s_star,
        "sig_S_star_display": sig_s_star_display,
        "MI": core["MI"],
        "MI_ci_pooled_low": mi_ci_pooled[0],
        "MI_ci_pooled_high": mi_ci_pooled[1],
        "MI_ci_hier_low": mi_ci_hier[0],
        "MI_ci_hier_high": mi_ci_hier[1],
        "R_hat_s1": R_hat_s1_point,
        "R_hat_s1_ci_low": rhat_s1_ci[0],
        "R_hat_s1_ci_high": rhat_s1_ci[1],
        "no_crossing_frac": no_crossing_frac,
        "n_resamples_dropped": n_resamples_dropped,
        "frac_resamples_dropped": frac_resamples_dropped,
        "n_dropped_no_crossing": n_dropped_no_crossing,
        "n_dropped_edge_crossing": n_dropped_edge_crossing,
        "n_dropped_r0_zero": n_dropped_r0_zero,
        "n_dropped_other": n_dropped_other,
        "all_items": all_items,
    }


def analyze_null_sensitivity(
    seed_groups: List[Dict[str, Any]],
    available_nulls: List[str],
) -> List[Dict[str, Any]]:
    """Compute S* and MI across different null choices."""
    all_items = []
    for sg in seed_groups:
        all_items.extend(sg["items"])
    S_grid = seed_groups[0]["S_grid"]
    model_name = seed_groups[0]["model_name"]
    corruption = seed_groups[0]["corruption"]

    rows = []
    for null_k in available_nulls:
        res = compute_metrics_for_items(all_items, S_grid, null_kind=null_k)
        if not res:
            continue
        rows.append({
            "model_name": model_name,
            "corruption": corruption,
            "null_baseline": null_k,
            "null_acc": res["acc_null"],
            "S_star": res["S_star_raw"],
            "MI": res["MI"],
        })
    return rows


def generate_plots(
    summaries: List[Dict[str, Any]],
    fig_dir: Path,
):
    """Generate publication-quality vector PDF figures with colorblind-safe styling."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN] matplotlib not available; skipping PDF figure generation.")
        return

    fig_dir.mkdir(parents=True, exist_ok=True)
    unique_models = sorted(list(set(s["model_name"] for s in summaries)))
    unique_corruptions = sorted(list(set(s["corruption"] for s in summaries)))

    # 1. Utility Curves: U(S) per model
    n_models = len(unique_models)
    fig, axes = plt.subplots(1, n_models, figsize=(5.5 * n_models, 4.5), squeeze=False, sharey=True)

    for col_idx, mname in enumerate(unique_models):
        ax = axes[0, col_idx]
        model_sums = [s for s in summaries if s["model_name"] == mname]
        for c_idx, s in enumerate(model_sums):
            corr = s["corruption"]
            color = CB_COLORS[c_idx % len(CB_COLORS)]
            style = CORRUPTION_STYLES.get(corr, {"linestyle": "-", "marker": "o"})
            ax.plot(
                s["S_grid"],
                s["U_S"],
                label=corr,
                color=color,
                linestyle=style["linestyle"],
                marker=style["marker"],
                linewidth=2.0,
                markersize=5,
            )
        ax.axhline(0, color="gray", linestyle=":", linewidth=1.2, label="Zero Utility")
        ax.set_title(f"{mname.split('/')[-1]}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Severity $S$", fontsize=11)
        if col_idx == 0:
            ax.set_ylabel("Multimodal Utility $U(S)$", fontsize=11)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(frameon=True, fontsize=9)

    plt.tight_layout()
    fig.savefig(fig_dir / "vlm_utility_curves.pdf", format="pdf", bbox_inches="tight")
    plt.close(fig)

    # 2. Normalized Reliance Curves: R_hat(S) per model
    fig, axes = plt.subplots(1, n_models, figsize=(5.5 * n_models, 4.5), squeeze=False, sharey=True)
    for col_idx, mname in enumerate(unique_models):
        ax = axes[0, col_idx]
        model_sums = [s for s in summaries if s["model_name"] == mname]
        for c_idx, s in enumerate(model_sums):
            corr = s["corruption"]
            color = CB_COLORS[c_idx % len(CB_COLORS)]
            style = CORRUPTION_STYLES.get(corr, {"linestyle": "-", "marker": "s"})
            ax.plot(
                s["S_grid"],
                s["R_hat_S"],
                label=corr,
                color=color,
                linestyle=style["linestyle"],
                marker=style["marker"],
                linewidth=2.0,
                markersize=5,
            )
        ax.set_title(f"{mname.split('/')[-1]}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Severity $S$", fontsize=11)
        if col_idx == 0:
            ax.set_ylabel(r"Normalized Reliance $\hat{R}(S)$", fontsize=11)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(frameon=True, fontsize=9)

    plt.tight_layout()
    fig.savefig(fig_dir / "vlm_reliance_curves.pdf", format="pdf", bbox_inches="tight")
    plt.close(fig)

    # 3. Bar chart of MI per model x corruption with CIs
    fig, ax = plt.subplots(figsize=(max(6, len(summaries) * 1.2), 4.5))
    bar_labels = []
    bar_vals = []
    bar_errs_low = []
    bar_errs_high = []
    bar_colors = []

    for idx, s in enumerate(summaries):
        lbl = f"{s['model_name'].split('/')[-1]}\n({s['corruption']})"
        mi = s["MI"] if s["MI"] is not None and not np.isnan(s["MI"]) else 0.0
        ci_l = s["MI_ci_pooled_low"] if not np.isnan(s["MI_ci_pooled_low"]) else mi
        ci_h = s["MI_ci_pooled_high"] if not np.isnan(s["MI_ci_pooled_high"]) else mi

        bar_labels.append(lbl)
        bar_vals.append(mi)
        bar_errs_low.append(max(0.0, mi - ci_l))
        bar_errs_high.append(max(0.0, ci_h - mi))
        bar_colors.append(CB_COLORS[idx % len(CB_COLORS)])

    x_pos = np.arange(len(bar_labels))
    errs = np.array([bar_errs_low, bar_errs_high])
    ax.bar(x_pos, bar_vals, yerr=errs, capsize=4, color=bar_colors, alpha=0.85, edgecolor="black")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(bar_labels, fontsize=9, rotation=15)
    ax.set_ylabel("Misalignment Index (MI)", fontsize=11)
    ax.set_title("Misalignment Index across Models and Corruptions (95% CI)", fontsize=12, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.6)

    plt.tight_layout()
    fig.savefig(fig_dir / "vlm_mi_bar.pdf", format="pdf", bbox_inches="tight")
    plt.close(fig)


def export_latex_table(df: pd.DataFrame, out_path: Path, caption: str, label: str):
    """Export clean booktabs LaTeX table."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        "\\small",
    ]
    # Format table using pandas to_latex with booktabs
    latex_content = df.to_latex(index=False, escape=False)
    lines.append(latex_content.strip())
    lines.append("\\end{table*}")

    out_path.write_text("\n".join(lines), encoding="utf-8")


def run_full_analysis(
    raw_dir: Path,
    out_dir: Path,
    table_dir: Path,
    fig_dir: Path,
    n_boot: int = 1000,
    seed: int = 42,
):
    """Main execution orchestrating all analysis steps."""
    out_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    print(f"Scanning raw shards from: {raw_dir}")
    shards_by_key = load_shards_grouped(raw_dir)
    if not shards_by_key:
        print(f"[WARN] No valid evaluation shards found in {raw_dir}.")
        return

    # Group by (model_name, corruption)
    model_corr_groups: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for (mname, corr, s), data in shards_by_key.items():
        mc_key = (mname, corr)
        model_corr_groups.setdefault(mc_key, []).append(data)

    print(f"Found {len(model_corr_groups)} unique (model, corruption) condition(s).")

    summary_rows = []
    summaries = []
    null_sensitivity_rows = []

    for (mname, corr), seed_groups in sorted(model_corr_groups.items()):
        print(f"Analyzing {mname} | {corr} ({len(seed_groups)} seed(s))...")
        res = analyze_model_corruption_all_seeds(
            mname, corr, seed_groups, n_boot=n_boot, seed=seed
        )
        if not res:
            continue
        summaries.append(res)

        point_mi = res["MI"]
        is_mi_defined = point_mi is not None and not np.isnan(point_mi)

        mi_str = f"{point_mi:.4f}" if is_mi_defined else "undefined"
        mi_ci_p_str = (
            f"[{res['MI_ci_pooled_low']:.4f}, {res['MI_ci_pooled_high']:.4f}]"
            if is_mi_defined and not np.isnan(res['MI_ci_pooled_low'])
            else "None"
        )
        mi_ci_h_str = (
            f"[{res['MI_ci_hier_low']:.4f}, {res['MI_ci_hier_high']:.4f}]"
            if is_mi_defined and not np.isnan(res['MI_ci_hier_low'])
            else "None"
        )

        # Conditional MI range across resamples that found a usable crossing
        n_usable = n_boot - res["n_resamples_dropped"]
        frac_usable = n_usable / n_boot
        if not np.isnan(res["MI_ci_pooled_low"]):
            cond_mi_range_str = f"[{res['MI_ci_pooled_low']:.4f}, {res['MI_ci_pooled_high']:.4f}]"
        else:
            cond_mi_range_str = "None"

        if res["frac_resamples_dropped"] > 0.05:
            print(
                f"[WARN] High fraction of bootstrap resamples dropped: "
                f"{res['frac_resamples_dropped']:.2%} ({res['n_resamples_dropped']}/{n_boot}) "
                f"for {mname} | {corr}. "
                f"Cause breakdown: no_crossing={res['n_dropped_no_crossing']}, "
                f"edge_crossing={res['n_dropped_edge_crossing']}, "
                f"r0_zero={res['n_dropped_r0_zero']}, "
                f"other={res['n_dropped_other']}."
            )

        row = {
            "model_name": mname,
            "corruption": corr,
            "n_seeds": res["n_seeds"],
            "n_items": res["n_items_total"],
            "clean_acc": round(res["clean_acc"], 4),
            "acc_s1": round(res["acc_s1"], 4),
            "acc_drop": round(res["clean_acc"] - res["acc_s1"], 4),
            "null_removed_acc": round(res["null_removed_acc"], 4),
            "null_black_acc": round(res["null_black_acc"], 4) if res["null_black_acc"] is not None else np.nan,
            "swapped_acc": round(res["swapped_acc"], 4) if res["swapped_acc"] is not None else np.nan,
            "mean_letter_mass": round(res["mean_letter_mass"], 4),
            "S_star": res["S_star_display"],
            "sig_S_star": res["sig_S_star_display"],
            "MI": mi_str,
            "MI_ci_pooled": mi_ci_p_str,
            "MI_ci_hierarchical": mi_ci_h_str,
            "conditional_mi_range": cond_mi_range_str,
            "frac_usable_resamples": round(frac_usable, 4),
            "R_hat_s1": round(res["R_hat_s1"], 4) if not np.isnan(res["R_hat_s1"]) else np.nan,
            "R_hat_s1_ci": (
                f"[{res['R_hat_s1_ci_low']:.4f}, {res['R_hat_s1_ci_high']:.4f}]"
                if not np.isnan(res["R_hat_s1_ci_low"])
                else "None"
            ),
            # Renamed from n/frac_resamples_dropped to no_usable_crossing_*
            "no_crossing_frac": round(res["no_crossing_frac"], 4),
            "n_no_usable_crossing": res["n_resamples_dropped"],
            "frac_no_usable_crossing": round(res["frac_resamples_dropped"], 4),
            "n_no_usable_crossing_no_cross": res["n_dropped_no_crossing"],
            "n_no_usable_crossing_edge": res["n_dropped_edge_crossing"],
            "n_no_usable_crossing_r0_zero": res["n_dropped_r0_zero"],
            "n_no_usable_crossing_other": res["n_dropped_other"],
        }
        summary_rows.append(row)

        # Inspect extra nulls if present
        sample_item = seed_groups[0]["items"][0]
        avail_nulls = list(sample_item.get("null_probs", {}).keys())
        null_sens = analyze_null_sensitivity(seed_groups, avail_nulls)
        null_sensitivity_rows.extend(null_sens)

    # 1. Summary DataFrame & CSV
    summary_df = pd.DataFrame(summary_rows)
    summary_csv_path = out_dir / "vlm_main_summary.csv"
    summary_df.to_csv(summary_csv_path, index=False)
    print(f"Exported summary CSV: {summary_csv_path}")

    # 2. LaTeX Table
    tex_path = table_dir / "vlm_main_summary.tex"
    export_latex_table(
        summary_df,
        tex_path,
        caption="Multimodal Reliance-Utility Benchmark Summary across Models and Corruptions.",
        label="tab:vlm_main_summary",
    )
    print(f"Exported LaTeX table: {tex_path}")

    # 3. Null Sensitivity
    if null_sensitivity_rows:
        sens_df = pd.DataFrame(null_sensitivity_rows)
        sens_csv_path = out_dir / "vlm_main_null_sensitivity.csv"
        sens_df.to_csv(sens_csv_path, index=False)
        sens_tex_path = table_dir / "vlm_main_null_sensitivity.tex"
        export_latex_table(
            sens_df,
            sens_tex_path,
            caption="Sensitivity of Crossing Point $S^*$ and Misalignment Index (MI) to Null Baseline Choice.",
            label="tab:vlm_null_sensitivity",
        )
        print(f"Exported null sensitivity: {sens_csv_path} and {sens_tex_path}")

    # 4. Correlation check: per model, Pearson correlation between MI and plain accuracy drop
    correlations = []
    for mname in sorted(list(set(s["model_name"] for s in summaries))):
        m_sums = [s for s in summaries if s["model_name"] == mname]
        valid_pairs = []
        for s in m_sums:
            mi_val = s["MI"]
            acc_drop = s["clean_acc"] - s["acc_s1"]
            if mi_val is not None and not np.isnan(mi_val):
                valid_pairs.append((float(mi_val), float(acc_drop)))

        if len(valid_pairs) >= 2:
            mis, drops = zip(*valid_pairs)
            # If all values identical (e.g. synthetic stub), correlation is NaN
            if np.std(mis) > 1e-8 and np.std(drops) > 1e-8:
                r_val, p_val = stats.pearsonr(mis, drops)
            else:
                r_val, p_val = 0.0, 1.0
            correlations.append({
                "model_name": mname,
                "n_corruptions": len(valid_pairs),
                "pearson_r": round(float(r_val), 4),
                "p_value": round(float(p_val), 4),
            })
        else:
            correlations.append({
                "model_name": mname,
                "n_corruptions": len(valid_pairs),
                "pearson_r": np.nan,
                "p_value": np.nan,
            })

    corr_df = pd.DataFrame(correlations)
    corr_csv_path = out_dir / "vlm_main_correlations.csv"
    corr_df.to_csv(corr_csv_path, index=False)
    print(f"Exported correlation check: {corr_csv_path}")

    # 5. Generate Vector PDF plots
    generate_plots(summaries, fig_dir)
    print(f"Generated vector PDF plots in: {fig_dir}")


def main():
    args = parse_args()
    run_full_analysis(
        raw_dir=Path(args.raw_dir),
        out_dir=Path(args.out_dir),
        table_dir=Path(args.table_dir),
        fig_dir=Path(args.fig_dir),
        n_boot=args.n_boot,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
