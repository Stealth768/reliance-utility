#!/usr/bin/env python3
"""Generate publication vector figures for the Extension Phase (SVMult 117 mm width).

Generates:
1. figures/fig3_ai2d_two_panel.pdf:
   - Left: AI2D Utility U(S) across severity S in [0, 1]
   - Right: AI2D Normalized Reliance R_hat(S) across severity S in [0, 1]
2. figures/fig4_open_and_stability.pdf:
   - Left: Open-ended sequence reliance R_hat_open(S) across severity S
   - Right: Donor derangement stability across seeds {0, 11, 12, 13} with 95% bootstrap CIs
"""

import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.metrics import reliance, normalized_reliance, utility, smooth, jsd

FIG_DIR = REPO_ROOT / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
PAPER_FIG_DIR = REPO_ROOT / "paper_extended" / "figures"
PAPER_FIG_DIR.mkdir(parents=True, exist_ok=True)

# 117 mm width in inches (SVMult textwidth)
WIDTH_INCHES = 117.0 / 25.4  # ~4.606 in
HEIGHT_INCHES = 2.15

# Colorblind-safe palette (Wong 2011)
CB_COLORS = {
    "Qwen2-VL-2B": "#0072B2",  # Blue
    "SmolVLM": "#D55E00",      # Vermilion
    "LLaVA-1.5-7B": "#009E73", # Green
}
MARKERS = {
    "Qwen2-VL-2B": "o",
    "SmolVLM": "s",
    "LLaVA-1.5-7B": "^",
}


def load_ai2d_data(model_name: str, corr: str):
    prefix_map = {
        "Qwen2-VL-2B": "ai2d_Qwen2-VL-2B-Instruct",
        "SmolVLM": "ai2d_SmolVLM-Instruct",
        "LLaVA-1.5-7B": "ai2d_llava-1.5-7b-hf",
    }
    pref = prefix_map[model_name]
    ai2d_dir = REPO_ROOT / "results" / "raw" / "vlm_ai2d"
    files = sorted(ai2d_dir.glob(f"{pref}_{corr}_seed0_shard*.json"))
    
    # Sort by shard index
    def get_shard_idx(p):
        bn = p.name
        return int(bn.split("_shard")[-1].replace(".json", "")) if "_shard" in bn else 0
    files.sort(key=get_shard_idx)

    all_items = []
    s_grid = None
    for f in files:
        d = json.loads(f.read_text(encoding="utf-8"))
        if s_grid is None:
            s_grid = np.array(d["S_grid"])
        all_items.extend(d["items"])
    return s_grid, all_items


def load_open_data(model_name: str, corr: str):
    prefix_map = {
        "Qwen2-VL-2B": "open_Qwen2-VL-2B-Instruct",
        "SmolVLM": "open_SmolVLM-Instruct",
        "LLaVA-1.5-7B": "open_llava-1.5-7b-hf",
    }
    pref = prefix_map[model_name]
    open_dir = REPO_ROOT / "results" / "raw" / "vlm_open"
    files = sorted(open_dir.glob(f"{pref}_{corr}_seed0_shard*.json"))
    
    def get_shard_idx(p):
        bn = p.name
        return int(bn.split("_shard")[-1].replace(".json", "")) if "_shard" in bn else 0
    files.sort(key=get_shard_idx)

    all_items = []
    s_grid = None
    for f in files:
        d = json.loads(f.read_text(encoding="utf-8"))
        if s_grid is None:
            s_grid = np.array(d["S_grid"])
        all_items.extend(d["items"])
    return s_grid, all_items


def generate_fig3_ai2d():
    """AI2D: Left panel U(S), Right panel R_hat(S)."""
    fig, (ax_u, ax_r) = plt.subplots(1, 2, figsize=(WIDTH_INCHES, HEIGHT_INCHES))
    rng = np.random.default_rng(42)
    n_boot = 500

    models = ["Qwen2-VL-2B", "SmolVLM", "LLaVA-1.5-7B"]
    corruptions = [("swapblend", "Swapped", "-"), ("gaussian", "Gaussian", "--")]

    for model in models:
        c = CB_COLORS[model]
        m = MARKERS[model]

        for corr, corr_label, ls in corruptions:
            s_grid, items = load_ai2d_data(model, corr)
            n_S = len(s_grid)
            n_items = len(items)

            correct_idx = np.array([it["correct_choice_idx"] for it in items])
            null_probs = np.array([it["null_probs"]["removed"] if "removed" in it["null_probs"] else it["null_probs"]["null_removed"] for it in items])
            probs_S = np.array([[it["probs_per_S"][s] for it in items] for s in range(n_S)])

            preds_S = np.argmax(probs_S, axis=-1)
            acc_S = np.mean(preds_S == correct_idx[None, :], axis=1)
            null_preds = np.argmax(null_probs, axis=-1)
            acc_null = float(np.mean(null_preds == correct_idx))

            U_point = acc_S - acc_null

            jsd_matrix = np.zeros((n_S, n_items))
            for s in range(n_S):
                ps = probs_S[s]
                pn = null_probs
                mid = 0.5 * (ps + pn)
                kl_p = np.sum(ps * np.log2(np.maximum(ps, 1e-12) / np.maximum(mid, 1e-12)), axis=-1)
                kl_n = np.sum(pn * np.log2(np.maximum(pn, 1e-12) / np.maximum(mid, 1e-12)), axis=-1)
                jsd_matrix[s] = np.clip(0.5 * (kl_p + kl_n), 0.0, 1.0)

            R_point = np.mean(jsd_matrix, axis=1)
            R_hat_point = R_point / R_point[0] if R_point[0] > 1e-12 else np.zeros_like(R_point)

            # Vectorized bootstrap
            boot_U = []
            boot_Rh = []
            for _ in range(n_boot):
                b_idx = rng.integers(0, n_items, size=n_items)
                b_acc_S = np.mean(preds_S[:, b_idx] == correct_idx[b_idx][None, :], axis=1)
                b_acc_null = float(np.mean(null_preds[b_idx] == correct_idx[b_idx]))
                boot_U.append(b_acc_S - b_acc_null)

                b_R = np.mean(jsd_matrix[:, b_idx], axis=1)
                b_Rh = b_R / b_R[0] if b_R[0] > 1e-12 else np.zeros_like(b_R)
                boot_Rh.append(b_Rh)

            u_lo, u_hi = np.percentile(boot_U, [2.5, 97.5], axis=0)
            r_lo, r_hi = np.percentile(boot_Rh, [2.5, 97.5], axis=0)

            lbl = f"{model}" if corr == "swapblend" else None
            ax_u.plot(s_grid, U_point, color=c, linestyle=ls, marker=m, markersize=3, label=lbl, alpha=0.9, lw=1.2)
            ax_u.fill_between(s_grid, u_lo, u_hi, color=c, alpha=0.12)

            ax_r.plot(s_grid, R_hat_point, color=c, linestyle=ls, marker=m, markersize=3, label=lbl, alpha=0.9, lw=1.2)
            ax_r.fill_between(s_grid, r_lo, r_hi, color=c, alpha=0.12)

    # Style
    for ax in (ax_u, ax_r):
        ax.set_xlabel("Severity $S$", fontsize=7.5, labelpad=2)
        ax.tick_params(axis="both", labelsize=6.5, pad=1)
        ax.grid(True, linestyle=":", alpha=0.4)
        ax.set_xlim(-0.02, 1.02)

    ax_u.axhline(0.0, color="gray", linestyle=":", lw=0.8)
    ax_u.set_ylabel("Modality Utility $U(S)$", fontsize=7.5, labelpad=2)
    ax_u.set_title("(a) AI2D Diagram Utility", fontsize=8, pad=3)

    ax_r.axhline(0.5, color="gray", linestyle=":", lw=0.8)
    ax_r.set_ylabel(r"Norm. Reliance $\hat{R}(S)$", fontsize=7.5, labelpad=2)
    ax_r.set_title("(b) AI2D Diagram Reliance", fontsize=8, pad=3)
    ax_r.set_ylim(-0.05, 1.15)

    # Legend on utility panel
    # Custom line handles for line style
    from matplotlib.lines import Line2D
    style_lines = [
        Line2D([0], [0], color="black", lw=1.2, ls="-", label="Swapped"),
        Line2D([0], [0], color="black", lw=1.2, ls="--", label="Gaussian"),
    ]
    leg_model = ax_u.legend(fontsize=5.5, loc="lower left", frameon=True, framealpha=0.85, borderpad=0.2)
    ax_r.legend(handles=style_lines, fontsize=5.5, loc="lower left", frameon=True, framealpha=0.85, borderpad=0.2)

    plt.tight_layout(pad=0.5)
    p1 = FIG_DIR / "fig3_ai2d_two_panel.pdf"
    p2 = PAPER_FIG_DIR / "fig3_ai2d_two_panel.pdf"
    fig.savefig(p1, bbox_inches="tight")
    fig.savefig(p2, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {p1} and {p2}")


def generate_fig4_open_and_stability():
    """Fig 4: (a) Open-ended sequence reliance curves, (b) Donor derangement stability."""
    fig, (ax_o, ax_p) = plt.subplots(1, 2, figsize=(WIDTH_INCHES, HEIGHT_INCHES))
    rng = np.random.default_rng(42)
    n_boot = 500

    models = ["Qwen2-VL-2B", "SmolVLM", "LLaVA-1.5-7B"]
    corruptions = [("swapblend", "Swapped", "-"), ("gaussian", "Gaussian", "--")]

    # Panel (a): Open-ended reliance
    for model in models:
        c = CB_COLORS[model]
        m = MARKERS[model]

        for corr, corr_label, ls in corruptions:
            s_grid, items = load_open_data(model, corr)
            n_S = len(s_grid)
            n_items = len(items)

            r0 = np.array([it["R_removed_per_S"][0] for it in items])
            r_S = np.array([[it["R_removed_per_S"][s] for it in items] for s in range(n_S)])

            R_point = np.mean(r_S, axis=1)
            R_hat_point = R_point / R_point[0] if R_point[0] > 1e-12 else np.zeros_like(R_point)

            # Vectorized bootstrap
            boot_Rh = []
            for _ in range(n_boot):
                b_idx = rng.integers(0, n_items, size=n_items)
                b_r0 = np.mean(r0[b_idx])
                b_rs = np.mean(r_S[:, b_idx], axis=1)
                b_rh = b_rs / b_r0 if b_r0 > 1e-12 else np.zeros_like(b_rs)
                boot_Rh.append(b_rh)

            r_lo, r_hi = np.percentile(boot_Rh, [2.5, 97.5], axis=0)

            lbl = f"{model}" if corr == "swapblend" else None
            ax_o.plot(s_grid, R_hat_point, color=c, linestyle=ls, marker=m, markersize=3, label=lbl, alpha=0.9, lw=1.2)
            ax_o.fill_between(s_grid, r_lo, r_hi, color=c, alpha=0.12)

    ax_o.axhline(0.5, color="gray", linestyle=":", lw=0.8)
    ax_o.set_xlabel("Severity $S$", fontsize=7.5, labelpad=2)
    ax_o.set_ylabel(r"Sequence Reliance $\hat{R}_{\text{open}}(S)$", fontsize=7.5, labelpad=2)
    ax_o.set_title("(a) Open-Ended Reliance", fontsize=8, pad=3)
    ax_o.tick_params(axis="both", labelsize=6.5, pad=1)
    ax_o.grid(True, linestyle=":", alpha=0.4)
    ax_o.set_xlim(-0.02, 1.02)
    ax_o.set_ylim(-0.05, 1.45)  # raised: Qwen Gaussian R_hat peaks at ~1.34 around S=0.125
    ax_o.legend(fontsize=5.5, loc="lower left", frameon=True, framealpha=0.85, borderpad=0.2)

    # Panel (b): Pairing Stability across seeds {0, 11, 12, 13}
    # Load pairing data
    pair_dir = REPO_ROOT / "results" / "raw" / "vlm_pairings"
    seeds = [0, 11, 12, 13]
    baseline_seed0 = {
        "Qwen2-VL-2B": (0.487, 0.437, 0.542),
        "SmolVLM": (0.493, 0.446, 0.547),
        "LLaVA-1.5-7B": (0.578, 0.514, 0.653),
    }

    # x offsets per model for readability
    x_offsets = {"Qwen2-VL-2B": -0.2, "SmolVLM": 0.0, "LLaVA-1.5-7B": 0.2}

    for model in models:
        c = CB_COLORS[model]
        m = MARKERS[model]
        off = x_offsets[model]

        m_pat = "Qwen2-VL-2B-Instruct" if model == "Qwen2-VL-2B" else ("SmolVLM-Instruct" if model == "SmolVLM" else "llava-1.5-7b-hf")

        means = []
        err_low = []
        err_high = []

        for s in seeds:
            if s == 0:
                pt, lo, hi = baseline_seed0[model]
            else:
                files = sorted(pair_dir.glob(f"aokvqa_{m_pat}_swapblend_pair{s}_seed0_shard*.json"))
                items = []
                for f in files:
                    d = json.loads(f.read_text(encoding="utf-8"))
                    items.extend(d["items"])
                n = len(items)
                r0 = np.array([it["null_probs"]["removed"] if "removed" in it["null_probs"] else it["null_probs"]["null_removed"] for it in items])
                p0 = np.array([it["probs_per_S"][0] for it in items])
                p1 = np.array([it["probs_per_S"][-1] for it in items])

                # jsd
                def calc_jsd(p_arr, q_arr):
                    mid = 0.5 * (p_arr + q_arr)
                    kl_p = np.sum(p_arr * np.log2(np.maximum(p_arr, 1e-12) / np.maximum(mid, 1e-12)), axis=-1)
                    kl_q = np.sum(q_arr * np.log2(np.maximum(q_arr, 1e-12) / np.maximum(mid, 1e-12)), axis=-1)
                    return np.clip(0.5 * (kl_p + kl_q), 0.0, 1.0)
                j0 = calc_jsd(p0, r0)
                j1 = calc_jsd(p1, r0)

                pt = float(np.mean(j1) / np.mean(j0))
                b_pts = []
                for _ in range(n_boot):
                    b_idx = rng.integers(0, n, size=n)
                    b_pts.append(float(np.mean(j1[b_idx]) / np.mean(j0[b_idx])))
                lo, hi = float(np.percentile(b_pts, 2.5)), float(np.percentile(b_pts, 97.5))

            means.append(pt)
            err_low.append(pt - lo)
            err_high.append(hi - pt)

        x_pos = np.arange(len(seeds)) + off
        ax_p.errorbar(x_pos, means, yerr=[err_low, err_high], fmt=m, color=c, label=model,
                     capsize=2.5, elinewidth=1.0, markersize=4, alpha=0.9)

        # Plot empirical range band per model
        ax_p.plot(x_pos, means, color=c, lw=0.8, alpha=0.5, linestyle=":")

    ax_p.set_xticks(np.arange(len(seeds)))
    ax_p.set_xticklabels([f"Seed {s}" for s in seeds], fontsize=6.5)
    ax_p.set_ylabel(r"Terminal Reliance $\hat{R}(1)$", fontsize=7.5, labelpad=2)
    ax_p.set_title("(b) Donor Derangement Stability", fontsize=8, pad=3)
    ax_p.tick_params(axis="both", labelsize=6.5, pad=1)
    ax_p.grid(True, linestyle=":", alpha=0.4)
    ax_p.set_ylim(0.35, 0.75)
    ax_p.axhline(0.5, color="gray", linestyle=":", lw=0.8)
    ax_p.legend(fontsize=5.5, loc="lower right", frameon=True, framealpha=0.85, borderpad=0.2)

    plt.tight_layout(pad=0.5)
    p1 = FIG_DIR / "fig4_open_and_stability.pdf"
    p2 = PAPER_FIG_DIR / "fig4_open_and_stability.pdf"
    fig.savefig(p1, bbox_inches="tight")
    fig.savefig(p2, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {p1} and {p2}")


if __name__ == "__main__":
    generate_fig3_ai2d()
    generate_fig4_open_and_stability()
