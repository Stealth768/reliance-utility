"""Generate publication figures for v0.27 (SVMult single-column 117 mm width).

Fig 1: Synthetic benchmark (two panels: U(S) and R_hat(S) for Fixed, Partial, Gated, Oracle)
Fig 2: VLM benchmark (two panels: U(S) and R_hat(S) for Qwen2-VL, SmolVLM, LLaVA-1.5;
       solid = swapped, dashed = Gaussian; no patch)
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

# 117 mm width in inches
WIDTH_INCHES = 117.0 / 25.4  # ~4.606 inches
HEIGHT_INCHES = 2.25

# Global publication style ensuring minimum font >= 8 pt sans-serif
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 8.0,
    "axes.labelsize": 8.5,
    "axes.titlesize": 8.5,
    "xtick.labelsize": 8.0,
    "ytick.labelsize": 8.0,
    "legend.fontsize": 8.0,
    "lines.linewidth": 1.4,
    "lines.markersize": 3.5,
    "figure.autolayout": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def generate_fig1_synthetic():
    raw_synth = REPO_ROOT / "results" / "raw" / "synthetic_v03"
    models = [
        ("FixedModel", "Fixed", "#D55E00", "-", "o"),
        ("PartialGatedModel", "Partial", "#E69F00", "--", "s"),
        ("GatedModel", "Gated", "#0072B2", "-.", "^"),
        ("OracleGatedModel", "Oracle", "#009E73", ":", "d"),
    ]

    fig, (ax_u, ax_r) = plt.subplots(1, 2, figsize=(WIDTH_INCHES, HEIGHT_INCHES))

    S_grid = np.linspace(0.0, 1.0, 21)

    for m_key, m_label, color, ls, marker in models:
        files = [f for f in raw_synth.glob(f"{m_key}_removed_*.json") if "manifest" not in f.name]
        u_list, r_list = [], []
        for f in files:
            d = json.loads(f.read_text(encoding="utf-8"))
            u_list.append(d["U"])
            r_list.append(d["R_hat"])
        mean_u = np.mean(u_list, axis=0)
        mean_r = np.mean(r_list, axis=0)
        u_sm = smooth(mean_u, window=3)

        ax_u.plot(S_grid, u_sm, label=m_label, color=color, linestyle=ls, marker=marker, markevery=3)
        ax_r.plot(S_grid, mean_r, label=m_label, color=color, linestyle=ls, marker=marker, markevery=3)

    # Styling panel 1 (Utility)
    ax_u.axhline(0.0, color="black", linestyle="--", linewidth=0.8, alpha=0.7)
    ax_u.set_xlabel("Severity $S$")
    ax_u.set_ylabel("Utility $U(S)$")
    ax_u.set_xlim(0, 1)
    ax_u.grid(True, linestyle=":", alpha=0.6)
    ax_u.legend(loc="lower left", framealpha=0.85, handlelength=1.8, borderpad=0.3, labelspacing=0.2)

    # Styling panel 2 (Reliance)
    ax_r.axhline(1.0, color="gray", linestyle=":", linewidth=0.8, alpha=0.7)
    ax_r.set_xlabel("Severity $S$")
    ax_r.set_ylabel(r"Reliance $\hat{R}(S)$")
    ax_r.set_xlim(0, 1)
    ax_r.set_ylim(-0.05, 1.85)
    ax_r.grid(True, linestyle=":", alpha=0.6)

    plt.subplots_adjust(left=0.12, right=0.98, top=0.95, bottom=0.19, wspace=0.32)
    out_pdf = FIG_DIR / "fig1_synthetic_two_panel.pdf"
    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"Saved {out_pdf}")


def generate_fig2_vlm():
    model_dirs = {
        "Qwen2-VL": ("vlm_main", "#0072B2", "o"),   # Blue, circle
        "SmolVLM": ("vlm_smol", "#D55E00", "s"),    # Vermilion, square
        "LLaVA-1.5": ("vlm_llava", "#009E73", "^"), # Bluish green, triangle
    }
    corruptions = [
        ("swapblend", "Swapped", "-"),
        ("gaussian", "Gaussian", "--"),
    ]

    fig, (ax_u, ax_r) = plt.subplots(1, 2, figsize=(WIDTH_INCHES, HEIGHT_INCHES))
    rng = np.random.default_rng(42)
    n_boot = 500

    for model_name, (dir_name, color, marker) in model_dirs.items():
        raw_dir = REPO_ROOT / "results" / "raw" / dir_name
        for corr_key, corr_label, ls in corruptions:
            files = sorted([f for f in raw_dir.glob(f"*{corr_key}*.json") if "manifest" not in f.name])
            all_items = []
            s_grid = None
            for f in files:
                d = json.loads(f.read_text(encoding="utf-8"))
                if s_grid is None:
                    s_grid = np.array(d["S_grid"])
                all_items.extend(d["items"])

            n_S = len(s_grid)
            n_items = len(all_items)
            correct_idx = np.array([it["correct_choice_idx"] for it in all_items])
            null_probs = np.array([it["null_probs"]["removed"] for it in all_items])
            probs_S = np.array([[it["probs_per_S"][s] for it in all_items] for s in range(n_S)])

            preds_S = np.argmax(probs_S, axis=-1)
            acc_S = np.mean(preds_S == correct_idx[None, :], axis=1)
            null_preds = np.argmax(null_probs, axis=-1)
            acc_null = float(np.mean(null_preds == correct_idx))

            # Precompute JSD matrix of shape (n_S, n_items)
            U_point = acc_S - acc_null
            jsd_matrix = np.zeros((n_S, n_items))
            for s in range(n_S):
                for i in range(n_items):
                    jsd_matrix[s, i] = jsd(probs_S[s, i], null_probs[i])

            R_point = reliance(np.mean(jsd_matrix, axis=1))
            R_hat_point = normalized_reliance(R_point)

            # Fast vectorized bootstrap for CI band
            boot_U = []
            boot_R_hat = []
            for _ in range(n_boot):
                b_idx = rng.integers(0, n_items, size=n_items)
                b_acc_S = np.mean(preds_S[:, b_idx] == correct_idx[b_idx][None, :], axis=1)
                b_acc_null = float(np.mean(null_preds[b_idx] == correct_idx[b_idx]))
                b_U = b_acc_S - b_acc_null

                b_jsd = np.mean(jsd_matrix[:, b_idx], axis=1)
                b_R = reliance(b_jsd)
                b_Rh = normalized_reliance(b_R)
                boot_U.append(b_U)
                boot_R_hat.append(b_Rh)

            u_lo, u_hi = np.percentile(boot_U, [2.5, 97.5], axis=0)
            r_lo, r_hi = np.percentile(boot_R_hat, [2.5, 97.5], axis=0)

            # Plot U
            lbl = f"{model_name}" if corr_key == "swapblend" else None
            ax_u.plot(s_grid, U_point, color=color, linestyle=ls, marker=marker, markevery=2, label=lbl)
            ax_u.fill_between(s_grid, u_lo, u_hi, color=color, alpha=0.15)

            # Plot R_hat
            ax_r.plot(s_grid, R_hat_point, color=color, linestyle=ls, marker=marker, markevery=2)
            ax_r.fill_between(s_grid, r_lo, r_hi, color=color, alpha=0.15)

    # Style panel 1 (Utility)
    ax_u.axhline(0.0, color="black", linestyle="--", linewidth=0.8, alpha=0.7)
    ax_u.set_xlabel("Severity $S$")
    ax_u.set_ylabel("Utility $U(S)$")
    ax_u.set_xlim(0, 1)
    ax_u.grid(True, linestyle=":", alpha=0.6)
    ax_u.legend(loc="lower left", framealpha=0.85, handlelength=1.5, borderpad=0.3, labelspacing=0.2)

    # Style panel 2 (Reliance)
    ax_r.axhline(0.5, color="gray", linestyle=":", linewidth=0.8, alpha=0.7)
    ax_r.set_xlabel("Severity $S$")
    ax_r.set_ylabel(r"Reliance $\hat{R}(S)$")
    ax_r.set_xlim(0, 1)
    ax_r.set_ylim(0.0, 1.05)
    ax_r.grid(True, linestyle=":", alpha=0.6)

    # Add line style legend to ax_r
    from matplotlib.lines import Line2D
    style_lines = [
        Line2D([0], [0], color="black", linestyle="-", label="Swapped"),
        Line2D([0], [0], color="black", linestyle="--", label="Gaussian"),
    ]
    ax_r.legend(handles=style_lines, loc="upper right", framealpha=0.85, handlelength=1.8, borderpad=0.3, labelspacing=0.2)

    plt.subplots_adjust(left=0.12, right=0.98, top=0.95, bottom=0.19, wspace=0.32)
    out_pdf = FIG_DIR / "fig2_vlm_two_panel.pdf"
    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"Saved {out_pdf}")


if __name__ == "__main__":
    generate_fig1_synthetic()
    generate_fig2_vlm()
