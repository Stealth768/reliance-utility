"""Generate publication-ready figures for synthetic reliance-utility validation."""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Set publication style: clean fonts, high-DPI vector settings
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.titlesize": 14,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

# Colorblind-safe palette (Okabe-Ito inspired) & distinct line styles for grayscale readability
MODEL_STYLES = {
    "FixedModel": {
        "label": "Fixed (Quality-Blind)",
        "color": "#D55E00",  # Vermilion
        "linestyle": "-",
        "marker": "o",
    },
    "PartialGatedModel": {
        "label": "Partial Gated",
        "color": "#E69F00",  # Amber
        "linestyle": "--",
        "marker": "s",
    },
    "GatedModel": {
        "label": "Gated (Quality-Aware)",
        "color": "#0072B2",  # Blue
        "linestyle": "-.",
        "marker": "^",
    },
}


def load_raw_data(raw_dir: Path) -> Dict[str, List[Dict]]:
    """Load raw result JSON files grouped by model name."""
    data_by_model: Dict[str, List[Dict]] = {
        "FixedModel": [],
        "PartialGatedModel": [],
        "GatedModel": [],
    }
    for file in sorted(raw_dir.glob("*.json")):
        if file.name.startswith("manifest") or "run_" in file.name:
            continue
        with open(file, "r", encoding="utf-8") as f:
            entry = json.load(f)
        m_name = entry.get("model_name")
        if m_name in data_by_model:
            data_by_model[m_name].append(entry)
    return data_by_model


def make_figures(
    raw_dir: str = "results/raw/synthetic",
    summary_csv: str = "results/derived/synthetic_summary.csv",
    output_dir: str = "figures",
) -> None:
    """Generate and save Figures 1 to 4 in vector PDF format.

    Parameters
    ----------
    raw_dir : str
        Directory containing raw JSON result files.
    summary_csv : str
        Path to synthetic summary CSV.
    output_dir : str
        Output directory for generated PDF figures.
    """
    raw_path = Path(raw_dir)
    fig_path = Path(output_dir)
    fig_path.mkdir(parents=True, exist_ok=True)

    data_by_model = load_raw_data(raw_path)
    df_summary = pd.read_csv(summary_csv) if Path(summary_csv).exists() else None

    # Common severity grid (21 points)
    S_grid = np.linspace(0.0, 1.0, 21)

    # -------------------------------------------------------------
    # Fig 1: U(S) for all 3 models (with CI bands across seeds)
    # -------------------------------------------------------------
    plt.figure(figsize=(7, 4.8))
    for m_name, style in MODEL_STYLES.items():
        entries = [e for e in data_by_model[m_name] if e.get("null_kind") == "zeros"]
        if not entries:
            continue
        U_curves = np.array([e["U"] for e in entries])
        mean_U = np.mean(U_curves, axis=0)
        std_U = np.std(U_curves, axis=0)
        sem_U = 1.96 * std_U / np.sqrt(max(len(entries), 1))

        plt.plot(
            S_grid,
            mean_U,
            label=style["label"],
            color=style["color"],
            linestyle=style["linestyle"],
            marker=style["marker"],
            markevery=2,
            linewidth=2.0,
        )
        plt.fill_between(
            S_grid,
            mean_U - sem_U,
            mean_U + sem_U,
            color=style["color"],
            alpha=0.18,
        )

    plt.axhline(0.0, color="#333333", linestyle=":", linewidth=1.2, label="Zero Utility (U=0)")
    plt.xlabel("Corruption Severity $S$")
    plt.ylabel("Modality Utility $U(S)$")
    plt.title("Figure 1: Modality Utility Across Degradation Sweeps")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(frameon=True, loc="upper right")
    plt.tight_layout()
    fig1_out = fig_path / "fig1_utility.pdf"
    plt.savefig(fig1_out, bbox_inches="tight")
    plt.close()
    print(f"Saved: {fig1_out}")

    # -------------------------------------------------------------
    # Fig 2: R_hat(S) for all 3 models with S* marked
    # -------------------------------------------------------------
    plt.figure(figsize=(7, 4.8))
    for m_name, style in MODEL_STYLES.items():
        entries = [e for e in data_by_model[m_name] if e.get("null_kind") == "zeros"]
        if not entries:
            continue
        R_hat_curves = np.array([e["R_hat"] for e in entries])
        mean_R_hat = np.mean(R_hat_curves, axis=0)

        # Average S* across entries
        s_stars = [e["S_star"] for e in entries if e.get("S_star") is not None]
        mean_s_star = float(np.mean(s_stars)) if s_stars else None

        line = plt.plot(
            S_grid,
            mean_R_hat,
            label=style["label"],
            color=style["color"],
            linestyle=style["linestyle"],
            marker=style["marker"],
            markevery=2,
            linewidth=2.0,
        )

        if mean_s_star is not None:
            plt.axvline(
                mean_s_star,
                color=style["color"],
                linestyle=":",
                alpha=0.8,
                label=f"{style['label']} $S^*={mean_s_star:.2f}$",
            )

    plt.xlabel("Corruption Severity $S$")
    plt.ylabel(r"Normalized Reliance $\hat{R}(S)$")
    plt.title(r"Figure 2: Normalized Reliance Curve and Crossing Severity $S^*$")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(frameon=True, loc="upper right")
    plt.tight_layout()
    fig2_out = fig_path / "fig2_reliance.pdf"
    plt.savefig(fig2_out, bbox_inches="tight")
    plt.close()
    print(f"Saved: {fig2_out}")

    # -------------------------------------------------------------
    # Fig 3: Reliance-Utility Trajectory (U on x, R_hat on y)
    # -------------------------------------------------------------
    plt.figure(figsize=(7.2, 5.0))
    for m_name, style in MODEL_STYLES.items():
        entries = [e for e in data_by_model[m_name] if e.get("null_kind") == "zeros"]
        if not entries:
            continue
        mean_U = np.mean([e["U"] for e in entries], axis=0)
        mean_R = np.mean([e["R_hat"] for e in entries], axis=0)

        plt.plot(
            mean_U,
            mean_R,
            linestyle=style["linestyle"],
            color=style["color"],
            label=style["label"],
            linewidth=1.8,
            alpha=0.7,
        )
        scatter = plt.scatter(
            mean_U,
            mean_R,
            c=S_grid,
            cmap="viridis",
            marker=style["marker"],
            s=40,
            edgecolors="k",
            linewidths=0.5,
            zorder=4,
        )

    plt.axvline(0.0, color="#333333", linestyle=":", linewidth=1.2, label="Crossing Point $U=0$")
    cbar = plt.colorbar(scatter)
    cbar.set_label("Corruption Severity $S$")
    plt.xlabel("Modality Utility $U(S)$")
    plt.ylabel(r"Normalized Reliance $\hat{R}(S)$")
    plt.title(r"Figure 3: Reliance-Utility Trajectory Traversal")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(frameon=True, loc="lower right")
    plt.tight_layout()
    fig3_out = fig_path / "fig3_trajectory.pdf"
    plt.savefig(fig3_out, bbox_inches="tight")
    plt.close()
    print(f"Saved: {fig3_out}")

    # -------------------------------------------------------------
    # Fig 4: MI Bar Chart with CIs, grouped by null variant
    # -------------------------------------------------------------
    plt.figure(figsize=(7.5, 4.8))
    null_variants = ["zeros", "mean"]
    model_keys = ["FixedModel", "PartialGatedModel", "GatedModel"]

    x = np.arange(len(null_variants))
    bar_width = 0.25

    for idx, m_name in enumerate(model_keys):
        style = MODEL_STYLES[m_name]
        means = []
        errors_lo = []
        errors_hi = []

        for nv in null_variants:
            sub = df_summary[(df_summary["model"] == m_name) & (df_summary["null_kind"] == nv)] if df_summary is not None else pd.DataFrame()
            # If MI is None, treat as NaN
            mi_vals = sub["MI"].dropna().values if not sub.empty else []
            if len(mi_vals) > 0:
                mean_mi = float(np.mean(mi_vals))
                std_mi = float(np.std(mi_vals))
                means.append(mean_mi)
                errors_lo.append(min(std_mi, mean_mi))
                errors_hi.append(min(std_mi, 1.0 - mean_mi))
            else:
                means.append(0.0)
                errors_lo.append(0.0)
                errors_hi.append(0.0)

        offset = (idx - 1) * bar_width
        plt.bar(
            x + offset,
            means,
            width=bar_width,
            yerr=[errors_lo, errors_hi],
            label=style["label"],
            color=style["color"],
            capsize=4,
            edgecolor="black",
            linewidth=0.8,
            alpha=0.85,
        )

    plt.xticks(x, [f"Null: {nv.capitalize()}" for nv in null_variants])
    plt.ylabel("Misalignment Index (MI)")
    plt.ylim(0.0, 1.1)
    plt.axhline(0.2, color="#0072B2", linestyle="--", linewidth=1.0, alpha=0.7, label="Gated Target (< 0.2)")
    plt.axhline(0.7, color="#D55E00", linestyle="--", linewidth=1.0, alpha=0.7, label="Fixed Target (> 0.7)")
    plt.title("Figure 4: Misalignment Index (MI) Comparison by Null Baseline")
    plt.grid(True, axis="y", linestyle="--", alpha=0.5)
    plt.legend(frameon=True, loc="upper right")
    plt.tight_layout()
    fig4_out = fig_path / "fig4_mi_barchart.pdf"
    plt.savefig(fig4_out, bbox_inches="tight")
    plt.close()
    print(f"Saved: {fig4_out}")


if __name__ == "__main__":
    make_figures()
