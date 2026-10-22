r"""Generate all LaTeX booktabs tables for the paper results section.

Outputs to tables/:
1. headline_main.tex             (\label{tab:headline})
2. paired_swap_vs_gauss.tex      (\label{tab:paired})
3. headline_replication.tex      (\label{tab:replication})
4. criterion_status_compact.tex  (\label{tab:criterion})
5. null_sensitivity.tex          (\label{tab:nullsens})
6. baseline_metrics.tex          (\label{tab:baseline})
7. synthetic_summary.tex         (\label{tab:synthetic_summary})

Also writes derived CSV:
- results/derived/paired_swap_vs_gauss.csv

All tables state sample size, B=1000, and seed 42 in captions.
All tables use \fittable and 3-decimal "x (lo, hi)" interval formatting.
No table* anywhere.
"""

import json
from pathlib import Path
import re
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_ci_bounds(ci_str: str) -> Tuple[Optional[float], Optional[float]]:
    """Parse lower and upper bounds from a CI string like '[0.4366, 0.5422]'."""
    if not isinstance(ci_str, str):
        return None, None
    m = re.search(r"\[([+-]?\d+\.?\d*),\s*([+-]?\d+\.?\d*)\]", ci_str)
    if m:
        return float(m.group(1)), float(m.group(2))
    return None, None


def fmt_num(val: Optional[float], signed: bool = False, decimals: int = 3) -> str:
    """Format single number with specified decimals."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "N/A"
    sign_fmt = "+" if signed else ""
    return f"{val:{sign_fmt}.{decimals}f}"


def fmt_interval(
    val: Optional[float],
    lo: Optional[float] = None,
    hi: Optional[float] = None,
    signed: bool = False,
    decimals: int = 3,
) -> str:
    """Format number with optional (lo, hi) interval in one cell: 'x (lo, hi)'."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "N/A"
    sign_fmt = "+" if signed else ""
    v_str = f"{val:{sign_fmt}.{decimals}f}"
    if lo is not None and hi is not None and not (np.isnan(lo) or np.isnan(hi)):
        lo_str = f"{lo:{sign_fmt}.{decimals}f}"
        hi_str = f"{hi:{sign_fmt}.{decimals}f}"
        return f"{v_str} ({lo_str}, {hi_str})"
    return v_str


def format_model_name(raw_name: str) -> str:
    if "Qwen" in raw_name:
        return "Qwen2-VL-2B"
    elif "Smol" in raw_name:
        return "SmolVLM-Instruct"
    elif "llava" in raw_name.lower():
        return "LLaVA-1.5-7B"
    return raw_name


def format_corruption(corr: str) -> str:
    mapping = {
        "swapblend": "Swapped (Swapblend)",
        "gaussian": "Gaussian Blur",
        "patch": "Patch Occlusion",
    }
    return mapping.get(corr.lower(), corr.capitalize())


def calc_jsd(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    m = 0.5 * (p + q)
    kl_p = np.sum(p * np.log2(np.maximum(p, 1e-12) / np.maximum(m, 1e-12)), axis=-1)
    kl_q = np.sum(q * np.log2(np.maximum(q, 1e-12) / np.maximum(m, 1e-12)), axis=-1)
    return np.clip(0.5 * (kl_p + kl_q), 0, 1)


def generate_headline_main(df_crit: pd.DataFrame, out_path: Path):
    """Generate Table 1: headline_main.tex (tab:headline)."""
    sub = df_crit[(df_crit["sample"] == "main") & (df_crit["null_baseline"] == "removed")].copy()

    model_order = ["Qwen2-VL-2B-Instruct", "SmolVLM-Instruct", "llava-1.5-7b-hf"]
    corr_order = ["swapblend", "gaussian", "patch"]

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Main sample headline results ($N=500$, bootstrap $B=1000$, seed 42). Terminal normalized reliance $\hat{R}(1)$ and terminal utility $U(1)$ across models and corruptions under the removed-image null baseline.}",
        r"\label{tab:headline}",
        r"\fittable{%",
        r"\begin{tabular}{llcccc}",
        r"\toprule",
        r"\textbf{Model} & \textbf{Corruption} & \textbf{Clean Acc} & \textbf{Null Acc} & $\hat{R}(1)$ & $U(1)$ \\",
        r"\midrule",
    ]

    for m in model_order:
        m_rows = sub[sub["model"] == m]
        m_display = format_model_name(m)
        first_row = True
        for c in corr_order:
            row_match = m_rows[m_rows["corruption"] == c]
            if row_match.empty:
                continue
            r = row_match.iloc[0]
            clean_acc = fmt_num(r["clean_acc"])
            null_acc = fmt_num(r["null_acc"])
            r_lo, r_hi = parse_ci_bounds(str(r["R_hat_1_ci_str"]))
            rhat = fmt_interval(r["R_hat_1"], r_lo, r_hi)
            u_lo, u_hi = parse_ci_bounds(str(r["U_1_ci_str"]))
            u1 = fmt_interval(r["U_1"], u_lo, u_hi, signed=True)
            corr_display = format_corruption(c)
            prefix = f"\\textbf{{{m_display}}}" if first_row else ""
            first_row = False
            lines.append(f"{prefix} & {corr_display} & {clean_acc} & {null_acc} & {rhat} & {u1} \\\\")
        lines.append(r"\midrule" if m != model_order[-1] else r"\bottomrule")

    lines.extend([
        r"\end{tabular}}",
        r"\end{table}",
    ])
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[OK] Wrote {out_path}")


def generate_paired_swap_vs_gauss(out_table: Path, out_csv: Path):
    """Generate Table 2: paired_swap_vs_gauss.tex (tab:paired) and paired_swap_vs_gauss.csv."""
    models_dir = {
        "Qwen2-VL-2B-Instruct": REPO_ROOT / "results" / "raw" / "vlm_main",
        "SmolVLM-Instruct": REPO_ROOT / "results" / "raw" / "vlm_smol",
        "llava-1.5-7b-hf": REPO_ROOT / "results" / "raw" / "vlm_llava",
    }

    records = []
    for m_name, raw_dir in models_dir.items():
        swap_items = {}
        for f in raw_dir.glob("*swapblend*.json"):
            if "manifest" in f.name:
                continue
            d = json.loads(f.read_text(encoding="utf-8"))
            for it in d["items"]:
                swap_items[str(it["question_id"])] = it

        gauss_items = {}
        for f in raw_dir.glob("*gaussian*.json"):
            if "manifest" in f.name:
                continue
            d = json.loads(f.read_text(encoding="utf-8"))
            for it in d["items"]:
                gauss_items[str(it["question_id"])] = it

        qids = sorted(list(set(swap_items.keys()) & set(gauss_items.keys())))
        N = len(qids)

        p0_swap = np.array([swap_items[q]["probs_per_S"][0] for q in qids])
        p1_swap = np.array([swap_items[q]["probs_per_S"][-1] for q in qids])
        pnull_swap = np.array([swap_items[q]["null_probs"]["removed"] for q in qids])
        d0_swap = calc_jsd(p0_swap, pnull_swap)
        d1_swap = calc_jsd(p1_swap, pnull_swap)
        rhat_swap = float(np.mean(d1_swap) / np.mean(d0_swap))

        p0_gauss = np.array([gauss_items[q]["probs_per_S"][0] for q in qids])
        p1_gauss = np.array([gauss_items[q]["probs_per_S"][-1] for q in qids])
        pnull_gauss = np.array([gauss_items[q]["null_probs"]["removed"] for q in qids])
        d0_gauss = calc_jsd(p0_gauss, pnull_gauss)
        d1_gauss = calc_jsd(p1_gauss, pnull_gauss)
        rhat_gauss = float(np.mean(d1_gauss) / np.mean(d0_gauss))

        pt_diff = rhat_swap - rhat_gauss

        rng = np.random.default_rng(42)
        boot_diff = np.zeros(1000)
        for b in range(1000):
            idx = rng.integers(0, N, size=N)
            b_rhat_swap = np.mean(d1_swap[idx]) / np.mean(d0_swap[idx])
            b_rhat_gauss = np.mean(d1_gauss[idx]) / np.mean(d0_gauss[idx])
            boot_diff[b] = b_rhat_swap - b_rhat_gauss

        ci_lo = float(np.percentile(boot_diff, 2.5))
        ci_hi = float(np.percentile(boot_diff, 97.5))
        zero_excluded = bool(ci_lo > 0 or ci_hi < 0)

        records.append({
            "model": m_name,
            "n_items": N,
            "rhat_swap": rhat_swap,
            "rhat_gauss": rhat_gauss,
            "delta_rhat": pt_diff,
            "ci_lo": ci_lo,
            "ci_hi": ci_hi,
            "zero_excluded": zero_excluded,
        })

    df_paired = pd.DataFrame(records)
    df_paired.to_csv(out_csv, index=False)
    print(f"[OK] Wrote {out_csv}")

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Within-model paired differences in terminal normalized reliance $\Delta \hat{R}(1) = \hat{R}(1)_{\text{swap}} - \hat{R}(1)_{\text{gauss}}$ on the main sample ($N=500$, bootstrap $B=1000$, seed 42).}",
        r"\label{tab:paired}",
        r"\fittable{%",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"\textbf{Model} & $\hat{R}(1)_{\text{swap}}$ & $\hat{R}(1)_{\text{gauss}}$ & $\Delta \hat{R}(1)$ & \textbf{Zero Excluded?} \\",
        r"\midrule",
    ]
    for r in records:
        m_display = format_model_name(r["model"])
        swap_str = fmt_num(r["rhat_swap"])
        gauss_str = fmt_num(r["rhat_gauss"])
        diff_str = fmt_interval(r["delta_rhat"], r["ci_lo"], r["ci_hi"], signed=True)
        excl_str = "Yes" if r["zero_excluded"] else "No"
        lines.append(f"{m_display} & {swap_str} & {gauss_str} & {diff_str} & {excl_str} \\\\")
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}}",
        r"\end{table}",
    ])
    out_table.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[OK] Wrote {out_table}")


def generate_headline_replication(df_crit: pd.DataFrame, out_path: Path):
    """Generate Table 3: headline_replication.tex (tab:replication)."""
    sub = df_crit[(df_crit["sample"] == "replication") & (df_crit["null_baseline"] == "removed")].copy()

    model_order = ["Qwen2-VL-2B-Instruct", "SmolVLM-Instruct", "llava-1.5-7b-hf"]
    corr_order = ["swapblend", "gaussian"]

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Replication sample headline results on held-out questions ($N=445$, bootstrap $B=1000$, seed 42). Terminal normalized reliance $\hat{R}(1)$ and terminal utility $U(1)$ across models and corruptions under the removed-image null baseline.}",
        r"\label{tab:replication}",
        r"\fittable{%",
        r"\begin{tabular}{llcccc}",
        r"\toprule",
        r"\textbf{Model} & \textbf{Corruption} & \textbf{Clean Acc} & \textbf{Null Acc} & $\hat{R}(1)$ & $U(1)$ \\",
        r"\midrule",
    ]

    for m in model_order:
        m_rows = sub[sub["model"] == m]
        m_display = format_model_name(m)
        first_row = True
        for c in corr_order:
            row_match = m_rows[m_rows["corruption"] == c]
            if row_match.empty:
                continue
            r = row_match.iloc[0]
            clean_acc = fmt_num(r["clean_acc"])
            null_acc = fmt_num(r["null_acc"])
            r_lo, r_hi = parse_ci_bounds(str(r["R_hat_1_ci_str"]))
            rhat = fmt_interval(r["R_hat_1"], r_lo, r_hi)
            u_lo, u_hi = parse_ci_bounds(str(r["U_1_ci_str"]))
            u1 = fmt_interval(r["U_1"], u_lo, u_hi, signed=True)
            corr_display = format_corruption(c)
            prefix = f"\\textbf{{{m_display}}}" if first_row else ""
            first_row = False
            lines.append(f"{prefix} & {corr_display} & {clean_acc} & {null_acc} & {rhat} & {u1} \\\\")
        lines.append(r"\midrule" if m != model_order[-1] else r"\bottomrule")

    lines.extend([
        r"\end{tabular}}",
        r"\end{table}",
    ])
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[OK] Wrote {out_path}")


def generate_criterion_status_compact(df_crit: pd.DataFrame, out_path: Path):
    """Generate Table 4: criterion_status_compact.tex (tab:criterion)."""
    model_order = ["Qwen2-VL-2B-Instruct", "SmolVLM-Instruct", "llava-1.5-7b-hf"]
    corr_order = ["swapblend", "gaussian", "patch"]
    cols_order = [
        ("main", "removed"),
        ("main", "black"),
        ("replication", "removed"),
        ("replication", "black"),
        ("pooled", "removed"),
        ("pooled", "black"),
    ]

    def get_cell_text(sample_val, null_val, m_val, c_val):
        match = df_crit[
            (df_crit["model"] == m_val) &
            (df_crit["corruption"] == c_val) &
            (df_crit["sample"] == sample_val) &
            (df_crit["null_baseline"] == null_val)
        ]
        if match.empty:
            return "--"
        r = match.iloc[0]
        u1_sig = bool(r["u1_significant"])
        rhat_status = str(r["R_hat_1_status"])
        met = bool(r["criterion_met"])

        if met:
            return r"\textbf{Pass}"
        elif u1_sig and rhat_status != "Strictly > 0.50":
            return r"Fail (lower CI of $\hat{R} \le 0.5$)"
        elif not u1_sig and rhat_status == "Strictly > 0.50":
            return r"Fail ($U \ge 0$)"
        else:
            return r"Fail (both)"

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Criterion status across models, corruptions, sample sizes ($N=500$ main, $N=445$ replication, $N=945$ pooled), and null baselines (bootstrap $B=1000$, seed 42). The confidence interval operationalization (upper bound of $U(1) < 0$ and lower bound of $\hat{R}(1) > 0.50$) is \emph{post hoc}. Gaussian blur and patch occlusion rows are included for completeness. Pass indicates both requirements are satisfied simultaneously.}",
        r"\label{tab:criterion}",
        r"\fittable{%",
        r"\begin{tabular}{llcccccc}",
        r"\toprule",
        r" & & \multicolumn{2}{c}{\textbf{Main ($N=500$)}} & \multicolumn{2}{c}{\textbf{Replication ($N=445$)}} & \multicolumn{2}{c}{\textbf{Pooled ($N=945$)}} \\",
        r"\cmidrule(lr){3-4} \cmidrule(lr){5-6} \cmidrule(lr){7-8}",
        r"\textbf{Model} & \textbf{Corruption} & \textbf{Removed} & \textbf{Black} & \textbf{Removed} & \textbf{Black} & \textbf{Removed} & \textbf{Black} \\",
        r"\midrule",
    ]

    for m in model_order:
        m_display = format_model_name(m)
        first_row = True
        for c in corr_order:
            prefix = f"\\textbf{{{m_display}}}" if first_row else ""
            first_row = False
            corr_display = format_corruption(c)
            cells = [get_cell_text(s, nb, m, c) for (s, nb) in cols_order]
            row_str = f"{prefix} & {corr_display} & " + " & ".join(cells) + r" \\"
            lines.append(row_str)
        lines.append(r"\midrule" if m != model_order[-1] else r"\bottomrule")

    lines.extend([
        r"\end{tabular}}",
        r"\end{table}",
    ])
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[OK] Wrote {out_path}")


def compute_full_sample_floor_jsd() -> Dict[str, str]:
    """Compute full-sample (all N=500 main items) floor JSD per model formatted with 3 decimals."""
    models_dir = {
        "Qwen2-VL-2B": REPO_ROOT / "results" / "raw" / "vlm_main",
        "SmolVLM-Instruct": REPO_ROOT / "results" / "raw" / "vlm_smol",
        "LLaVA-1.5-7B": REPO_ROOT / "results" / "raw" / "vlm_llava",
    }
    floor_res = {}
    for m_short, raw_dir in models_dir.items():
        shards = [f for f in raw_dir.glob("*swapblend*.json") if "manifest" not in f.name]
        items = []
        seen = set()
        for s in sorted(shards):
            d = json.loads(s.read_text(encoding="utf-8"))
            for it in d["items"]:
                qid = str(it["question_id"])
                if qid not in seen:
                    seen.add(qid)
                    items.append(it)
        N = len(items)
        p_blk = np.array([it["null_probs"]["black"] for it in items])
        p_rem = np.array([it["null_probs"]["removed"] for it in items])
        floor_jsd = calc_jsd(p_blk, p_rem)
        mean_val = float(np.mean(floor_jsd))

        rng = np.random.default_rng(42)
        boot = [float(np.mean(floor_jsd[rng.integers(0, N, size=N)])) for _ in range(1000)]
        ci_lo = float(np.percentile(boot, 2.5))
        ci_hi = float(np.percentile(boot, 97.5))
        floor_res[m_short] = fmt_interval(mean_val, ci_lo, ci_hi)
    return floor_res


def generate_null_sensitivity(df_crit: pd.DataFrame, out_path: Path):
    """Generate Table 5: null_sensitivity.tex (tab:nullsens)."""
    sub = df_crit[df_crit["sample"] == "main"].copy()
    floor_lookup = compute_full_sample_floor_jsd()

    model_order = ["Qwen2-VL-2B-Instruct", "SmolVLM-Instruct", "llava-1.5-7b-hf"]
    corr_order = ["swapblend", "gaussian", "patch"]

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Sensitivity of terminal reliance $\hat{R}(1)$, terminal utility $U(1)$, and unimodal divergence to the choice of null baseline (Removed vs.\ Black image) on the main sample ($N=500$, bootstrap $B=1000$, seed 42). The unimodal floor divergence $\text{JSD}(B_{\text{black}} \parallel B_{\text{removed}})$ is computed over the full sample of all $N=500$ questions (not restricted to the agreement subset) and reported once per model.}",
        r"\label{tab:nullsens}",
        r"\fittable{%",
        r"\begin{tabular}{llccccc}",
        r"\toprule",
        r" & & \multicolumn{2}{c}{\textbf{Removed Null Baseline}} & \multicolumn{2}{c}{\textbf{Black Null Baseline}} & \textbf{Full-Sample Unimodal Floor} \\",
        r"\cmidrule(lr){3-4} \cmidrule(lr){5-6} \cmidrule(lr){7-7}",
        r"\textbf{Model} & \textbf{Corruption} & $U(1)$ & $\hat{R}(1)$ & $U(1)$ & $\hat{R}(1)$ & $\text{JSD}(\text{blk} \parallel \text{rem})$ \\",
        r"\midrule",
    ]

    for m in model_order:
        m_display = format_model_name(m)
        m_floor = floor_lookup.get(m_display, "N/A")
        first_row = True
        for c in corr_order:
            prefix = f"\\textbf{{{m_display}}}" if first_row else ""
            floor_str = m_floor if first_row else ""
            first_row = False
            corr_display = format_corruption(c)

            r_rem = sub[(sub["model"] == m) & (sub["corruption"] == c) & (sub["null_baseline"] == "removed")].iloc[0]
            r_blk = sub[(sub["model"] == m) & (sub["corruption"] == c) & (sub["null_baseline"] == "black")].iloc[0]

            u1_rem_lo, u1_rem_hi = parse_ci_bounds(str(r_rem["U_1_ci_str"]))
            u1_rem = fmt_interval(r_rem["U_1"], u1_rem_lo, u1_rem_hi, signed=True)

            rhat_rem_lo, rhat_rem_hi = parse_ci_bounds(str(r_rem["R_hat_1_ci_str"]))
            rhat_rem = fmt_interval(r_rem["R_hat_1"], rhat_rem_lo, rhat_rem_hi)

            u1_blk_lo, u1_blk_hi = parse_ci_bounds(str(r_blk["U_1_ci_str"]))
            u1_blk = fmt_interval(r_blk["U_1"], u1_blk_lo, u1_blk_hi, signed=True)

            rhat_blk_lo, rhat_blk_hi = parse_ci_bounds(str(r_blk["R_hat_1_ci_str"]))
            rhat_blk = fmt_interval(r_blk["R_hat_1"], rhat_blk_lo, rhat_blk_hi)

            lines.append(f"{prefix} & {corr_display} & {u1_rem} & {rhat_rem} & {u1_blk} & {rhat_blk} & {floor_str} \\\\")
        lines.append(r"\midrule" if m != model_order[-1] else r"\bottomrule")

    lines.extend([
        r"\end{tabular}}",
        r"\end{table}",
    ])
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[OK] Wrote {out_path}")


def generate_baseline_metrics(df_base: pd.DataFrame, out_path: Path):
    """Generate Table 6: baseline_metrics.tex (tab:baseline)."""
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Static visual reliance and sensitivity baseline metrics on the swapped image across models on main ($N=500$) and replication ($N=445$) samples (bootstrap $B=1000$, seed 42). Metrics: Visual Reliance Score ($\text{VRS} = \text{acc}_{\text{clean}} - \text{acc}_{\text{swapped}}$), Black-image drop ($\text{acc}_{\text{clean}} - \text{acc}_{\text{black}}$), Image Sensitivity ($P[\arg\max(\text{clean}) \ne \arg\max(\text{swapped})]$), and Visual Harm Rate ($P[\text{clean wrong} \land \text{swapped correct}]$).}",
        r"\label{tab:baseline}",
        r"\fittable{%",
        r"\begin{tabular}{llcccc}",
        r"\toprule",
        r"\textbf{Model} & \textbf{Sample} & \textbf{VRS} & \textbf{Black-image drop} & \textbf{Sensitivity} & \textbf{Harm Rate} \\",
        r"\midrule",
    ]

    for split_val, split_label in [("main", "Main ($N=500$)"), ("replication", "Replication ($N=445$)")]:
        sub = df_base[df_base["split"] == split_val].copy()
        for _, r in sub.iterrows():
            m_display = format_model_name(r["model_name"])
            vrs = fmt_interval(r["VRS"], r["VRS_ci_lo"], r["VRS_ci_hi"])
            bdrop = fmt_interval(r["Blank_Drop"], r["Blank_Drop_ci_lo"], r["Blank_Drop_ci_hi"])
            sens = fmt_interval(r["Image_Sensitivity"], r["Image_Sensitivity_ci_lo"], r["Image_Sensitivity_ci_hi"])
            harm = fmt_interval(r["Visual_Harm_Rate"], r["Visual_Harm_Rate_ci_lo"], r["Visual_Harm_Rate_ci_hi"])

            lines.append(f"{m_display} & {split_label} & {vrs} & {bdrop} & {sens} & {harm} \\\\")
        if split_val == "main":
            lines.append(r"\midrule")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}}",
        r"\end{table}",
    ])
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[OK] Wrote {out_path}")


def generate_synthetic_summary(synth_csv: Path, out_path: Path):
    """Generate Table: synthetic_summary.tex (tab:synthetic_summary)."""
    df = pd.read_csv(synth_csv)
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Synthetic benchmark results across architectures and null baselines (Iteration v0.4, $N=2000$ test items per seed, 5 seeds, bootstrap $B=1000$, seed 42).}",
        r"\label{tab:synthetic-summary}",
        r"\fittable{%",
        r"\begin{tabular}{llccccc}",
        r"\toprule",
        r"\textbf{Model} & \textbf{Null} & \textbf{Clean Acc} & \textbf{Null Acc} & $S^*$ & $\text{MI}_{\text{pooled}}$ & $\text{MI}_{\text{hier}}$ \\",
        r"\midrule",
    ]

    for _, r in df.iterrows():
        m_name = str(r["model"]).replace("_", r"\_")
        null_k = str(r["null_kind"]).replace("_", r"\_")
        clean_acc = fmt_num(r["clean_acc"])
        null_acc = fmt_num(r["null_acc"])

        # S*
        if pd.isna(r["S_star"]) or (isinstance(r["S_star_pooled_CI"], str) and "no crossing" in r["S_star_pooled_CI"]):
            s_star = "no crossing"
        else:
            s_star = fmt_interval(r["S_star"], r["S_star_lo"], r["S_star_hi"])

        # MI pooled
        if pd.isna(r["MI"]) or (isinstance(r["MI_pooled_CI"], str) and "no crossing" in r["MI_pooled_CI"]):
            mi_pool = "no crossing"
        else:
            mi_pool = fmt_interval(r["MI"], r["MI_lo"], r["MI_hi"])

        # MI hier
        if pd.isna(r["MI"]) or (isinstance(r["MI_hier_CI"], str) and "no crossing" in r["MI_hier_CI"]):
            mi_hier = "no crossing"
        else:
            mi_hier = fmt_interval(r["MI"], r["MI_hier_lo"], r["MI_hier_hi"])

        lines.append(f"{m_name} & {null_k} & {clean_acc} & {null_acc} & {s_star} & {mi_pool} & {mi_hier} \\\\")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}}",
        r"\end{table}",
    ])
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[OK] Wrote {out_path}")


def main():
    crit_csv = REPO_ROOT / "results" / "derived" / "criterion_status_by_null_and_sample.csv"
    agree_csv = REPO_ROOT / "results" / "derived" / "item_level_agreement_analysis.csv"
    base_csv = REPO_ROOT / "results" / "derived" / "baseline_metrics_summary.csv"
    synth_csv = REPO_ROOT / "results" / "derived" / "synthetic_summary_v04.csv"

    if not crit_csv.exists():
        raise FileNotFoundError(f"Missing {crit_csv}")
    if not agree_csv.exists():
        raise FileNotFoundError(f"Missing {agree_csv}")
    if not base_csv.exists():
        raise FileNotFoundError(f"Missing {base_csv}")
    if not synth_csv.exists():
        raise FileNotFoundError(f"Missing {synth_csv}")

    df_crit = pd.read_csv(crit_csv)
    df_base = pd.read_csv(base_csv)

    tables_dir = REPO_ROOT / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    derived_dir = REPO_ROOT / "results" / "derived"

    generate_headline_main(df_crit, tables_dir / "headline_main.tex")
    generate_paired_swap_vs_gauss(tables_dir / "paired_swap_vs_gauss.tex", derived_dir / "paired_swap_vs_gauss.csv")
    generate_headline_replication(df_crit, tables_dir / "headline_replication.tex")
    generate_criterion_status_compact(df_crit, tables_dir / "criterion_status_compact.tex")
    generate_null_sensitivity(df_crit, tables_dir / "null_sensitivity.tex")
    generate_baseline_metrics(df_base, tables_dir / "baseline_metrics.tex")
    generate_synthetic_summary(synth_csv, tables_dir / "synthetic_summary.tex")
    print("[SUCCESS] All 7 paper tables generated.")


if __name__ == "__main__":
    main()
