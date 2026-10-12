"""Compute replication (vlm_rep) metrics table with U(S=1) bootstrap CIs.

Prints the swapblend and gaussian rows needed for paper reporting:
  clean / removed / swapped accuracy, R_hat(1) with CI, U(1) with CI.
"""
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.analyze_vlm_all import load_shards_grouped, compute_metrics_for_items
from src.metrics.metrics import bootstrap_ci


def u_ci_at_s1(items, S_grid, null_kind="removed"):
    """Bootstrap 95% CI for U(S=1) = acc(S=1) - null_acc via paired per-item differences."""
    mets = compute_metrics_for_items(items, S_grid, null_kind=null_kind)
    if not mets:
        return float("nan"), float("nan"), float("nan")

    u_point = float(mets["U_S"][-1])  # already stored

    # Per-item paired difference: correct_at_s1 - correct_at_null
    acc_s1_per_item = mets["acc_S_matrix"][-1]          # shape (n,), last row = S=1
    acc_null_per_item = mets["acc_null_per_item"]        # shape (n,)
    paired = acc_s1_per_item - acc_null_per_item         # shape (n,)

    def _u_mean(data):
        return float(data.mean())

    ci_result = bootstrap_ci(_u_mean, paired, n_boot=2000, seed=42)
    if ci_result is None or ci_result.lo is None:
        return u_point, float("nan"), float("nan")
    return u_point, float(ci_result.lo), float(ci_result.hi)


def rhat_ci_at_s1(items, S_grid, null_kind="removed"):
    """R_hat(S=1) with 95% CI directly from the summary CSV (already computed)."""
    # We return the pre-computed value from mets (analysis script stores it)
    mets = compute_metrics_for_items(items, S_grid, null_kind=null_kind)
    if not mets:
        return float("nan"), float("nan"), float("nan")
    # R_hat_S is shape (n_S,) — last entry = R_hat(1)
    r1 = float(mets["R_hat_S"][-1])
    return r1


def swapped_acc(items, S_grid):
    """Accuracy on the swapped control."""
    correct_idx = np.array([it["correct_choice_idx"] for it in items])
    swapped_probs = np.array([it["swapped_probs"] for it in items])
    preds = swapped_probs.argmax(axis=1)
    return float((preds == correct_idx).mean())


def main():
    raw_dir = Path("results/raw/vlm_rep")
    groups = load_shards_grouped(raw_dir, no_siblings=True)
    if not groups:
        print("ERROR: No shards found in", raw_dir)
        sys.exit(1)

    # Pull R_hat CI from the summary CSV (pre-computed by analyze_vlm_all)
    import pandas as pd
    summary_csv = Path("results/derived/vlm_rep/vlm_main_summary.csv")
    summary_df = pd.read_csv(summary_csv) if summary_csv.exists() else None

    header = (
        f"{'Corruption':<12} {'N':>5} "
        f"{'Acc_clean':>10} {'Acc_removed':>12} {'Acc_swapped':>12} "
        f"{'R_hat(1)':>10} {'R_hat_CI':>22} "
        f"{'U(1)':>8} {'U(1)_CI':>22} {'Sig':>4}"
    )
    print("\n" + "=" * len(header))
    print("Replication: Qwen/Qwen2-VL-2B-Instruct | N=445 held-out | seed=1")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for corruption in ["swapblend", "gaussian"]:
        key = next((k for k in groups if k[1] == corruption), None)
        if key is None:
            print(f"{corruption:<12}  NOT FOUND")
            continue

        g = groups[key]
        items = g["items"]
        S_grid = g["S_grid"]
        n = len(items)
        mets = compute_metrics_for_items(items, S_grid, null_kind="removed")

        clean_acc = float(mets["clean_acc"])
        null_acc = float(mets["acc_null"])
        sw_acc = swapped_acc(items, S_grid)
        r1 = rhat_ci_at_s1(items, S_grid)

        # Pull pre-computed R_hat CI from CSV
        r_ci_str = "N/A"
        if summary_df is not None:
            row = summary_df[(summary_df["corruption"] == corruption)]
            if len(row):
                r_ci_str = str(row.iloc[0].get("R_hat_s1_ci", row.iloc[0].get("R_hat_s1_ci_lo", "N/A")))
                # try alternate column name
                if "R_hat_s1_ci" in summary_df.columns:
                    r_ci_str = str(row.iloc[0]["R_hat_s1_ci"])

        u1, u_lo, u_hi = u_ci_at_s1(items, S_grid)
        u_sig = "**" if (not np.isnan(u_lo) and u_lo < 0 and u_hi < 0) else "ns"

        print(
            f"{corruption:<12} {n:>5} "
            f"{clean_acc:>10.4f} {null_acc:>12.4f} {sw_acc:>12.4f} "
            f"{r1:>10.4f} {r_ci_str:>22} "
            f"{u1:>+8.4f} [{u_lo:+.4f},{u_hi:+.4f}] {u_sig:>4}"
        )

    print("=" * len(header))
    print("\nNote:")
    print("  Acc_removed  = null-removed accuracy (model's baseline when query image is removed)")
    print("  Acc_swapped  = accuracy on swapped control (donor image replacing query)")
    print("  R_hat(1)     = normalised reliance at S=1: D(S=1) / D(S=0)")
    print("  R_hat_CI     = 95% bootstrap CI (from analyze_vlm_all)")
    print("  U(1)         = acc(S=1) - null_removed_acc")
    print("  **           = both CI bounds < 0 (significant below 0); ns = not significant")
    print("  U(1) CI      = 2000-resample paired bootstrap 95% CI")


if __name__ == "__main__":
    main()
