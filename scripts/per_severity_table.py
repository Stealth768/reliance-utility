"""Print per-severity tables (TASK 1) for each Qwen main corruption.

Outputs: acc, U_raw, U_smooth, U_95CI_lo, U_95CI_hi, R, R_hat, R_hat_95CI_lo, R_hat_95CI_hi
for all 9 severity grid points.
"""
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from src.metrics import reliance, normalized_reliance, utility, smooth

RAW_DIR = REPO_ROOT / "results" / "raw" / "vlm_main"
CORRUPTIONS = ["gaussian", "patch", "swapblend"]
N_BOOT = 1000
SEED = 42


def load_corruption(corr: str):
    bundles = {}
    data = None
    for jf in sorted(RAW_DIR.glob(f"*{corr}*.json")):
        if "manifest" in jf.name:
            continue
        data = json.loads(jf.read_text(encoding="utf-8"))
        if "items" not in data:
            continue
        for it in data["items"]:
            bundles[it["question_id"]] = it
    if data is None:
        raise FileNotFoundError(f"No shards found for {corr}")
    S_grid = np.array(data["S_grid"])
    items = list(bundles.values())
    return items, S_grid


def compute_curves(items, S_grid):
    n_S = len(S_grid)
    n_items = len(items)
    correct_idx = np.array([it["correct_choice_idx"] for it in items])
    null_probs = np.array([it["null_probs"]["removed"] for it in items])
    probs_S = np.array([[it["probs_per_S"][s] for it in items] for s in range(n_S)])
    preds_S = np.argmax(probs_S, axis=-1)
    acc_S_matrix = (preds_S == correct_idx[None, :]).astype(float)
    acc_deg = np.mean(acc_S_matrix, axis=1)
    acc_null_per_item = (np.argmax(null_probs, axis=-1) == correct_idx).astype(float)
    acc_null = float(np.mean(acc_null_per_item))

    jsd_matrix = np.zeros((n_S, n_items))
    for s in range(n_S):
        ps = probs_S[s]
        pn = null_probs
        m = 0.5 * (ps + pn)
        kl_p = np.sum(ps * np.log2(np.maximum(ps, 1e-12) / np.maximum(m, 1e-12)), axis=-1)
        kl_n = np.sum(pn * np.log2(np.maximum(pn, 1e-12) / np.maximum(m, 1e-12)), axis=-1)
        jsd_matrix[s] = np.clip(0.5 * (kl_p + kl_n), 0, 1)

    D_S = np.mean(jsd_matrix, axis=1)
    U_S = utility(acc_deg, acc_null)
    U_sm = smooth(U_S, window=3)
    R_S = reliance(D_S)
    R_hat_S = normalized_reliance(R_S)

    # Pooled bootstrap for U CI and R_hat CI
    rng = np.random.default_rng(SEED)
    boot_U = np.zeros((N_BOOT, n_S))
    boot_Rhat = np.full((N_BOOT, n_S), np.nan)
    for b in range(N_BOOT):
        idx = rng.integers(0, n_items, size=n_items)
        b_acc = np.mean(acc_S_matrix[:, idx], axis=1)
        b_null = float(np.mean(acc_null_per_item[idx]))
        b_U = utility(b_acc, b_null)
        boot_U[b] = b_U
        b_D = np.mean(jsd_matrix[:, idx], axis=1)
        b_R = reliance(b_D)
        try:
            b_Rhat = normalized_reliance(b_R)
            boot_Rhat[b] = b_Rhat
        except Exception:
            pass

    U_lo = np.percentile(boot_U, 2.5, axis=0)
    U_hi = np.percentile(boot_U, 97.5, axis=0)
    Rhat_lo = np.nanpercentile(boot_Rhat, 2.5, axis=0)
    Rhat_hi = np.nanpercentile(boot_Rhat, 97.5, axis=0)

    return {
        "acc_null": acc_null,
        "acc_deg": acc_deg,
        "U_S": U_S,
        "U_sm": U_sm,
        "U_lo": U_lo,
        "U_hi": U_hi,
        "R_S": R_S,
        "R_hat_S": R_hat_S,
        "Rhat_lo": Rhat_lo,
        "Rhat_hi": Rhat_hi,
        "S_grid": S_grid,
        "n_items": n_items,
    }


def print_table(corr: str, c: dict):
    sg = c["S_grid"]
    n = c["n_items"]
    print(f"\n=== {corr.upper()} | N={n} | acc_null_removed={c['acc_null']:.4f} ===")
    hdr = (
        f"{'S':>6}  {'acc':>6}  {'U_raw':>7}  {'U_sm':>7}  "
        f"{'U95lo':>7}  {'U95hi':>7}  "
        f"{'R':>7}  {'R_hat':>7}  {'Rh95lo':>7}  {'Rh95hi':>7}"
    )
    print(hdr)
    print("-" * len(hdr))
    for i, s in enumerate(sg):
        print(
            f"{s:6.3f}  {c['acc_deg'][i]:6.4f}  {c['U_S'][i]:7.4f}  {c['U_sm'][i]:7.4f}  "
            f"{c['U_lo'][i]:7.4f}  {c['U_hi'][i]:7.4f}  "
            f"{c['R_S'][i]:7.4f}  {c['R_hat_S'][i]:7.4f}  {c['Rhat_lo'][i]:7.4f}  {c['Rhat_hi'][i]:7.4f}"
        )
    print()


def main():
    for corr in CORRUPTIONS:
        items, S_grid = load_corruption(corr)
        c = compute_curves(items, S_grid)
        print_table(corr, c)


if __name__ == "__main__":
    main()
