"""Comprehensive script to compute and print all Task 2 metrics:
1. Summary table for main models (Qwen, LLaVA, SmolVLM) and replication (Qwen, LLaVA).
2. SmolVLM per-severity U and R_hat tables.
3. Paired swapblend-vs-gaussian R_hat(1) differences.
4. Pairwise cross-model R_hat(1) and U(1) differences.
5. Letter mass check (<0.90 check).
"""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.metrics import utility, reliance, normalized_reliance, smooth, bootstrap_ci

def load_shards(raw_dir):
    corruptions = ['swapblend', 'gaussian', 'patch']
    data = {}
    for corr in corruptions:
        bundles = {}
        for jf in sorted(Path(raw_dir).glob(f"*{corr}*.json")):
            if "manifest" in jf.name: continue
            d = json.loads(jf.read_text(encoding="utf-8"))
            for it in d["items"]:
                bundles[str(it["question_id"])] = it
        if bundles:
            data[corr] = bundles
    return data

def calc_u1_and_ci(items, seed=42, n_boot=2000):
    correct_idx = np.array([it["correct_choice_idx"] for it in items])
    # S=1 accuracy per item
    s1_acc = np.array([int(np.argmax(it["probs_per_S"][-1]) == it["correct_choice_idx"]) for it in items], dtype=float)
    # null removed accuracy per item
    null_acc = np.array([int(np.argmax(it["null_probs"]["removed"]) == it["correct_choice_idx"]) for it in items], dtype=float)
    paired = s1_acc - null_acc
    u1_point = float(paired.mean())

    def _mean(data): return float(data.mean())
    res = bootstrap_ci(_mean, paired, n_boot=n_boot, seed=seed)
    return u1_point, (float(res.lo), float(res.hi))

def print_summary_table():
    print("=" * 140)
    print("TASK 2: MAIN AND REPLICATION SUMMARY TABLES")
    print("=" * 140)

    for group_name, csv_path, raw_dirs in [
        ("MAIN GROUP (N=500)", "results/derived/vlm_main_summary.csv", {
            "Qwen/Qwen2-VL-2B-Instruct": "results/raw/vlm_main",
            "llava-hf/llava-1.5-7b-hf": "results/raw/vlm_llava",
            "HuggingFaceTB/SmolVLM-Instruct": "results/raw/vlm_smol",
        }),
        ("REPLICATION GROUP (N=445, Held-Out)", "results/derived/vlm_rep/vlm_main_summary.csv", {
            "Qwen/Qwen2-VL-2B-Instruct": "results/raw/vlm_rep",
            "llava-hf/llava-1.5-7b-hf": "results/raw/vlm_rep",
        })
    ]:
        print(f"\n>>> {group_name} <<<")
        df = pd.read_csv(csv_path)
        hdr = (
            f"{'Model':<18} | {'Corr':<9} | {'Clean':>5} | {'Rem':>5} | {'Blk':>5} | {'Swap':>5} | "
            f"{'U(1)':>7} {'U(1) 95% CI':>19} {'Sig':>3} | "
            f"{'R_hat(1)':>8} {'R_hat 95% CI':>18} | "
            f"{'S*':>6} | {'Sig S*':>6} | {'MI':>9} | {'NoCross%':>8} | {'Mass':>6}"
        )
        print("-" * len(hdr))
        print(hdr)
        print("-" * len(hdr))

        for _, row in df.iterrows():
            m_full = row["model_name"]
            m_short = "Qwen2-VL" if "Qwen" in m_full else ("LLaVA-1.5" if "llava" in m_full else "SmolVLM")
            corr = row["corruption"]

            # Load raw items to calculate U(1) and CI
            r_dir = raw_dirs.get(m_full)
            u1_val, (u_lo, u_hi) = np.nan, (np.nan, np.nan)
            if r_dir:
                shards = [f for f in Path(r_dir).glob(f"*{corr}*.json") if "manifest" not in f.name]
                if shards:
                    items = []
                    seen_qids = set()
                    for s in sorted(shards):
                        d = json.loads(s.read_text(encoding="utf-8"))
                        if d.get("model_name") == m_full or m_short.lower() in s.name.lower():
                            for it in d["items"]:
                                qid = str(it["question_id"])
                                if qid not in seen_qids:
                                    seen_qids.add(qid)
                                    items.append(it)
                    if items:
                        u1_val, (u_lo, u_hi) = calc_u1_and_ci(items)

            u_sig = "**" if (u_lo < 0 and u_hi < 0) else "ns"
            u_ci_str = f"[{u_lo:+.4f}, {u_hi:+.4f}]" if not np.isnan(u_lo) else "N/A"

            clean = f"{row['clean_acc']:.3f}"
            rem = f"{row['null_removed_acc']:.3f}"
            blk = f"{row['null_black_acc']:.3f}"
            swap = f"{row['swapped_acc']:.3f}"
            rhat = f"{row['R_hat_s1']:.4f}"
            rhat_ci = str(row['R_hat_s1_ci'])
            s_star = str(row['S_star'])
            sig_s = str(row['sig_S_star'])
            mi = str(row['MI'])
            nocross = f"{row['frac_no_usable_crossing']:.3f}"
            mass = f"{row['mean_letter_mass']:.4f}"

            print(
                f"{m_short:<18} | {corr:<9} | {clean:>5} | {rem:>5} | {blk:>5} | {swap:>5} | "
                f"{u1_val:>+7.4f} {u_ci_str:>19} {u_sig:>3} | "
                f"{rhat:>8} {rhat_ci:>18} | "
                f"{s_star:>6} | {sig_s:>6} | {mi:>9} | {nocross:>8} | {mass:>6}"
            )
        print("-" * len(hdr))

def smolvlm_per_severity():
    print("\n" + "=" * 105)
    print("SMOLVLM-INSTRUCT PER-SEVERITY U(S) AND R_hat(S) TABLES (N=500)")
    print("=" * 105)

    data = load_shards("results/raw/vlm_smol")
    for corr in ["swapblend", "gaussian", "patch"]:
        bundles = data[corr]
        items = list(bundles.values())
        n_items = len(items)
        S_grid = np.linspace(0.0, 1.0, 9)
        n_S = len(S_grid)

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

        # Bootstrap CIs
        rng = np.random.default_rng(42)
        boot_U = np.zeros((1000, n_S))
        boot_Rhat = np.zeros((1000, n_S))
        for b in range(1000):
            idx = rng.integers(0, n_items, size=n_items)
            b_acc = np.mean(acc_S_matrix[:, idx], axis=1)
            b_null = float(np.mean(acc_null_per_item[idx]))
            boot_U[b] = utility(b_acc, b_null)
            b_D = np.mean(jsd_matrix[:, idx], axis=1)
            b_R = reliance(b_D)
            try:
                boot_Rhat[b] = normalized_reliance(b_R)
            except Exception:
                boot_Rhat[b] = np.nan

        u_lo = np.percentile(boot_U, 2.5, axis=0)
        u_hi = np.percentile(boot_U, 97.5, axis=0)
        rhat_lo = np.nanpercentile(boot_Rhat, 2.5, axis=0)
        rhat_hi = np.nanpercentile(boot_Rhat, 97.5, axis=0)

        print(f"\n--- SmolVLM | Corruption: {corr.upper()} ---")
        print(f"{'S':>5} | {'Acc':>6} | {'U(raw)':>8} | {'U(smooth)':>9} | {'U 95% CI':>21} | {'R(S)':>7} | {'R_hat(S)':>8} | {'R_hat 95% CI':>18}")
        print("-" * 105)
        for s_idx, s in enumerate(S_grid):
            print(
                f"{s:5.3f} | {acc_deg[s_idx]:6.4f} | {U_S[s_idx]:+8.4f} | {U_sm[s_idx]:+9.4f} | "
                f"[{u_lo[s_idx]:+8.4f}, {u_hi[s_idx]:+8.4f}] | {R_S[s_idx]:7.4f} | {R_hat_S[s_idx]:8.4f} | "
                f"[{rhat_lo[s_idx]:.4f}, {rhat_hi[s_idx]:.4f}]"
            )

def paired_differences():
    print("\n" + "=" * 105)
    print("PAIRED DIFFERENCES IN R_hat(1) (BOOTSTRAP OVER QUESTIONS, N_BOOT=2000, SEED=42)")
    print("=" * 105)

    # Load 3 main models
    models = {
        "Qwen": load_shards("results/raw/vlm_main"),
        "LLaVA": load_shards("results/raw/vlm_llava"),
        "SmolVLM": load_shards("results/raw/vlm_smol"),
    }

    common_qids = sorted(list(
        set(models["Qwen"]["swapblend"].keys()) &
        set(models["LLaVA"]["swapblend"].keys()) &
        set(models["SmolVLM"]["swapblend"].keys())
    ))
    print(f"Aligned {len(common_qids)} common question IDs across all 3 models.")

    def calc_jsd(p, q):
        m = 0.5 * (p + q)
        kl_p = np.sum(p * np.log2(np.maximum(p, 1e-12) / np.maximum(m, 1e-12)), axis=-1)
        kl_q = np.sum(q * np.log2(np.maximum(q, 1e-12) / np.maximum(m, 1e-12)), axis=-1)
        return np.clip(0.5 * (kl_p + kl_q), 0, 1)

    vecs = {}
    for m_name, m_data in models.items():
        vecs[m_name] = {}
        for corr in ["swapblend", "gaussian", "patch"]:
            items = [m_data[corr][qid] for qid in common_qids]
            null_probs = np.array([it["null_probs"]["removed"] for it in items])
            p_clean = np.array([it["probs_per_S"][0] for it in items])
            p_s1 = np.array([it["probs_per_S"][-1] for it in items])
            correct_idx = np.array([it["correct_choice_idx"] for it in items])

            s1_acc = (np.argmax(p_s1, axis=-1) == correct_idx).astype(float)
            null_acc = (np.argmax(null_probs, axis=-1) == correct_idx).astype(float)

            d0 = calc_jsd(p_clean, null_probs)
            ds1 = calc_jsd(p_s1, null_probs)
            vecs[m_name][corr] = {
                "d0": d0, "ds1": ds1,
                "s1_acc": s1_acc, "null_acc": null_acc
            }

    N_BOOT = 2000
    rng = np.random.default_rng(42)
    n_items = len(common_qids)

    boot_rhat1 = {m: {c: np.zeros(N_BOOT) for c in ["swapblend", "gaussian", "patch"]} for m in models}
    boot_u1 = {m: {c: np.zeros(N_BOOT) for c in ["swapblend", "gaussian", "patch"]} for m in models}

    for b in range(N_BOOT):
        idx = rng.integers(0, n_items, size=n_items)
        for m_name in models:
            for corr in ["swapblend", "gaussian", "patch"]:
                v = vecs[m_name][corr]
                r0 = float(np.mean(v["d0"][idx]))
                rs1 = float(np.mean(v["ds1"][idx]))
                boot_rhat1[m_name][corr][b] = rs1 / r0 if r0 > 1e-6 else np.nan
                boot_u1[m_name][corr][b] = float(np.mean(v["s1_acc"][idx]) - np.mean(v["null_acc"][idx]))

    print("\n1. WITHIN-MODEL PAIRED DIFFERENCES: SWAPBLEND minus GAUSSIAN R_hat(1)")
    print("-" * 75)
    for m in models:
        pt_diff = float(
            (np.mean(vecs[m]["swapblend"]["ds1"]) / np.mean(vecs[m]["swapblend"]["d0"])) -
            (np.mean(vecs[m]["gaussian"]["ds1"]) / np.mean(vecs[m]["gaussian"]["d0"]))
        )
        boot_diff = boot_rhat1[m]["swapblend"] - boot_rhat1[m]["gaussian"]
        ci = np.percentile(boot_diff, [2.5, 97.5])
        sig = "**" if (ci[0] > 0 or ci[1] < 0) else "ns"
        print(f"  {m:<10} : Delta R_hat(1) = {pt_diff:+.4f} [{ci[0]:+.4f}, {ci[1]:+.4f}] ({sig})")

    print("\n2. PAIRWISE CROSS-MODEL DIFFERENCES IN R_hat(1)")
    print("-" * 75)
    pairs = [("LLaVA", "Qwen"), ("SmolVLM", "Qwen"), ("SmolVLM", "LLaVA")]
    for corr in ["swapblend", "gaussian", "patch"]:
        print(f"  Corruption: {corr.upper()}")
        for m1, m2 in pairs:
            pt_diff = float(
                (np.mean(vecs[m1][corr]["ds1"]) / np.mean(vecs[m1][corr]["d0"])) -
                (np.mean(vecs[m2][corr]["ds1"]) / np.mean(vecs[m2][corr]["d0"]))
            )
            boot_diff = boot_rhat1[m1][corr] - boot_rhat1[m2][corr]
            ci = np.percentile(boot_diff, [2.5, 97.5])
            sig = "**" if (ci[0] > 0 or ci[1] < 0) else "ns"
            print(f"    {m1} - {m2:<7} : Delta R_hat(1) = {pt_diff:+.4f} [{ci[0]:+.4f}, {ci[1]:+.4f}] ({sig})")

    print("\n3. PAIRWISE CROSS-MODEL DIFFERENCES IN U(1)")
    print("-" * 75)
    for corr in ["swapblend", "gaussian", "patch"]:
        print(f"  Corruption: {corr.upper()}")
        for m1, m2 in pairs:
            u1_pt = float(np.mean(vecs[m1][corr]["s1_acc"]) - np.mean(vecs[m1][corr]["null_acc"]))
            u2_pt = float(np.mean(vecs[m2][corr]["s1_acc"]) - np.mean(vecs[m2][corr]["null_acc"]))
            pt_diff = u1_pt - u2_pt
            boot_diff = boot_u1[m1][corr] - boot_u1[m2][corr]
            ci = np.percentile(boot_diff, [2.5, 97.5])
            sig = "**" if (ci[0] > 0 or ci[1] < 0) else "ns"
            print(f"    {m1} - {m2:<7} : Delta U(1)     = {pt_diff:+.4f} [{ci[0]:+.4f}, {ci[1]:+.4f}] ({sig})")

def check_letter_mass():
    print("\n" + "=" * 75)
    print("LETTER MASS CHECK (THRESHOLD >= 0.90)")
    print("=" * 75)
    for p in ["results/raw/vlm_main", "results/raw/vlm_llava", "results/raw/vlm_smol", "results/raw/vlm_rep"]:
        for s in Path(p).glob("*.json"):
            if "manifest" in s.name: continue
            d = json.loads(s.read_text(encoding="utf-8"))
            m_name = d.get("model_name", "unknown")
            masses = []
            for it in d["items"]:
                if "mass_per_S" in it: masses.extend(it["mass_per_S"])
                if "null_mass" in it:
                    for v in it["null_mass"].values(): masses.append(v)
            mean_m = np.mean(masses) if masses else 1.0
            if mean_m < 0.90:
                print(f"  [ALERT] Model {m_name} in {s.name} has mean letter mass {mean_m:.4f} < 0.90!")
    print("All models across all conditions checked.")

if __name__ == "__main__":
    print_summary_table()
    smolvlm_per_severity()
    paired_differences()
    check_letter_mass()
