"""Task 2: Compute cross-model table and pairwise paired bootstrap differences."""
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
from src.metrics import utility, reliance, normalized_reliance

def load_data(raw_dir, model_slug):
    corruptions = ['swapblend', 'gaussian', 'patch']
    data = {}
    for corr in corruptions:
        bundles = {}
        for jf in sorted(Path(raw_dir).glob(f"*{corr}*.json")):
            if "manifest" in jf.name: continue
            d = json.loads(jf.read_text(encoding="utf-8"))
            for it in d["items"]:
                bundles[it["question_id"]] = it
        data[corr] = bundles
    return data

qwen_data = load_data("results/raw/vlm_main", "Qwen")
llava_data = load_data("results/raw/vlm_llava", "llava")

# Align questions across both models (N=500 common questions)
common_qids = sorted(list(set(qwen_data["swapblend"].keys()) & set(llava_data["swapblend"].keys())))
print(f"Aligned {len(common_qids)} common question IDs across models.")

def extract_vectors(bundles, qids):
    items = [bundles[qid] for qid in qids]
    n = len(items)
    correct_idx = np.array([it["correct_choice_idx"] for it in items])
    null_probs = np.array([it["null_probs"]["removed"] for it in items])
    # S=0 (clean) and S=1 (terminal)
    p_clean = np.array([it["probs_per_S"][0] for it in items])
    p_s1 = np.array([it["probs_per_S"][-1] for it in items])

    # Utility per item
    clean_acc_item = (np.argmax(p_clean, axis=-1) == correct_idx).astype(float)
    s1_acc_item = (np.argmax(p_s1, axis=-1) == correct_idx).astype(float)
    null_acc_item = (np.argmax(null_probs, axis=-1) == correct_idx).astype(float)

    # JSD per item at S=0 and S=1
    def calc_jsd(p, q):
        m = 0.5 * (p + q)
        kl_p = np.sum(p * np.log2(np.maximum(p, 1e-12) / np.maximum(m, 1e-12)), axis=-1)
        kl_q = np.sum(q * np.log2(np.maximum(q, 1e-12) / np.maximum(m, 1e-12)), axis=-1)
        return np.clip(0.5 * (kl_p + kl_q), 0, 1)

    d0_item = calc_jsd(p_clean, null_probs)
    ds1_item = calc_jsd(p_s1, null_probs)

    return {
        "s1_acc_item": s1_acc_item,
        "null_acc_item": null_acc_item,
        "d0_item": d0_item,
        "ds1_item": ds1_item,
    }

qwen_vecs = {c: extract_vectors(qwen_data[c], common_qids) for c in ['swapblend', 'gaussian', 'patch']}
llava_vecs = {c: extract_vectors(llava_data[c], common_qids) for c in ['swapblend', 'gaussian', 'patch']}

# Bootstrap paired resamples over questions
N_BOOT = 2000
rng = np.random.default_rng(42)
n_items = len(common_qids)

boot_results = {
    'Qwen': {c: {'U1': np.zeros(N_BOOT), 'Rhat1': np.zeros(N_BOOT)} for c in ['swapblend', 'gaussian', 'patch']},
    'LLaVA': {c: {'U1': np.zeros(N_BOOT), 'Rhat1': np.zeros(N_BOOT)} for c in ['swapblend', 'gaussian', 'patch']},
}

for b in range(N_BOOT):
    idx = rng.integers(0, n_items, size=n_items)
    for model_name, vecs in [('Qwen', qwen_vecs), ('LLaVA', llava_vecs)]:
        for corr in ['swapblend', 'gaussian', 'patch']:
            v = vecs[corr]
            u1 = float(np.mean(v["s1_acc_item"][idx]) - np.mean(v["null_acc_item"][idx]))
            r0 = float(np.mean(v["d0_item"][idx]))
            rs1 = float(np.mean(v["ds1_item"][idx]))
            rhat1 = rs1 / r0 if r0 > 1e-6 else np.nan
            boot_results[model_name][corr]['U1'][b] = u1
            boot_results[model_name][corr]['Rhat1'][b] = rhat1

print("\n=== POINT ESTIMATES & 95% CIs ===")
for m in ['Qwen', 'LLaVA']:
    for c in ['swapblend', 'gaussian', 'patch']:
        v = qwen_vecs[c] if m == 'Qwen' else llava_vecs[c]
        u1_pt = float(np.mean(v["s1_acc_item"]) - np.mean(v["null_acc_item"]))
        rhat1_pt = float(np.mean(v["ds1_item"]) / np.mean(v["d0_item"]))
        u_ci = np.percentile(boot_results[m][c]['U1'], [2.5, 97.5])
        r_ci = np.percentile(boot_results[m][c]['Rhat1'], [2.5, 97.5])
        print(f"{m:6s} | {c:10s} : R_hat(1) = {rhat1_pt:.4f} [{r_ci[0]:.4f}, {r_ci[1]:.4f}] | U(1) = {u1_pt:+.4f} [{u_ci[0]:+.4f}, {u_ci[1]:+.4f}]")

print("\n=== PAIRED DIFFERENCES IN R_hat(1) (PAIRED OVER QUESTIONS) ===")
# 1. Within model: swapblend minus gaussian
for m in ['Qwen', 'LLaVA']:
    v_sw = qwen_vecs['swapblend'] if m == 'Qwen' else llava_vecs['swapblend']
    v_ga = qwen_vecs['gaussian'] if m == 'Qwen' else llava_vecs['gaussian']
    rhat_sw_pt = float(np.mean(v_sw["ds1_item"]) / np.mean(v_sw["d0_item"]))
    rhat_ga_pt = float(np.mean(v_ga["ds1_item"]) / np.mean(v_ga["d0_item"]))
    diff_pt = rhat_sw_pt - rhat_ga_pt
    diff_boot = boot_results[m]['swapblend']['Rhat1'] - boot_results[m]['gaussian']['Rhat1']
    diff_ci = np.percentile(diff_boot, [2.5, 97.5])
    print(f"{m} [swapblend - gaussian]: {diff_pt:+.4f} [{diff_ci[0]:+.4f}, {diff_ci[1]:+.4f}]")

# 2. Across models: LLaVA minus Qwen per corruption
for c in ['swapblend', 'gaussian', 'patch']:
    v_ll = llava_vecs[c]
    v_qw = qwen_vecs[c]
    rhat_ll_pt = float(np.mean(v_ll["ds1_item"]) / np.mean(v_ll["d0_item"]))
    rhat_qw_pt = float(np.mean(v_qw["ds1_item"]) / np.mean(v_qw["d0_item"]))
    diff_pt = rhat_ll_pt - rhat_qw_pt
    diff_boot = boot_results['LLaVA'][c]['Rhat1'] - boot_results['Qwen'][c]['Rhat1']
    diff_ci = np.percentile(diff_boot, [2.5, 97.5])
    print(f"[LLaVA - Qwen] on {c:10s}: {diff_pt:+.4f} [{diff_ci[0]:+.4f}, {diff_ci[1]:+.4f}]")
