import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
from src.metrics import utility, reliance, normalized_reliance, smooth

def analyze_model_per_severity(model_name, raw_dir):
    corruptions = ['swapblend', 'gaussian', 'patch']
    results = {}
    for corr in corruptions:
        bundles = {}
        data = None
        for jf in sorted(Path(raw_dir).glob(f'*{corr}*.json')):
            if 'manifest' in jf.name: continue
            data = json.loads(jf.read_text(encoding='utf-8'))
            for it in data['items']:
                bundles[it['question_id']] = it
        items = list(bundles.values())
        S_grid = np.array(data['S_grid'])
        n_S = len(S_grid)
        n_items = len(items)

        correct_idx = np.array([it['correct_choice_idx'] for it in items])
        null_probs = np.array([it['null_probs']['removed'] for it in items])
        probs_S = np.array([[it['probs_per_S'][s] for it in items] for s in range(n_S)])
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

        # Bootstrap
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

        results[corr] = {
            'S_grid': S_grid,
            'acc_deg': acc_deg,
            'U_S': U_S,
            'U_sm': U_sm,
            'u_lo': u_lo,
            'u_hi': u_hi,
            'R_S': R_S,
            'R_hat_S': R_hat_S,
            'rhat_lo': rhat_lo,
            'rhat_hi': rhat_hi,
            'items_dict': bundles,
            'acc_S_matrix': acc_S_matrix,
            'acc_null_per_item': acc_null_per_item,
            'jsd_matrix': jsd_matrix
        }
    return results

print('Computing for LLaVA and Qwen...')
llava_res = analyze_model_per_severity('LLaVA', 'results/raw/vlm_llava')
qwen_res = analyze_model_per_severity('Qwen', 'results/raw/vlm_main')

for corr in ['swapblend', 'gaussian', 'patch']:
    r = llava_res[corr]
    print(f'=== LLAVA {corr.upper()} PER-SEVERITY TABLE (N=500) ===')
    print('   S     Acc     U(raw)   U(smooth)     U 95% CI          R(S)    R_hat(S)    R_hat 95% CI')
    print('------------------------------------------------------------------------------------------')
    for i, s in enumerate(r['S_grid']):
        ulo = r['u_lo'][i]
        uhi = r['u_hi'][i]
        rlo = r['rhat_lo'][i]
        rhi = r['rhat_hi'][i]
        print(f"{s:6.3f}  {r['acc_deg'][i]:6.4f}  {r['U_S'][i]:7.4f}  {r['U_sm'][i]:7.4f}   [{ulo:7.4f}, {uhi:7.4f}]   {r['R_S'][i]:6.4f}   {r['R_hat_S'][i]:6.4f}   [{rlo:6.4f}, {rhi:6.4f}]")
    print()

print('=== U(1) SIGNIFICANCE SUMMARY ===')
for mname, mres in [('Qwen2-VL-2B', qwen_res), ('LLaVA-1.5-7B', llava_res)]:
    for corr in ['swapblend', 'gaussian', 'patch']:
        r = mres[corr]
        u1 = r['U_S'][-1]
        lo = r['u_lo'][-1]
        hi = r['u_hi'][-1]
        sig = hi < 0
        print(f"{mname:15s} | {corr:10s}: U(1) = {u1:7.4f} [{lo:7.4f}, {hi:7.4f}] -> Significantly < 0? {sig}")
print()
