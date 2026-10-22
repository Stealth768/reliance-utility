"""Compute raw floor JSD as a fraction of swapblend R(1) per model on main sample.

Note: Because Jensen-Shannon Divergence is not additive, the ratio of unimodal
floor JSD to swapblend R(1) is an informal descriptive heuristic rather than an
additive decomposition of reliance.

Saves results with 95% bootstrap CI (B=1000, seed=42), labeled post hoc,
to results/derived/floor_fraction.csv.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent


def calc_jsd(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Compute Jensen-Shannon Divergence between two sets of probability distributions."""
    m = 0.5 * (p + q)
    kl_p = np.sum(p * np.log2(np.maximum(p, 1e-12) / np.maximum(m, 1e-12)), axis=-1)
    kl_q = np.sum(q * np.log2(np.maximum(q, 1e-12) / np.maximum(m, 1e-12)), axis=-1)
    return np.clip(0.5 * (kl_p + kl_q), 0.0, 1.0)


def compute_floor_fraction(
    out_csv: Path,
    n_boot: int = 1000,
    seed: int = 42,
) -> pd.DataFrame:
    models_dir = {
        "Qwen2-VL-2B-Instruct": REPO_ROOT / "results" / "raw" / "vlm_main",
        "SmolVLM-Instruct": REPO_ROOT / "results" / "raw" / "vlm_smol",
        "llava-1.5-7b-hf": REPO_ROOT / "results" / "raw" / "vlm_llava",
    }

    rows = []
    for m_name, raw_dir in models_dir.items():
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
        p_s1 = np.array([it["probs_per_S"][-1] for it in items])

        floor_jsd = calc_jsd(p_blk, p_rem)
        swap_r1 = calc_jsd(p_s1, p_rem)

        m_f = float(np.mean(floor_jsd))
        m_s = float(np.mean(swap_r1))
        pt_frac = m_f / m_s

        rng = np.random.default_rng(seed)
        boot_f = []
        boot_s = []
        boot_frac = []
        for _ in range(n_boot):
            idx = rng.integers(0, N, size=N)
            bf = float(np.mean(floor_jsd[idx]))
            bs = float(np.mean(swap_r1[idx]))
            boot_f.append(bf)
            boot_s.append(bs)
            boot_frac.append(bf / bs)

        rows.append({
            "model": m_name,
            "sample": "main",
            "n_items": N,
            "raw_floor_jsd": m_f,
            "raw_floor_jsd_ci_lo": float(np.percentile(boot_f, 2.5)),
            "raw_floor_jsd_ci_hi": float(np.percentile(boot_f, 97.5)),
            "swapblend_R_1": m_s,
            "swapblend_R_1_ci_lo": float(np.percentile(boot_s, 2.5)),
            "swapblend_R_1_ci_hi": float(np.percentile(boot_s, 97.5)),
            "floor_fraction": pt_frac,
            "floor_fraction_ci_lo": float(np.percentile(boot_frac, 2.5)),
            "floor_fraction_ci_hi": float(np.percentile(boot_frac, 97.5)),
            "ratio_swap_over_floor": m_s / m_f,
            "analysis_status": "post hoc",
            "floor_pct": round(pt_frac * 100.0, 1),
        })

    df = pd.DataFrame(rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"[OK] Saved {out_csv}")
    return df


def main():
    out_csv = REPO_ROOT / "results" / "derived" / "floor_fraction.csv"
    df = compute_floor_fraction(out_csv=out_csv)
    for _, r in df.iterrows():
        print(
            f"  {r['model']}: floor_fraction = {r['floor_fraction']:.4f} "
            f"[{r['floor_fraction_ci_lo']:.4f}, {r['floor_fraction_ci_hi']:.4f}] "
            f"(floor={r['raw_floor_jsd']:.4f}, swap_r1={r['swapblend_R_1']:.4f}, "
            f"ratio={r['ratio_swap_over_floor']:.2f}x) [{r['analysis_status']}]"
        )


if __name__ == "__main__":
    main()
