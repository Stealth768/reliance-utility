"""TASK 4: Investigate whether swapped_probs matches probs_per_S at S=1."""
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = REPO_ROOT / "results" / "raw" / "vlm_main"

import numpy as np


def check_corruption(corr: str):
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
        print(f"No shards found for {corr}")
        return
    items = list(bundles.values())
    S_grid = data["S_grid"]

    n_match = 0
    n_total = 0
    max_diff = 0.0
    for it in items:
        if "swapped_probs" not in it:
            break
        swapped = np.array(it["swapped_probs"])
        s1_probs = np.array(it["probs_per_S"][-1])
        diff = float(np.max(np.abs(swapped - s1_probs)))
        max_diff = max(max_diff, diff)
        if np.allclose(swapped, s1_probs, atol=1e-6):
            n_match += 1
        n_total += 1

    print(f"\n=== {corr.upper()} ===")
    print(f"  S_grid last value: {S_grid[-1]}")
    print(f"  N items with swapped_probs: {n_total}")
    print(f"  swapped_probs == probs_per_S[S=1.0]: {n_match}/{n_total}")
    print(f"  Max abs difference: {max_diff:.8f}")
    if n_total > 0:
        print("  First 3 examples:")
        for it in items[:3]:
            if "swapped_probs" not in it:
                break
            swapped = np.array(it["swapped_probs"])
            s1_probs = np.array(it["probs_per_S"][-1])
            print(f"    qid={it['question_id']}: swapped={[round(v,4) for v in swapped.tolist()]}  S=1={[round(v,4) for v in s1_probs.tolist()]}")


if __name__ == "__main__":
    for corr in ["gaussian", "patch", "swapblend"]:
        check_corruption(corr)
