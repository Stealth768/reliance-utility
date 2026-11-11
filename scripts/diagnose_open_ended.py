#!/usr/bin/env python3
"""Diagnostic script for Open-Ended VLM benchmark."""

import glob
import json
import os
import sys
from pathlib import Path
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
open_dir = REPO_ROOT / "results" / "raw" / "vlm_open"

models = [
    ("Qwen2-VL-2B", "open_Qwen2-VL-2B-Instruct"),
    ("SmolVLM", "open_SmolVLM-Instruct"),
    ("LLaVA-1.5-7B", "open_llava-1.5-7b-hf"),
]

for model, m_pat in models:
    print("=" * 70)
    print(f"DIAGNOSTIC FOR MODEL: {model}")
    print("=" * 70)
    for corr in ["swapblend", "gaussian", "patch"]:
        files = sorted(open_dir.glob(f"{m_pat}_{corr}_seed0_shard*.json"))
        if not files:
            continue
        items = []
        for f in files:
            with open(f, "r", encoding="utf-8") as fp:
                items.extend(json.load(fp)["items"])

        n = len(items)
        clean_u = [it.get("clean_utility", 0.0) for it in items]
        null_u = [it.get("null_utility", {}).get("removed", 0.0) for it in items]
        s1_u = [it.get("utility_per_S", [])[-1] if it.get("utility_per_S") else 0.0 for it in items]
        diff_u = [s - nu for s, nu in zip(s1_u, null_u)]

        # Check generated answers
        clean_ans = [it.get("clean_answer", "") for it in items]
        null_ans = [it.get("null_answers", {}).get("removed", "") for it in items]
        s1_ans = [it.get("answers_per_S", [])[-1] if it.get("answers_per_S") else "" for it in items]

        empty_clean = sum(1 for a in clean_ans if not a.strip()) / n
        empty_null = sum(1 for a in null_ans if not a.strip()) / n
        empty_s1 = sum(1 for a in s1_ans if not a.strip()) / n

        identical_clean_null = sum(1 for c, nu in zip(clean_ans, null_ans) if c.strip() == nu.strip()) / n
        identical_clean_s1 = sum(1 for c, s in zip(clean_ans, s1_ans) if c.strip() == s.strip()) / n
        identical_null_s1 = sum(1 for nu, s in zip(null_ans, s1_ans) if nu.strip() == s.strip()) / n

        print(f"\nCorruption: {corr} (N={n})")
        print(f"  Clean utility (acc) mean: {np.mean(clean_u):.4f} (max={np.max(clean_u):.4f}, non-zero={np.count_nonzero(clean_u)}/{n})")
        print(f"  Null  utility (acc) mean: {np.mean(null_u):.4f} (max={np.max(null_u):.4f}, non-zero={np.count_nonzero(null_u)}/{n})")
        print(f"  S=1   utility (acc) mean: {np.mean(s1_u):.4f} (max={np.max(s1_u):.4f}, non-zero={np.count_nonzero(s1_u)}/{n})")
        print(f"  U(1) = S1 - Null    mean: {np.mean(diff_u):.4f} (non-zero={np.count_nonzero(diff_u)}/{n})")
        print(f"  Empty answers: Clean={empty_clean*100:.1f}%, Null={empty_null*100:.1f}%, S1={empty_s1*100:.1f}%")
        print(f"  Identical answers: Clean==Null: {identical_clean_null*100:.1f}%, Clean==S1: {identical_clean_s1*100:.1f}%, Null==S1: {identical_null_s1*100:.1f}%")

        # Also check sequence reliance values
        r0 = [it["R0"]["removed"] if isinstance(it.get("R0"), dict) else it.get("R_removed_per_S", [0])[0] for it in items]
        r1 = [it.get("R_removed_per_S", [0])[-1] for it in items]
        rhat = np.mean(r1) / np.mean(r0) if np.mean(r0) > 0 else 0.0
        print(f"  R(0) mean: {np.mean(r0):.4f}, R(1) mean: {np.mean(r1):.4f}, R_hat(1): {rhat:.4f}")

        # Sample answers
        print("  Sample answers (first 3 items):")
        for i in range(min(3, n)):
            c_a = clean_ans[i].replace('\n', ' ')[:30]
            n_a = null_ans[i].replace('\n', ' ')[:30]
            s_a = s1_ans[i].replace('\n', ' ')[:30]
            print(f"    Q{i}: Clean='{c_a}' (u={clean_u[i]:.2f}) | Null='{n_a}' (u={null_u[i]:.2f}) | S1='{s_a}' (u={s1_u[i]:.2f})")
