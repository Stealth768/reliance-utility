#!/usr/bin/env python3
"""Fix direct_answers parsing in open-ended shard JSON files.

The HuggingFace A-OKVQA validation parquet stores `direct_answers` as a string representation
of a list (e.g. "['cat', 'kitten']"). During benchmark execution, this string was iterated
character-by-character, causing vqa_soft_accuracy to evaluate near 0.0.
This script re-computes `clean_utility`, `null_utility`, and `utility_per_S` for all items
using properly parsed list `direct_answers`.
"""

import ast
import json
import os
import sys
from pathlib import Path
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.metrics.open_ended import vqa_soft_accuracy

# 1. Load ground truth direct_answers from parquet
parquet_url = "https://huggingface.co/api/datasets/HuggingFaceM4/A-OKVQA/parquet/default/validation/0.parquet"
print(f"Loading ground truth from {parquet_url}...")
df = pd.read_parquet(parquet_url)
gt_map = {}
for _, row in df.iterrows():
    qid = str(row["question_id"])
    raw_da = row["direct_answers"]
    if isinstance(raw_da, str):
        try:
            da_list = ast.literal_eval(raw_da)
        except Exception:
            da_list = [raw_da]
    else:
        da_list = list(raw_da)
    gt_map[qid] = da_list
print(f"Loaded ground truth for {len(gt_map)} questions.")

# 2. Update each shard in results/raw/vlm_open
open_dir = REPO_ROOT / "results" / "raw" / "vlm_open"
shard_files = sorted(open_dir.glob("*.json"))
print(f"Updating {len(shard_files)} open-ended shard files...")

fixed_count = 0
for sf in shard_files:
    if "manifest" in sf.name:
        continue
    with open(sf, "r", encoding="utf-8") as fp:
        data = json.load(fp)

    items = data.get("items", [])
    for it in items:
        qid = str(it["question_id"])
        da = gt_map.get(qid, [])

        # Clean utility
        c_ans = it.get("clean_answer", "")
        it["clean_utility"] = vqa_soft_accuracy(c_ans, da)

        # Null utility
        null_answers = it.get("null_answers", {})
        null_u = {}
        for nk, n_ans in null_answers.items():
            null_u[nk] = vqa_soft_accuracy(n_ans, da)
        it["null_utility"] = null_u

        # Utility per S
        answers_per_s = it.get("answers_per_S", [])
        u_per_s = [vqa_soft_accuracy(ans, da) for ans in answers_per_s]
        it["utility_per_S"] = u_per_s

    # Save updated shard
    with open(sf, "w", encoding="utf-8") as fp:
        json.dump(data, fp, indent=2)
    fixed_count += 1

print(f"Successfully rescored and updated {fixed_count} shard files in results/raw/vlm_open/.")
