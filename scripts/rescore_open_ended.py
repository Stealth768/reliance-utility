import sys, os, ast, json, glob
import pandas as pd
import numpy as np

sys.path.insert(0, os.getcwd())
from src.metrics.open_ended import vqa_soft_accuracy, normalize_answer

# 1. Load parquet ground truth direct_answers
parquet_url = "https://huggingface.co/api/datasets/HuggingFaceM4/A-OKVQA/parquet/default/validation/0.parquet"
print("Loading A-OKVQA validation ground truth from parquet...")
df = pd.read_parquet(parquet_url)

gt_direct_answers = {}
for _, row in df.iterrows():
    qid = str(row["question_id"])
    raw_da = row["direct_answers"]
    if isinstance(raw_da, str):
        try:
            parsed_da = ast.literal_eval(raw_da)
        except Exception:
            parsed_da = [raw_da]
    else:
        parsed_da = list(raw_da)
    gt_direct_answers[qid] = parsed_da

print(f"Loaded ground truth for {len(gt_direct_answers)} questions.")

# 2. Rescore open-ended shards
repo = "."
open_dir = os.path.join(repo, "results", "raw", "vlm_open")

for model, m_pat in [
    ("Qwen2-VL-2B", "open_Qwen2-VL-2B-Instruct"),
    ("SmolVLM", "open_SmolVLM-Instruct"),
    ("LLaVA-1.5-7B", "open_llava-1.5-7b-hf"),
]:
    print("=" * 70)
    print(f"CORRECTED SCORING FOR: {model}")
    print("=" * 70)
    for corr in ["swapblend", "gaussian"]:
        files = sorted(glob.glob(os.path.join(open_dir, f"{m_pat}_{corr}_seed0_shard*.json")))
        if not files:
            continue
        items = []
        for f in files:
            with open(f, "r") as fp:
                items.extend(json.load(fp)["items"])
        
        n = len(items)
        clean_scores = []
        null_scores = []
        s1_scores = []
        
        for it in items:
            qid = str(it["question_id"])
            da = gt_direct_answers.get(qid, [])
            
            c_ans = it.get("clean_answer", "")
            n_ans = it.get("null_answers", {}).get("removed", "")
            s_ans = it.get("answers_per_S", [])[-1] if it.get("answers_per_S") else ""
            
            clean_scores.append(vqa_soft_accuracy(c_ans, da))
            null_scores.append(vqa_soft_accuracy(n_ans, da))
            s1_scores.append(vqa_soft_accuracy(s_ans, da))
            
        clean_scores = np.array(clean_scores)
        null_scores = np.array(null_scores)
        s1_scores = np.array(s1_scores)
        u1_scores = s1_scores - null_scores
        
        # Bootstrap 95% CI
        rng = np.random.RandomState(42)
        boot_u1 = []
        for _ in range(1000):
            idx = rng.randint(0, n, size=n)
            boot_u1.append(np.mean(u1_scores[idx]))
        boot_u1 = np.array(boot_u1)
        ci_u1 = (float(np.percentile(boot_u1, 2.5)), float(np.percentile(boot_u1, 97.5)))
        
        print(f"\nModel {model} | Corr: {corr} (N={n})")
        print(f"  True Clean Accuracy: {np.mean(clean_scores)*100:.2f}% (non-zero: {np.count_nonzero(clean_scores)}/{n})")
        print(f"  True Null  Accuracy: {np.mean(null_scores)*100:.2f}% (non-zero: {np.count_nonzero(null_scores)}/{n})")
        print(f"  True S=1   Accuracy: {np.mean(s1_scores)*100:.2f}% (non-zero: {np.count_nonzero(s1_scores)}/{n})")
        print(f"  True U(1) Mean: {np.mean(u1_scores):+.4f} (95% CI: [{ci_u1[0]:+.4f}, {ci_u1[1]:+.4f}])")
        
        # Sample first 3
        for i in range(min(3, n)):
            qid = str(items[i]["question_id"])
            da = gt_direct_answers.get(qid, [])[:3]
            print(f"    Q{i}: Clean='{items[i]['clean_answer']}' -> score={clean_scores[i]:.2f} | GT={da}")
