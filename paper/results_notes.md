# Benchmark Execution Notes & Result Transfer Log

Use this template to record session metadata and transfer key metrics from Kaggle outputs to the paper repository.

---

## Model Evaluation Run Record

- **Model Identifier**: `[e.g., Qwen/Qwen2-VL-2B-Instruct]`
- **Evaluation Date**: `YYYY-MM-DD`
- **Kaggle Session / Account**: `[e.g., Account A, Session 1]`
- **GPU Accelerator Used**: `[e.g., Tesla T4 x2]`
- **Git Commit / Tag**: `[e.g., v0.9-runner / be74575]`
- **Dataset Sample ID File**: `results/derived/main_ids.json (N=500)`
- **Evaluation Seeds**: `[e.g., 0, 1, 2]`
- **Assistant Prefix (if applicable)**: `[e.g., answer_colon or None]`

---

## Pre-Flight Check Summary (--check)
- **Mean Letter Mass**: `0.XXXX` ($\ge 0.90$ threshold)
- **Clean Accuracy (10 items)**: `XX.X%`
- **Text-Only Accuracy (10 items)**: `XX.X%`
- **Check Status**: `[PASS / FAIL]`

---

## Key Transfer Metrics (from results/derived/vlm_main_summary.csv)

| Corruption | Seed(s) | Clean Acc | Acc at S=1 | Null Removed Acc | $S^*$ Crossing | Sig. $S^*$ | Misalignment Index (MI) | 95% Bootstrap CI |
|---|---|---|---|---|---|---|---|---|
| Gaussian | 0, 1, 2 | 0.XXXX | 0.XXXX | 0.XXXX | 0.XXX | 0.XXX | 0.XXXX | [0.XXXX, 0.XXXX] |
| Patch | 0, 1, 2 | 0.XXXX | 0.XXXX | 0.XXXX | 0.XXX | 0.XXX | 0.XXXX | [0.XXXX, 0.XXXX] |
| SwapBlend | 0, 1, 2 | 0.XXXX | 0.XXXX | 0.XXXX | 0.XXX | 0.XXX | 0.XXXX | [0.XXXX, 0.XXXX] |

---

## Output Archive Verification
- **Output Zip Filename**: `[e.g., vlm_main_raw_qwen.zip]`
- **SHA-256 Checksum**: `...`
- **Extracted Shards**: 20 shards x 3 corruptions = 60 shards verified.
