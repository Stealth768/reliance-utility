# Kaggle Multi-Account Orchestration Runbook

Complete operations runbook for running all VLM reliance-utility benchmark experiments across multiple Kaggle GPU sessions with zero shard collisions.

---

## Multi-Account Allocation Matrix (No Overlapping Shards)

| Job ID | Account / Session | Target Accelerator | Model / Task | Seeds | Shard Count | Tag | Est. Runtime | Output Artifact |
|---|---|---|---|---|---|---|---|---|
| **JOB-1A** | Account A (Sess 1) | GPU T4 x2 / P100 | Qwen2-VL-2B | Seed 0 | 20 shards x 3 corruptions | `v0.9-runner` | ~3.5 hours | `vlm_qwen_seed0.zip` |
| **JOB-1B** | Account A (Sess 2) | GPU T4 x2 / P100 | Qwen2-VL-2B | Seeds 1, 2 | 20 shards x 3 x 2 seeds | `v0.9-runner` | ~7 hours | `vlm_qwen_seeds1_2.zip` |
| **JOB-2A** | Account B (Sess 1) | GPU T4 x2 / P100 | SmolVLM-Instruct | Check mode | 10 items (2 variants) | `v0.9-runner` | ~5 mins | Console output |
| **JOB-2B** | Account B (Sess 2) | GPU T4 x2 / P100 | SmolVLM-Instruct | Seeds 0, 1, 2 | 20 shards x 3 x 3 seeds | `v0.9-runner` | ~6.5 hours | `vlm_smolvlm_all.zip` |
| **JOB-3A** | Account C (Sess 1) | GPU T4 x2 / P100 | LLaVA-1.5-7B (4-bit) | Check mode | 10 items (2 variants) | `v0.9-runner` | ~10 mins | Console output |
| **JOB-3B** | Account C (Sess 2) | GPU T4 x2 / P100 | LLaVA-1.5-7B (4-bit) | Seeds 0, 1, 2 | 20 shards x 3 x 3 seeds | `v0.9-runner` | ~8.5 hours | `vlm_llava_all.zip` |
| **JOB-4**  | Any Account / Local | GPU / CPU | Modality Dropout | Seeds 0–4 | 4 $p$-values x 5 seeds | `v0.11-dropout-code` | ~1.5 hours | `fusion_raw.zip` |
| **LOCAL**  | Local Machine | CPU | Merge & Analysis | All | Merge + Tables + Figures | `v0.10-analysis` | ~3 mins | Derived summary CSV/PDFs |

> **IMPORTANT SECURITY & POLICY NOTE**:
> Each Kaggle account must have its own independent phone verification and its own read-only GitHub token stored in Kaggle User Secrets (`GITHUB_TOKEN`). Do not reuse credentials across accounts.

---

## Account Setup Cell (Run first in every Kaggle session)

```python
import os
from kaggle_secrets import UserSecretsClient

try:
    user_secrets = UserSecretsClient()
    github_token = user_secrets.get_secret("GITHUB_TOKEN")
    clone_url = f"https://{github_token}@github.com/Stealth768/reliance-utility.git"
except Exception:
    clone_url = "https://github.com/Stealth768/reliance-utility.git"

!git clone {clone_url} /kaggle/working/reliance-utility
!git -C /kaggle/working/reliance-utility remote set-url origin https://github.com/Stealth768/reliance-utility.git
!git -C /kaggle/working/reliance-utility fetch --tags
!git -C /kaggle/working/reliance-utility checkout v0.9-runner
!pip install -q -U transformers accelerate datasets pillow torchvision bitsandbytes
```

```python
import torch
print("CUDA Available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Device Name:", torch.cuda.get_device_name(0))
    print("Device Count:", torch.cuda.device_count())
```

---

## JOB-1A: Qwen2-VL-2B-Instruct — Seed 0 (Account A, Session 1)

```bash
%%bash
cd /kaggle/working/reliance-utility
python scripts/run_vlm_pilot.py \
    --model Qwen/Qwen2-VL-2B-Instruct \
    --data_ids results/derived/main_ids.json \
    --n 500 \
    --grid 9 \
    --corruptions gaussian,patch,swapblend \
    --seed 0 \
    --shard_size 25 \
    --out /kaggle/working/results/raw/vlm_main
```

```bash
%%bash
cd /kaggle/working
zip -r vlm_qwen_seed0.zip results/raw/vlm_main/Qwen2-VL-2B-Instruct*seed0*.json results/raw/vlm_main/manifest_Qwen2-VL-2B-Instruct*seed0*
ls -lh vlm_qwen_seed0.zip
```

---

## JOB-1B: Qwen2-VL-2B-Instruct — Seeds 1 & 2 (Account A, Session 2)

```bash
%%bash
cd /kaggle/working/reliance-utility
python scripts/run_vlm_pilot.py \
    --model Qwen/Qwen2-VL-2B-Instruct \
    --data_ids results/derived/main_ids.json \
    --n 500 \
    --grid 9 \
    --corruptions gaussian,patch,swapblend \
    --seed 1 \
    --shard_size 25 \
    --out /kaggle/working/results/raw/vlm_main

python scripts/run_vlm_pilot.py \
    --model Qwen/Qwen2-VL-2B-Instruct \
    --data_ids results/derived/main_ids.json \
    --n 500 \
    --grid 9 \
    --corruptions gaussian,patch,swapblend \
    --seed 2 \
    --shard_size 25 \
    --out /kaggle/working/results/raw/vlm_main
```

```bash
%%bash
cd /kaggle/working
zip -r vlm_qwen_seeds1_2.zip results/raw/vlm_main/Qwen2-VL-2B-Instruct*seed1*.json results/raw/vlm_main/Qwen2-VL-2B-Instruct*seed2*.json results/raw/vlm_main/manifest_Qwen2-VL-2B-Instruct*seed1* results/raw/vlm_main/manifest_Qwen2-VL-2B-Instruct*seed2*
ls -lh vlm_qwen_seeds1_2.zip
```

---

## JOB-2A: SmolVLM-Instruct Pre-Flight Check (Account B, Session 1)

```bash
%%bash
cd /kaggle/working/reliance-utility
echo "=== Check SmolVLM: answer_colon ==="
python scripts/run_vlm_pilot.py \
    --model HuggingFaceTB/SmolVLM-Instruct \
    --assistant_prefix answer_colon \
    --check 2>&1

echo "=== Check SmolVLM: the_answer_is ==="
python scripts/run_vlm_pilot.py \
    --model HuggingFaceTB/SmolVLM-Instruct \
    --assistant_prefix the_answer_is \
    --check 2>&1
```

---

## JOB-2B: SmolVLM-Instruct Full Run — Seeds 0, 1, 2 (Account B, Session 2)
*(Replace `--assistant_prefix answer_colon` with the chosen variant if `the_answer_is` attained higher mass $\ge 0.90$)*

```bash
%%bash
cd /kaggle/working/reliance-utility
for S in 0 1 2; do
    python scripts/run_vlm_pilot.py \
        --model HuggingFaceTB/SmolVLM-Instruct \
        --assistant_prefix answer_colon \
        --data_ids results/derived/main_ids.json \
        --n 500 \
        --grid 9 \
        --corruptions gaussian,patch,swapblend \
        --seed $S \
        --shard_size 25 \
        --out /kaggle/working/results/raw/vlm_main
done
```

```bash
%%bash
cd /kaggle/working
zip -r vlm_smolvlm_all.zip results/raw/vlm_main/SmolVLM-Instruct*.json results/raw/vlm_main/manifest_SmolVLM-Instruct*
ls -lh vlm_smolvlm_all.zip
```

---

## JOB-3A: LLaVA-1.5-7B Pre-Flight Check (Account C, Session 1)

```bash
%%bash
cd /kaggle/working/reliance-utility
echo "=== Check LLaVA: answer_colon ==="
python scripts/run_vlm_pilot.py \
    --model llava-hf/llava-1.5-7b-hf \
    --assistant_prefix answer_colon \
    --check 2>&1

echo "=== Check LLaVA: the_answer_is ==="
python scripts/run_vlm_pilot.py \
    --model llava-hf/llava-1.5-7b-hf \
    --assistant_prefix the_answer_is \
    --check 2>&1
```

---

## JOB-3B: LLaVA-1.5-7B Full Run — Seeds 0, 1, 2 (Account C, Session 2)

```bash
%%bash
cd /kaggle/working/reliance-utility
for S in 0 1 2; do
    python scripts/run_vlm_pilot.py \
        --model llava-hf/llava-1.5-7b-hf \
        --assistant_prefix answer_colon \
        --data_ids results/derived/main_ids.json \
        --n 500 \
        --grid 9 \
        --corruptions gaussian,patch,swapblend \
        --seed $S \
        --shard_size 25 \
        --out /kaggle/working/results/raw/vlm_main
done
```

```bash
%%bash
cd /kaggle/working
zip -r vlm_llava_all.zip results/raw/vlm_main/llava-1.5-7b-hf*.json results/raw/vlm_main/manifest_llava-1.5-7b-hf*
ls -lh vlm_llava_all.zip
```

---

## JOB-4: Modality Dropout Fusion Study

```bash
%%bash
cd /kaggle/working/reliance-utility
git checkout v0.11-dropout-code
python scripts/train_fusion.py --dataset synthetic --p 0,0.1,0.25,0.5 --seeds 0,1,2,3,4 --epochs 30 --out results/raw/fusion
python scripts/train_fusion.py --dataset fashionmnist --p 0,0.1,0.25,0.5 --seeds 0,1,2,3,4 --epochs 20 --out results/raw/fusion --device cuda
```

```bash
%%bash
cd /kaggle/working/reliance-utility
zip -r fusion_raw.zip results/raw/fusion
ls -lh fusion_raw.zip
```

---

## Local Assembly: Merge & Analysis

After downloading the zip files to your local workstation:

```bash
# 1. Merge all downloaded zip archives safely into results/raw/vlm_main
python scripts/merge_shards.py \
    --src /path/to/vlm_qwen_seed0.zip \
    --src /path/to/vlm_qwen_seeds1_2.zip \
    --src /path/to/vlm_smolvlm_all.zip \
    --src /path/to/vlm_llava_all.zip \
    --out results/raw/vlm_main

# 2. Verify complete coverage for each condition
python scripts/check_run_complete.py --dir results/raw/vlm_main --model Qwen/Qwen2-VL-2B-Instruct --corruption gaussian --seed 0
python scripts/check_run_complete.py --dir results/raw/vlm_main --model Qwen/Qwen2-VL-2B-Instruct --corruption patch --seed 0
python scripts/check_run_complete.py --dir results/raw/vlm_main --model Qwen/Qwen2-VL-2B-Instruct --corruption swapblend --seed 0

# 3. Regenerate all derived paper summaries, LaTeX tables, and vector figures
python scripts/make_all.py

# 4. Verify all numbers in paper/*.tex agree with generated tables
python scripts/check_numbers.py
```
