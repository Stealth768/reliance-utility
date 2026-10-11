# Kaggle Notebook Instructions: VLM Reliance-Utility Main Benchmark (N=500)

Run these cells in a Kaggle Notebook with a GPU accelerator (**GPU T4 x2** or **P100**).

> **IMPORTANT REMINDERS**:
> 1. Run `--check` before any full run. If letter mass < 0.90, the model is rejected.
> 2. Download and save the output zip file (`vlm_main_raw_<model>.zip`) **immediately after EACH model finishes**.

---

### Cell 1: Clone Repository, Clean Remote, Checkout Tag & Install Dependencies

```python
import os
from kaggle_secrets import UserSecretsClient

try:
    user_secrets = UserSecretsClient()
    github_token = user_secrets.get_secret("GITHUB_TOKEN")
    clone_url = f"https://{github_token}@github.com/Stealth768/reliance-utility.git"
    print("Using GITHUB_TOKEN secret from Kaggle.")
except Exception:
    clone_url = "https://github.com/Stealth768/reliance-utility.git"
    print("Attempting public clone.")

!git clone {clone_url} /kaggle/working/reliance-utility
!git -C /kaggle/working/reliance-utility remote set-url origin https://github.com/Stealth768/reliance-utility.git
!git -C /kaggle/working/reliance-utility fetch --tags
!git -C /kaggle/working/reliance-utility checkout v0.9-runner

# Install required dependencies (including 4-bit quantization support)
!pip install -q -U transformers accelerate datasets pillow torchvision bitsandbytes
```

---

### Cell 2: Verify GPU Accelerator

```python
import torch

cuda_ok = torch.cuda.is_available()
print("CUDA Available:", cuda_ok)
if cuda_ok:
    print("Device Name:", torch.cuda.get_device_name(0))
    print("Device Count:", torch.cuda.device_count())
else:
    raise RuntimeError("GPU not detected! Turn on GPU in Session options -> Accelerator.")
```

---

### Cell 3a: Pre-Flight Check — Qwen2-VL-2B-Instruct

```bash
%%bash
cd /kaggle/working/reliance-utility
python scripts/run_vlm_pilot.py \
    --model Qwen/Qwen2-VL-2B-Instruct \
    --check
```

---

### Cell 3b: Pre-Flight Check — SmolVLM-Instruct (Two Prefix Variants)

```bash
%%bash
cd /kaggle/working/reliance-utility
echo "=== Testing Variant 1: answer_colon ==="
python scripts/run_vlm_pilot.py \
    --model HuggingFaceTB/SmolVLM-Instruct \
    --assistant_prefix answer_colon \
    --check 2>&1

echo "=== Testing Variant 2: the_answer_is ==="
python scripts/run_vlm_pilot.py \
    --model HuggingFaceTB/SmolVLM-Instruct \
    --assistant_prefix the_answer_is \
    --check 2>&1
```

---

### Cell 3c: Pre-Flight Check — LLaVA-1.5-7B (4-bit NF4)

```bash
%%bash
cd /kaggle/working/reliance-utility
echo "=== Testing Variant 1: answer_colon ==="
python scripts/run_vlm_pilot.py \
    --model llava-hf/llava-1.5-7b-hf \
    --assistant_prefix answer_colon \
    --check 2>&1

echo "=== Testing Variant 2: the_answer_is ==="
python scripts/run_vlm_pilot.py \
    --model llava-hf/llava-1.5-7b-hf \
    --assistant_prefix the_answer_is \
    --check 2>&1
```

---

### Cell 4: Model 1 Execution — Qwen2-VL-2B-Instruct (N=500, Seeds 0, 1, 2)

```bash
%%bash
cd /kaggle/working/reliance-utility

for SEED in 0 1 2; do
    echo "Running Qwen2-VL-2B with seed ${SEED}..."
    python scripts/run_vlm_pilot.py \
        --model Qwen/Qwen2-VL-2B-Instruct \
        --data_ids results/derived/main_ids.json \
        --n 500 \
        --grid 9 \
        --corruptions gaussian,patch,swapblend \
        --seed ${SEED} \
        --shard_size 25 \
        --out /kaggle/working/results/raw/vlm_main
done
```

```bash
%%bash
cd /kaggle/working
zip -r vlm_main_raw_qwen.zip results/raw/vlm_main/Qwen2-VL-2B-Instruct*.json results/raw/vlm_main/manifest_Qwen2-VL-2B-Instruct*
ls -lh vlm_main_raw_qwen.zip
```
> **DOWNLOAD NOW**: Download `vlm_main_raw_qwen.zip` from Kaggle Output viewer before proceeding.

---

### Cell 5: Model 2 Execution — SmolVLM-Instruct (N=500, Seeds 0, 1, 2)
*(Set `--assistant_prefix` to the variant that passed the $\ge 0.90$ mass check)*

```bash
%%bash
cd /kaggle/working/reliance-utility

for SEED in 0 1 2; do
    echo "Running SmolVLM-Instruct with seed ${SEED}..."
    python scripts/run_vlm_pilot.py \
        --model HuggingFaceTB/SmolVLM-Instruct \
        --assistant_prefix answer_colon \
        --data_ids results/derived/main_ids.json \
        --n 500 \
        --grid 9 \
        --corruptions gaussian,patch,swapblend \
        --seed ${SEED} \
        --shard_size 25 \
        --out /kaggle/working/results/raw/vlm_main
done
```

```bash
%%bash
cd /kaggle/working
zip -r vlm_main_raw_smolvlm.zip results/raw/vlm_main/SmolVLM-Instruct*.json results/raw/vlm_main/manifest_SmolVLM-Instruct*
ls -lh vlm_main_raw_smolvlm.zip
```
> **DOWNLOAD NOW**: Download `vlm_main_raw_smolvlm.zip` from Kaggle Output viewer.

---

### Cell 6: Model 3 Execution — LLaVA-1.5-7B (4-bit NF4, N=500, Seeds 0, 1, 2)

```bash
%%bash
cd /kaggle/working/reliance-utility

for SEED in 0 1 2; do
    echo "Running LLaVA-1.5-7B with seed ${SEED}..."
    python scripts/run_vlm_pilot.py \
        --model llava-hf/llava-1.5-7b-hf \
        --assistant_prefix answer_colon \
        --data_ids results/derived/main_ids.json \
        --n 500 \
        --grid 9 \
        --corruptions gaussian,patch,swapblend \
        --seed ${SEED} \
        --shard_size 25 \
        --out /kaggle/working/results/raw/vlm_main
done
```

```bash
%%bash
cd /kaggle/working
zip -r vlm_main_raw_llava.zip results/raw/vlm_main/llava-1.5-7b-hf*.json results/raw/vlm_main/manifest_llava-1.5-7b-hf*
ls -lh vlm_main_raw_llava.zip
```
> **DOWNLOAD NOW**: Download `vlm_main_raw_llava.zip` from Kaggle Output viewer.

---

### Local Post-Processing
After downloading all archives to your local machine:
```bash
python scripts/merge_shards.py --src /path/to/zips --out results/raw/vlm_main
python scripts/make_all.py
```
