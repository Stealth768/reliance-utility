# Kaggle Notebook Instructions: VLM Reliance-Utility Main Benchmark (N=500)

Run these cells in a Kaggle Notebook with a GPU accelerator (e.g. **GPU T4 x2** or **P100**).

> **IMPORTANT REMINDER**:
> Download and save the output zip file (`vlm_main_raw_<model>.zip`) **after EACH model finishes**. Kaggle sessions can disconnect or run out of execution quota, so saving per-model outputs ensures no compute is lost.

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
# Immediately strip the token from git config so it is never saved to disk or exported
!git -C /kaggle/working/reliance-utility remote set-url origin https://github.com/Stealth768/reliance-utility.git
# Checkout the reproducible main code tag
!git -C /kaggle/working/reliance-utility checkout v0.7-main-code

# Install latest packages
!pip install -q -U transformers accelerate datasets pillow torchvision
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

### Cell 3: Pre-Flight Model Checks (--check mode)

Runs 10 questions clean and text-only per model, prints mean letter mass, clean accuracy, and text-only accuracy. **Refuses to continue if letter mass < 0.90**.

```bash
%%bash
cd /kaggle/working/reliance-utility

# 1. Check Qwen2-VL-2B
python scripts/run_vlm_pilot.py \
    --model Qwen/Qwen2-VL-2B-Instruct \
    --check

# 2. Check SmolVLM-Instruct
python scripts/run_vlm_pilot.py \
    --model HuggingFaceTB/SmolVLM-Instruct \
    --check

# 3. Check InternVL2-2B
python scripts/run_vlm_pilot.py \
    --model OpenGVLab/InternVL2-2B \
    --check
```

---

### Cell 4: Model 1 Execution — Qwen2-VL-2B-Instruct (N=500)

Evaluates on disjoint N=500 sample across `gaussian`, `patch`, and `swapblend` corruptions (8 shards of 25 = 20 shards total):

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

#### Zip and Download Model 1 Results:
```bash
%%bash
cd /kaggle/working
zip -r vlm_main_raw_qwen.zip results/raw/vlm_main/Qwen2-VL-2B-Instruct*.json results/raw/vlm_main/manifest_Qwen2-VL-2B-Instruct*
ls -lh vlm_main_raw_qwen.zip
```
> **REMINDER**: Download `vlm_main_raw_qwen.zip` from the Kaggle Output viewer right now before running the next model.

---

### Cell 5: Model 2 Execution — SmolVLM-Instruct (N=500)

```bash
%%bash
cd /kaggle/working/reliance-utility

python scripts/run_vlm_pilot.py \
    --model HuggingFaceTB/SmolVLM-Instruct \
    --data_ids results/derived/main_ids.json \
    --n 500 \
    --grid 9 \
    --corruptions gaussian,patch,swapblend \
    --seed 0 \
    --shard_size 25 \
    --out /kaggle/working/results/raw/vlm_main
```

#### Zip and Download Model 2 Results:
```bash
%%bash
cd /kaggle/working
zip -r vlm_main_raw_smolvlm.zip results/raw/vlm_main/SmolVLM-Instruct*.json results/raw/vlm_main/manifest_SmolVLM-Instruct*
ls -lh vlm_main_raw_smolvlm.zip
```
> **REMINDER**: Download `vlm_main_raw_smolvlm.zip` from the Kaggle Output viewer right now before running the next model.

---

### Cell 6: Model 3 Execution — InternVL2-2B (N=500)

```bash
%%bash
cd /kaggle/working/reliance-utility

python scripts/run_vlm_pilot.py \
    --model OpenGVLab/InternVL2-2B \
    --data_ids results/derived/main_ids.json \
    --n 500 \
    --grid 9 \
    --corruptions gaussian,patch,swapblend \
    --seed 0 \
    --shard_size 25 \
    --out /kaggle/working/results/raw/vlm_main
```

#### Zip and Download Model 3 Results:
```bash
%%bash
cd /kaggle/working
zip -r vlm_main_raw_internvl.zip results/raw/vlm_main/InternVL2-2B*.json results/raw/vlm_main/manifest_InternVL2-2B*
ls -lh vlm_main_raw_internvl.zip
```
> **REMINDER**: Download `vlm_main_raw_internvl.zip` from the Kaggle Output viewer now.

---

### Local Execution After Downloads
Once you extract the zip archives into your local `results/raw/vlm_main/`:
```bash
python scripts/analyze_vlm_main.py --raw_dir results/raw/vlm_main
```
This generates the full cross-model summary table, exports `results/derived/vlm_main_summary.csv`, and produces vector PDFs in `figures/`.
