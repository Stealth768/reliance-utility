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
!git -C /kaggle/working/reliance-utility checkout v0.8-llava-code

# Install packages (bitsandbytes required for LLaVA-1.5 4-bit NF4 loading)
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

### Cell 3a: Pre-Flight Check — Qwen2-VL-2B

Runs 10 questions clean and text-only. **Refuses to continue if letter mass < 0.90**.

```bash
%%bash
cd /kaggle/working/reliance-utility
python scripts/run_vlm_pilot.py \
    --model Qwen/Qwen2-VL-2B-Instruct \
    --check
```

---

### Cell 3b: Pre-Flight Check — SmolVLM-Instruct (two prefix variants)

Try `answer_colon` first; if it passes (mass >= 0.90), use it for the full run.
If it fails, try `the_answer_is`. If neither passes, SmolVLM is dropped.

```bash
%%bash
cd /kaggle/working/reliance-utility

echo "=== Variant 1: answer_colon ==="
python scripts/run_vlm_pilot.py \
    --model HuggingFaceTB/SmolVLM-Instruct \
    --assistant_prefix answer_colon \
    --check

echo "=== Variant 2: the_answer_is (run only if variant 1 failed) ==="
# python scripts/run_vlm_pilot.py \
#     --model HuggingFaceTB/SmolVLM-Instruct \
#     --assistant_prefix the_answer_is \
#     --check
```

---

### Cell 3c: Pre-Flight Check — LLaVA-1.5-7B-hf (two prefix variants)

Try `answer_colon` first; if it passes (mass >= 0.90), use it for the full run.
If it fails, try `the_answer_is`. If neither passes, LLaVA is dropped.

```bash
%%bash
cd /kaggle/working/reliance-utility

echo "=== Variant 1: answer_colon ==="
python scripts/run_vlm_pilot.py \
    --model llava-hf/llava-1.5-7b-hf \
    --assistant_prefix answer_colon \
    --check

echo "=== Variant 2: the_answer_is (run only if variant 1 failed) ==="
# python scripts/run_vlm_pilot.py \
#     --model llava-hf/llava-1.5-7b-hf \
#     --assistant_prefix the_answer_is \
#     --check
```

---

### Cell 4: Model 1 Execution — Qwen2-VL-2B-Instruct (N=500)

Evaluates on disjoint N=500 sample across `gaussian`, `patch`, and `swapblend` corruptions (20 shards of 25):

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

> **NOTE**: Replace `--assistant_prefix answer_colon` with whichever variant passed in Cell 3b.

```bash
%%bash
cd /kaggle/working/reliance-utility

python scripts/run_vlm_pilot.py \
    --model HuggingFaceTB/SmolVLM-Instruct \
    --assistant_prefix answer_colon \
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

### Cell 6: Model 3 Execution — LLaVA-1.5-7B-hf (N=500)

> **NOTE**: InternVL2 was dropped due to remote-code incompatibility with transformers>=4.50 (see DEVIATIONS.md).
> LLaVA-1.5-7B-hf replaces it, loaded in 4-bit NF4 quantisation via bitsandbytes.
> Replace `--assistant_prefix answer_colon` with whichever variant passed in Cell 3c.

```bash
%%bash
cd /kaggle/working/reliance-utility

python scripts/run_vlm_pilot.py \
    --model llava-hf/llava-1.5-7b-hf \
    --assistant_prefix answer_colon \
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
zip -r vlm_main_raw_llava.zip results/raw/vlm_main/llava-1.5-7b-hf*.json results/raw/vlm_main/manifest_llava-1.5-7b-hf*
ls -lh vlm_main_raw_llava.zip
```
> **REMINDER**: Download `vlm_main_raw_llava.zip` from the Kaggle Output viewer now.

---

### Cell 7: Local Merge & Analysis (run locally after downloads)

> This cell is for reference only — run locally, not on Kaggle.

```bash
# Merge shards from each model's zip into results/raw/vlm_main/
python scripts/merge_shards.py \
    --src vlm_main_raw_qwen.zip \
    --src vlm_main_raw_smolvlm.zip \
    --src vlm_main_raw_llava.zip \
    --out results/raw/vlm_main

# Run full analysis pipeline
python scripts/make_all.py
```
