# Kaggle Notebook Instructions: VLM Reliance-Utility Pilot

Run these cells in a Kaggle Notebook with a GPU accelerator (e.g. **GPU T4 x2** or **P100**).

---

### Cell 1: Clone Repository and Checkout Tag

> **Note on Private Repositories**: If the GitHub repository is private, add your GitHub Personal Access Token to Kaggle via **Add-ons -> Secrets** with the label `GITHUB_TOKEN`.

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

!git clone {clone_url}
!cd reliance-utility && git checkout v0.5-vlm-pilot-code
!pip install -U transformers accelerate datasets pillow
```

---

### Cell 2: Verify GPU and Qwen2-VL Environment

```python
import torch
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Device name:", torch.cuda.get_device_name(0))
    print("Memory allocated:", torch.cuda.memory_allocated(0) / 1e9, "GB")
```

---

### Cell 3: Execute VLM Pilot Benchmark Runner

Runs Qwen2-VL-2B-Instruct on the sampled $N=200$ A-OKVQA validation questions across both corruptions (`gaussian`, `patch`) and severity grid ($S \in [0, 1]$ in 9 steps). Raw results are saved as resumable shards to `/kaggle/working/results/raw/vlm_pilot`.

```bash
%%bash
cd /kaggle/working/reliance-utility

python scripts/run_vlm_pilot.py \
    --model Qwen/Qwen2-VL-2B-Instruct \
    --n 200 \
    --grid 9 \
    --corruptions gaussian,patch \
    --seed 0 \
    --shard_size 25 \
    --out /kaggle/working/results/raw/vlm_pilot
```

---

### Cell 4: Compress Raw Results for Local Analysis

```bash
%%bash
cd /kaggle/working
zip -r vlm_pilot_raw_results.zip results/raw/vlm_pilot
ls -lh vlm_pilot_raw_results.zip
```

Download `vlm_pilot_raw_results.zip` from the Kaggle Output file viewer, unzip into your local `results/raw/vlm_pilot/` directory, and run `python scripts/analyze_vlm_pilot.py`.
