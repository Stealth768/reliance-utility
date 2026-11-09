# Kaggle Notebook Instructions: Reliance-Utility Extension Phase

Instructions and executable cells for executing Extension Phase experiments (AI2D Second MC Dataset, Extra Swapped Pairings, Open-Ended Free Generation) across Kaggle GPU sessions.

---

## 2-Account Unified Job Allocation

To complete all Extension runs in parallel using 2 Kaggle accounts (with 12-hour session limits and GPU T4 x2 accelerators):

| Account | Tasks Included | Models | Dataset | Shards / Samples | Est. Runtime | Output Artifact |
|---|---|---|---|---|---|---|
| **Account 1** | Open-Ended Generation + Extra Swapped Pairings | Qwen2-VL-2B, SmolVLM, LLaVA-1.5 | A-OKVQA | Open N=500 (seed 0); Pairings seeds 11, 12, 13 | ~5.5 hours | `account1_open_and_pairings.zip` |
| **Account 2** | AI2D Full Multiple-Choice Suite | Qwen2-VL-2B, SmolVLM, LLaVA-1.5 | AI2D | Main N=500 (seed 0) + Rep N=400 (seed 1) | ~7.5 hours | `account2_ai2d_all.zip` |

---

## Kaggle Environment Note: PyTorch & TorchAudio CUDA Mismatch Fix

Kaggle GPU environments have PyTorch compiled against CUDA 13.2, whereas pre-installed `torchaudio` is built against CUDA 12.8. When HuggingFace `transformers` attempts to import audio utilities, `torchaudio` raises a `RuntimeError` which breaks `AutoProcessor`. 
Running `pip uninstall -y torchaudio` resolves this cleanly without impacting vision or text models.

---

## Account 1: Single Self-Contained Cell

Copy and run this entire Python cell in Account 1:

```python
import os
import sys
import time
import base64
import subprocess
from IPython.display import HTML, Javascript, display

print("=" * 60)
print("ACCOUNT 1: OPEN-ENDED GENERATION + EXTRA PAIRINGS")
print("=" * 60)

# -------------------------------------------------------------
# STEP 1: FIX KAGGLE CUDA CONFLICT & INSTALL DEPENDENCIES
# -------------------------------------------------------------
print("\n>>> Step 1/5: Fixing CUDA conflict and installing dependencies...")
subprocess.run("pip uninstall -y torchaudio", shell=True)
subprocess.run(
    "pip install -q -U transformers accelerate datasets pillow torchvision bitsandbytes",
    shell=True, check=True
)

# -------------------------------------------------------------
# STEP 2: CLONE OR UPDATE REPOSITORY (EXTENSION BRANCH)
# -------------------------------------------------------------
print("\n>>> Step 2/5: Synchronizing GitHub repository (extension branch)...")
repo_dir = "/kaggle/working/reliance-utility"
if not os.path.exists(repo_dir):
    subprocess.run(
        f"git clone https://github.com/Stealth768/reliance-utility.git {repo_dir}",
        shell=True, check=True
    )
else:
    print("Repository exists on disk. Resetting and pulling latest...")
    subprocess.run(
        f"cd {repo_dir} && git reset --hard HEAD",
        shell=True, check=True
    )

subprocess.run(
    f"cd {repo_dir} && git fetch --all --tags && git checkout extension && git pull origin extension",
    shell=True, check=True
)

# Verify GPU and AutoProcessor import
import torch
from transformers import AutoProcessor
assert torch.cuda.is_available(), "GPU accelerator not detected! Please enable GPU T4 x2 or P100 in Kaggle settings."
print(f"CUDA GPU Verified: {torch.cuda.get_device_name(0)}")
print("AutoProcessor cleanly verified with zero CUDA version conflicts!")

# -------------------------------------------------------------
# STEP 3: PRE-FLIGHT VERIFICATION CHECKS
# -------------------------------------------------------------
print("\n>>> Step 3/5: Running pre-flight validation checks...")
check_commands = [
    f"cd {repo_dir} && python scripts/run_vlm_open.py --model Qwen/Qwen2-VL-2B-Instruct --check",
    f"cd {repo_dir} && python scripts/run_vlm_open.py --model HuggingFaceTB/SmolVLM-Instruct --assistant_prefix answer_colon --check"
]
for cmd in check_commands:
    print(f"Executing check: {cmd}")
    subprocess.run(cmd, shell=True, check=True)

# -------------------------------------------------------------
# STEP 4: PRODUCTION RUNS
# -------------------------------------------------------------
print("\n>>> Step 4/5: Running production tasks...")
os.makedirs("/kaggle/working/results/raw/vlm_open", exist_ok=True)
os.makedirs("/kaggle/working/results/raw/vlm_pairings", exist_ok=True)

prod_tasks = [
    # 1. Qwen2-VL-2B Open-Ended (N=500, seed 0)
    (
        "Open-Ended: Qwen2-VL-2B-Instruct (N=500)",
        f"cd {repo_dir} && python scripts/run_vlm_open.py "
        f"--model Qwen/Qwen2-VL-2B-Instruct "
        f"--data_ids results/derived/main_ids.json "
        f"--n 500 --seed 0 --shard_size 25 "
        f"--out /kaggle/working/results/raw/vlm_open"
    ),
    # 2. SmolVLM Open-Ended (N=500, seed 0)
    (
        "Open-Ended: SmolVLM-Instruct (N=500)",
        f"cd {repo_dir} && python scripts/run_vlm_open.py "
        f"--model HuggingFaceTB/SmolVLM-Instruct "
        f"--assistant_prefix answer_colon "
        f"--data_ids results/derived/main_ids.json "
        f"--n 500 --seed 0 --shard_size 25 "
        f"--out /kaggle/working/results/raw/vlm_open"
    ),
]

# 3. Extra swapped pairings (seeds 11, 12, 13) for Qwen2-VL, SmolVLM, LLaVA-1.5
for pair_seed in [11, 12, 13]:
    prod_tasks.append((
        f"Pairing Seed {pair_seed}: Qwen2-VL-2B",
        f"cd {repo_dir} && python scripts/run_vlm_pilot.py "
        f"--model Qwen/Qwen2-VL-2B-Instruct --dataset aokvqa "
        f"--corruptions swapblend --pairing_seed {pair_seed} --seed 0 "
        f"--out /kaggle/working/results/raw/vlm_pairings"
    ))
    prod_tasks.append((
        f"Pairing Seed {pair_seed}: SmolVLM",
        f"cd {repo_dir} && python scripts/run_vlm_pilot.py "
        f"--model HuggingFaceTB/SmolVLM-Instruct --dataset aokvqa "
        f"--assistant_prefix answer_colon "
        f"--corruptions swapblend --pairing_seed {pair_seed} --seed 0 "
        f"--out /kaggle/working/results/raw/vlm_pairings"
    ))
    prod_tasks.append((
        f"Pairing Seed {pair_seed}: LLaVA-1.5-7B",
        f"cd {repo_dir} && python scripts/run_vlm_pilot.py "
        f"--model llava-hf/llava-1.5-7b-hf --dataset aokvqa "
        f"--assistant_prefix answer_colon "
        f"--corruptions swapblend --pairing_seed {pair_seed} --seed 0 "
        f"--out /kaggle/working/results/raw/vlm_pairings"
    ))

total_tasks = len(prod_tasks)
t_start = time.time()
for idx, (title, cmd) in enumerate(prod_tasks, 1):
    print(f"\n[{idx}/{total_tasks}] Starting {title}...")
    t_task = time.time()
    subprocess.run(cmd, shell=True, check=True)
    print(f"[{idx}/{total_tasks}] Completed {title} in {(time.time() - t_task)/60:.1f} min")

print(f"\nAll tasks finished in {(time.time() - t_start)/60:.1f} min!")

# -------------------------------------------------------------
# STEP 5: ZIP SHARDS & TRIGGER AUTOMATIC DOWNLOAD
# -------------------------------------------------------------
print("\n>>> Step 5/5: Packaging shards and triggering browser download...")
zip_name = "account1_open_and_pairings.zip"
zip_path = f"/kaggle/working/{zip_name}"

subprocess.run(
    f"cd /kaggle/working && zip -r -q {zip_name} results/raw/vlm_open results/raw/vlm_pairings",
    shell=True, check=True
)

file_size_mb = os.path.getsize(zip_path) / (1024 * 1024)
print(f"Archive created: {zip_name} ({file_size_mb:.2f} MB)")

with open(zip_path, "rb") as f:
    b64_data = base64.b64encode(f.read()).decode("utf-8")

# JavaScript automatic browser download
js_code = f"""
(function() {{
    var a = document.createElement('a');
    a.href = 'data:application/zip;base64,{b64_data}';
    a.download = '{zip_name}';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
}})();
"""
display(Javascript(js_code))

# Fallback HTML download button
html_code = f"""
<div style="background-color: #e6ffed; border: 2px solid #52c41a; padding: 15px; border-radius: 8px; margin-top: 10px;">
    <h3 style="color: #274916; margin: 0 0 8px 0;"> Download Triggered Automatically!</h3>
    <p style="margin: 0 0 10px 0;">If your browser popup blocker prevented the automatic download, click below:</p>
    <a href="data:application/zip;base64,{b64_data}" download="{zip_name}" 
       style="display: inline-block; background: #1890ff; color: white; padding: 8px 16px; border-radius: 4px; text-decoration: none; font-weight: bold;">
       Click to Download {zip_name} ({file_size_mb:.2f} MB)
    </a>
</div>
"""
display(HTML(html_code))
```

---

## Account 2: Single Self-Contained Cell

Copy and run this entire Python cell in Account 2:

```python
import os
import sys
import time
import base64
import subprocess
from IPython.display import HTML, Javascript, display

print("=" * 60)
print("ACCOUNT 2: AI2D MULTIPLE-CHOICE BENCHMARK (ALL MODELS)")
print("=" * 60)

# -------------------------------------------------------------
# STEP 1: FIX KAGGLE CUDA CONFLICT & INSTALL DEPENDENCIES
# -------------------------------------------------------------
print("\n>>> Step 1/5: Fixing CUDA conflict and installing dependencies...")
subprocess.run("pip uninstall -y torchaudio", shell=True)
subprocess.run(
    "pip install -q -U transformers accelerate datasets pillow torchvision bitsandbytes",
    shell=True, check=True
)

# -------------------------------------------------------------
# STEP 2: CLONE OR UPDATE REPOSITORY (EXTENSION BRANCH)
# -------------------------------------------------------------
print("\n>>> Step 2/5: Synchronizing GitHub repository (extension branch)...")
repo_dir = "/kaggle/working/reliance-utility"
if not os.path.exists(repo_dir):
    subprocess.run(
        f"git clone https://github.com/Stealth768/reliance-utility.git {repo_dir}",
        shell=True, check=True
    )
else:
    print("Repository exists on disk. Resetting and pulling latest...")
    subprocess.run(
        f"cd {repo_dir} && git reset --hard HEAD",
        shell=True, check=True
    )

subprocess.run(
    f"cd {repo_dir} && git fetch --all --tags && git checkout extension && git pull origin extension",
    shell=True, check=True
)

# Verify GPU and AutoProcessor import
import torch
from transformers import AutoProcessor
assert torch.cuda.is_available(), "GPU accelerator not detected! Please enable GPU T4 x2 or P100 in Kaggle settings."
print(f"CUDA GPU Verified: {torch.cuda.get_device_name(0)}")
print("AutoProcessor cleanly verified with zero CUDA version conflicts!")

# -------------------------------------------------------------
# STEP 3: PRE-FLIGHT VERIFICATION CHECKS
# -------------------------------------------------------------
print("\n>>> Step 3/5: Running pre-flight validation checks on AI2D...")
check_commands = [
    f"cd {repo_dir} && python scripts/run_vlm_pilot.py --model Qwen/Qwen2-VL-2B-Instruct --dataset ai2d --check",
    f"cd {repo_dir} && python scripts/run_vlm_pilot.py --model HuggingFaceTB/SmolVLM-Instruct --dataset ai2d --assistant_prefix answer_colon --check",
    f"cd {repo_dir} && python scripts/run_vlm_pilot.py --model llava-hf/llava-1.5-7b-hf --dataset ai2d --assistant_prefix answer_colon --check"
]
for cmd in check_commands:
    print(f"Executing check: {cmd}")
    subprocess.run(cmd, shell=True, check=True)

# -------------------------------------------------------------
# STEP 4: PRODUCTION RUNS
# -------------------------------------------------------------
print("\n>>> Step 4/5: Running production AI2D tasks...")
os.makedirs("/kaggle/working/results/raw/vlm_ai2d", exist_ok=True)

prod_tasks = [
    # 1. Qwen2-VL-2B AI2D Main (N=500, seed 0)
    (
        "AI2D Main: Qwen2-VL-2B-Instruct (N=500)",
        f"cd {repo_dir} && python scripts/run_vlm_pilot.py "
        f"--model Qwen/Qwen2-VL-2B-Instruct "
        f"--dataset ai2d "
        f"--data_ids results/derived/ai2d_main_ids.json "
        f"--n 500 --seed 0 --shard_size 25 "
        f"--out /kaggle/working/results/raw/vlm_ai2d"
    ),
    # 2. Qwen2-VL-2B AI2D Replication (N=400, seed 1)
    (
        "AI2D Replication: Qwen2-VL-2B-Instruct (N=400)",
        f"cd {repo_dir} && python scripts/run_vlm_pilot.py "
        f"--model Qwen/Qwen2-VL-2B-Instruct "
        f"--dataset ai2d "
        f"--data_ids results/derived/ai2d_rep_ids.json "
        f"--n 400 --seed 1 --shard_size 25 "
        f"--out /kaggle/working/results/raw/vlm_ai2d"
    ),
    # 3. SmolVLM AI2D Main (N=500, seed 0)
    (
        "AI2D Main: SmolVLM-Instruct (N=500)",
        f"cd {repo_dir} && python scripts/run_vlm_pilot.py "
        f"--model HuggingFaceTB/SmolVLM-Instruct "
        f"--dataset ai2d "
        f"--assistant_prefix answer_colon "
        f"--data_ids results/derived/ai2d_main_ids.json "
        f"--n 500 --seed 0 --shard_size 25 "
        f"--out /kaggle/working/results/raw/vlm_ai2d"
    ),
    # 4. LLaVA-1.5-7B AI2D Main (N=500, seed 0)
    (
        "AI2D Main: LLaVA-1.5-7B-hf (N=500)",
        f"cd {repo_dir} && python scripts/run_vlm_pilot.py "
        f"--model llava-hf/llava-1.5-7b-hf "
        f"--dataset ai2d "
        f"--assistant_prefix answer_colon "
        f"--data_ids results/derived/ai2d_main_ids.json "
        f"--n 500 --seed 0 --shard_size 25 "
        f"--out /kaggle/working/results/raw/vlm_ai2d"
    ),
]

total_tasks = len(prod_tasks)
t_start = time.time()
for idx, (title, cmd) in enumerate(prod_tasks, 1):
    print(f"\n[{idx}/{total_tasks}] Starting {title}...")
    t_task = time.time()
    subprocess.run(cmd, shell=True, check=True)
    print(f"[{idx}/{total_tasks}] Completed {title} in {(time.time() - t_task)/60:.1f} min")

print(f"\nAll AI2D tasks finished in {(time.time() - t_start)/60:.1f} min!")

# -------------------------------------------------------------
# STEP 5: ZIP SHARDS & TRIGGER AUTOMATIC DOWNLOAD
# -------------------------------------------------------------
print("\n>>> Step 5/5: Packaging shards and triggering browser download...")
zip_name = "account2_ai2d_all.zip"
zip_path = f"/kaggle/working/{zip_name}"

subprocess.run(
    f"cd /kaggle/working && zip -r -q {zip_name} results/raw/vlm_ai2d",
    shell=True, check=True
)

file_size_mb = os.path.getsize(zip_path) / (1024 * 1024)
print(f"Archive created: {zip_name} ({file_size_mb:.2f} MB)")

with open(zip_path, "rb") as f:
    b64_data = base64.b64encode(f.read()).decode("utf-8")

# JavaScript automatic browser download
js_code = f"""
(function() {{
    var a = document.createElement('a');
    a.href = 'data:application/zip;base64,{b64_data}';
    a.download = '{zip_name}';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
}})();
"""
display(Javascript(js_code))

# Fallback HTML download button
html_code = f"""
<div style="background-color: #e6ffed; border: 2px solid #52c41a; padding: 15px; border-radius: 8px; margin-top: 10px;">
    <h3 style="color: #274916; margin: 0 0 8px 0;"> Download Triggered Automatically!</h3>
    <p style="margin: 0 0 10px 0;">If your browser popup blocker prevented the automatic download, click below:</p>
    <a href="data:application/zip;base64,{b64_data}" download="{zip_name}" 
       style="display: inline-block; background: #1890ff; color: white; padding: 8px 16px; border-radius: 4px; text-decoration: none; font-weight: bold;">
       Click to Download {zip_name} ({file_size_mb:.2f} MB)
    </a>
</div>
"""
display(HTML(html_code))
```
