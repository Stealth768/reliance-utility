# Kaggle Notebook Instructions: Reliance-Utility Extension Phase

This document contains the execution scripts for all Extension Phase GPU runs across **2 Kaggle accounts** organized into **Wave 1** and **Wave 2**, with automatic browser downloads and clickable fallback download buttons included.

> [!IMPORTANT]
> **Execution Mode**:
> - If running via **"Save & Run All" (Commit mode)**: Kaggle runs headlessly and saves all outputs to `/kaggle/working/` in the Output tab. When you open the committed notebook version or view outputs, the file is ready to download.
> - If viewing the cell outputs in the notebook: The JavaScript and HTML payload triggers automatic browser download + shows a 1-click fallback button!
> - Ensure accelerator is set to **GPU T4 x2** (or GPU P100) and **Internet ON**.

---

## Overview of Allocation

| Wave | Session | Target Models & Tasks | Est. Time | Output Archive |
|---|---|---|---|---|
| **Wave 1** | **Account 1** | Open-Ended Qwen2-VL (3 corruptions) + Extra Swapped Pairings (seeds 11, 12, 13 for Qwen, Smol, LLaVA) | ~7.0h | `account1_wave1.zip` |
| **Wave 1** | **Account 2** | AI2D MC: Qwen Main (N=500) + Qwen Rep (N=400) + SmolVLM Main (N=500) + LLaVA Main (N=500) | ~7.7h | `account2_wave1_ai2d.zip` |
| **Wave 2** | **Account 1** | Open-Ended SmolVLM (2 corruptions: `gaussian,swapblend`, N=500) | ~9.6h | `account1_wave2_smolvlm_open.zip` |
| **Wave 2** | **Account 2** | Open-Ended LLaVA-1.5-7B (2 corruptions: `gaussian,swapblend`, N=500) | ~9.6h | `account2_wave2_llava_open.zip` |

---

# WAVE 1

Start both Account 1 and Account 2 simultaneously.

---

### Wave 1 — Account 1: Open-Ended Qwen + Extra Pairings

Paste this code block into a single Python cell in Account 1:

```python
import os
import sys
import time
import base64
import subprocess
from IPython.display import HTML, Javascript, display

print("=" * 60)
print("WAVE 1 - ACCOUNT 1: QWEN2-VL OPEN-ENDED + EXTRA PAIRINGS")
print("=" * 60)

# STEP 1: FIX CUDA CONFLICT & INSTALL DEPENDENCIES
print("\n>>> Step 1/5: Fixing CUDA conflict and installing dependencies...")
subprocess.run("pip uninstall -y torchaudio", shell=True)
subprocess.run(
    "pip install -q -U transformers accelerate datasets pillow torchvision bitsandbytes",
    shell=True, check=True
)

# STEP 2: CLONE OR UPDATE REPOSITORY (EXTENSION BRANCH)
print("\n>>> Step 2/5: Synchronizing GitHub repository (extension branch)...")
repo_dir = "/kaggle/working/reliance-utility"
if not os.path.exists(repo_dir):
    subprocess.run(
        f"git clone https://github.com/Stealth768/reliance-utility.git {repo_dir}",
        shell=True, check=True
    )
else:
    print("Repository exists on disk. Resetting and pulling latest...")
    subprocess.run(f"cd {repo_dir} && git reset --hard HEAD", shell=True, check=True)

subprocess.run(
    f"cd {repo_dir} && git fetch --all --tags && git checkout extension && git pull origin extension",
    shell=True, check=True
)

# Verify GPU
import torch
from transformers import AutoProcessor
assert torch.cuda.is_available(), "GPU accelerator not detected! Enable GPU in Kaggle settings."
print(f"CUDA GPU Verified: {torch.cuda.get_device_name(0)}")

# STEP 3: PRE-FLIGHT CHECK
print("\n>>> Step 3/5: Running pre-flight check...")
check_cmd = f"cd {repo_dir} && python -u scripts/run_vlm_open.py --model Qwen/Qwen2-VL-2B-Instruct --check"
subprocess.run(check_cmd, shell=True, check=True)

# STEP 4: PRODUCTION RUNS
print("\n>>> Step 4/5: Running production tasks...")
out_open = "/kaggle/working/results/raw/vlm_open"
out_pair = "/kaggle/working/results/raw/vlm_pairings"
os.makedirs(out_open, exist_ok=True)
os.makedirs(out_pair, exist_ok=True)

prod_tasks = [
    # 1. Qwen2-VL Open-Ended (N=500, 3 corruptions)
    (
        "Open-Ended: Qwen2-VL-2B-Instruct (N=500, 3 corruptions)",
        f"cd {repo_dir} && python -u scripts/run_vlm_open.py "
        f"--model Qwen/Qwen2-VL-2B-Instruct "
        f"--data_ids results/derived/main_ids.json "
        f"--n 500 --seed 0 --shard_size 25 "
        f"--out {out_open}"
    ),
]

# 2. Extra swapped pairings (seeds 11, 12, 13) for all 3 models
for pair_seed in [11, 12, 13]:
    prod_tasks.append((
        f"Pairing Seed {pair_seed}: Qwen2-VL-2B",
        f"cd {repo_dir} && python -u scripts/run_vlm_pilot.py "
        f"--model Qwen/Qwen2-VL-2B-Instruct --dataset aokvqa "
        f"--corruptions swapblend --pairing_seed {pair_seed} --seed 0 "
        f"--out {out_pair}"
    ))
    prod_tasks.append((
        f"Pairing Seed {pair_seed}: SmolVLM",
        f"cd {repo_dir} && python -u scripts/run_vlm_pilot.py "
        f"--model HuggingFaceTB/SmolVLM-Instruct --dataset aokvqa "
        f"--assistant_prefix answer_colon "
        f"--corruptions swapblend --pairing_seed {pair_seed} --seed 0 "
        f"--out {out_pair}"
    ))
    prod_tasks.append((
        f"Pairing Seed {pair_seed}: LLaVA-1.5-7B",
        f"cd {repo_dir} && python -u scripts/run_vlm_pilot.py "
        f"--model llava-hf/llava-1.5-7b-hf --dataset aokvqa "
        f"--assistant_prefix answer_colon "
        f"--corruptions swapblend --pairing_seed {pair_seed} --seed 0 "
        f"--out {out_pair}"
    ))

total_tasks = len(prod_tasks)
t_start = time.time()
for idx, (title, cmd) in enumerate(prod_tasks, 1):
    print(f"\n[{idx}/{total_tasks}] Starting {title}...", flush=True)
    t_task = time.time()
    subprocess.run(cmd, shell=True, check=True)
    print(f"[{idx}/{total_tasks}] Completed {title} in {(time.time() - t_task)/60:.1f} min", flush=True)

print(f"\nAll tasks finished in {(time.time() - t_start)/60:.1f} min!")

# STEP 5: ZIP OUTPUT ARCHIVE & AUTO-DOWNLOAD
print("\n>>> Step 5/5: Packaging shards into zip and triggering download...")
zip_name = "account1_wave1.zip"
zip_path = f"/kaggle/working/{zip_name}"

subprocess.run(
    f"cd /kaggle/working && zip -r -q {zip_name} results/raw/vlm_open results/raw/vlm_pairings",
    shell=True, check=True
)
file_size_mb = os.path.getsize(zip_path) / (1024 * 1024)
print(f"SUCCESS! Archive created: {zip_name} ({file_size_mb:.2f} MB)")

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
    <h3 style="color: #274916; margin: 0 0 8px 0;">🎉 Download Triggered Automatically!</h3>
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

### Wave 1 — Account 2: AI2D Multiple-Choice Suite

Paste this code block into a single Python cell in Account 2:

```python
import os
import sys
import time
import base64
import subprocess
from IPython.display import HTML, Javascript, display

print("=" * 60)
print("WAVE 1 - ACCOUNT 2: AI2D MULTIPLE-CHOICE (ALL MODELS)")
print("=" * 60)

# STEP 1: FIX CUDA CONFLICT & INSTALL DEPENDENCIES
print("\n>>> Step 1/5: Fixing CUDA conflict and installing dependencies...")
subprocess.run("pip uninstall -y torchaudio", shell=True)
subprocess.run(
    "pip install -q -U transformers accelerate datasets pillow torchvision bitsandbytes",
    shell=True, check=True
)

# STEP 2: CLONE OR UPDATE REPOSITORY (EXTENSION BRANCH)
print("\n>>> Step 2/5: Synchronizing GitHub repository (extension branch)...")
repo_dir = "/kaggle/working/reliance-utility"
if not os.path.exists(repo_dir):
    subprocess.run(
        f"git clone https://github.com/Stealth768/reliance-utility.git {repo_dir}",
        shell=True, check=True
    )
else:
    print("Repository exists on disk. Resetting and pulling latest...")
    subprocess.run(f"cd {repo_dir} && git reset --hard HEAD", shell=True, check=True)

subprocess.run(
    f"cd {repo_dir} && git fetch --all --tags && git checkout extension && git pull origin extension",
    shell=True, check=True
)

# Verify GPU
import torch
from transformers import AutoProcessor
assert torch.cuda.is_available(), "GPU accelerator not detected! Enable GPU in Kaggle settings."
print(f"CUDA GPU Verified: {torch.cuda.get_device_name(0)}")

# STEP 3: PRE-FLIGHT CHECKS
print("\n>>> Step 3/5: Running pre-flight checks on AI2D...")
check_commands = [
    f"cd {repo_dir} && python -u scripts/run_vlm_pilot.py --model Qwen/Qwen2-VL-2B-Instruct --dataset ai2d --check",
    f"cd {repo_dir} && python -u scripts/run_vlm_pilot.py --model HuggingFaceTB/SmolVLM-Instruct --dataset ai2d --assistant_prefix answer_colon --check",
    f"cd {repo_dir} && python -u scripts/run_vlm_pilot.py --model llava-hf/llava-1.5-7b-hf --dataset ai2d --assistant_prefix answer_colon --check"
]
for cmd in check_commands:
    print(f"Executing check: {cmd}", flush=True)
    subprocess.run(cmd, shell=True, check=True)

# STEP 4: PRODUCTION RUNS
print("\n>>> Step 4/5: Running production AI2D tasks...")
out_ai2d = "/kaggle/working/results/raw/vlm_ai2d"
os.makedirs(out_ai2d, exist_ok=True)

prod_tasks = [
    # 1. Qwen2-VL AI2D Main (N=500, seed 0)
    (
        "AI2D Main: Qwen2-VL-2B-Instruct (N=500)",
        f"cd {repo_dir} && python -u scripts/run_vlm_pilot.py "
        f"--model Qwen/Qwen2-VL-2B-Instruct "
        f"--dataset ai2d "
        f"--data_ids results/derived/ai2d_main_ids.json "
        f"--n 500 --seed 0 --shard_size 25 "
        f"--out {out_ai2d}"
    ),
    # 2. Qwen2-VL AI2D Replication (N=400, seed 1)
    (
        "AI2D Replication: Qwen2-VL-2B-Instruct (N=400)",
        f"cd {repo_dir} && python -u scripts/run_vlm_pilot.py "
        f"--model Qwen/Qwen2-VL-2B-Instruct "
        f"--dataset ai2d "
        f"--data_ids results/derived/ai2d_rep_ids.json "
        f"--n 400 --seed 1 --shard_size 25 "
        f"--out {out_ai2d}"
    ),
    # 3. SmolVLM AI2D Main (N=500, seed 0)
    (
        "AI2D Main: SmolVLM-Instruct (N=500)",
        f"cd {repo_dir} && python -u scripts/run_vlm_pilot.py "
        f"--model HuggingFaceTB/SmolVLM-Instruct "
        f"--dataset ai2d "
        f"--assistant_prefix answer_colon "
        f"--data_ids results/derived/ai2d_main_ids.json "
        f"--n 500 --seed 0 --shard_size 25 "
        f"--out {out_ai2d}"
    ),
    # 4. LLaVA-1.5-7B AI2D Main (N=500, seed 0)
    (
        "AI2D Main: LLaVA-1.5-7B-hf (N=500)",
        f"cd {repo_dir} && python -u scripts/run_vlm_pilot.py "
        f"--model llava-hf/llava-1.5-7b-hf "
        f"--dataset ai2d "
        f"--assistant_prefix answer_colon "
        f"--data_ids results/derived/ai2d_main_ids.json "
        f"--n 500 --seed 0 --shard_size 25 "
        f"--out {out_ai2d}"
    ),
]

total_tasks = len(prod_tasks)
t_start = time.time()
for idx, (title, cmd) in enumerate(prod_tasks, 1):
    print(f"\n[{idx}/{total_tasks}] Starting {title}...", flush=True)
    t_task = time.time()
    subprocess.run(cmd, shell=True, check=True)
    print(f"[{idx}/{total_tasks}] Completed {title} in {(time.time() - t_task)/60:.1f} min", flush=True)

print(f"\nAll AI2D tasks finished in {(time.time() - t_start)/60:.1f} min!")

# STEP 5: ZIP OUTPUT ARCHIVE & AUTO-DOWNLOAD
print("\n>>> Step 5/5: Packaging shards into zip and triggering download...")
zip_name = "account2_wave1_ai2d.zip"
zip_path = f"/kaggle/working/{zip_name}"

subprocess.run(
    f"cd /kaggle/working && zip -r -q {zip_name} results/raw/vlm_ai2d",
    shell=True, check=True
)
file_size_mb = os.path.getsize(zip_path) / (1024 * 1024)
print(f"SUCCESS! Archive created: {zip_name} ({file_size_mb:.2f} MB)")

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
    <h3 style="color: #274916; margin: 0 0 8px 0;">🎉 Download Triggered Automatically!</h3>
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

# WAVE 2

Run after downloading and verifying Wave 1 outputs.

---

### Wave 2 — Account 1: SmolVLM Open-Ended

```python
import os
import sys
import time
import base64
import subprocess
from IPython.display import HTML, Javascript, display

print("=" * 60)
print("WAVE 2 - ACCOUNT 1: SMOLVLM OPEN-ENDED (GAUSSIAN + SWAPBLEND)")
print("=" * 60)

# STEP 1: FIX CUDA CONFLICT & INSTALL DEPENDENCIES
print("\n>>> Step 1/5: Fixing CUDA conflict and installing dependencies...")
subprocess.run("pip uninstall -y torchaudio", shell=True)
subprocess.run(
    "pip install -q -U transformers accelerate datasets pillow torchvision bitsandbytes",
    shell=True, check=True
)

# STEP 2: CLONE OR UPDATE REPOSITORY (EXTENSION BRANCH)
print("\n>>> Step 2/5: Synchronizing GitHub repository (extension branch)...")
repo_dir = "/kaggle/working/reliance-utility"
if not os.path.exists(repo_dir):
    subprocess.run(
        f"git clone https://github.com/Stealth768/reliance-utility.git {repo_dir}",
        shell=True, check=True
    )
else:
    print("Repository exists on disk. Resetting and pulling latest...")
    subprocess.run(f"cd {repo_dir} && git reset --hard HEAD", shell=True, check=True)

subprocess.run(
    f"cd {repo_dir} && git fetch --all --tags && git checkout extension && git pull origin extension",
    shell=True, check=True
)

# Verify GPU
import torch
from transformers import AutoProcessor
assert torch.cuda.is_available(), "GPU accelerator not detected! Enable GPU in Kaggle settings."
print(f"CUDA GPU Verified: {torch.cuda.get_device_name(0)}")

# STEP 3: PRE-FLIGHT CHECK
print("\n>>> Step 3/5: Running pre-flight check...")
check_cmd = f"cd {repo_dir} && python -u scripts/run_vlm_open.py --model HuggingFaceTB/SmolVLM-Instruct --assistant_prefix answer_colon --check"
subprocess.run(check_cmd, shell=True, check=True)

# STEP 4: PRODUCTION RUN
print("\n>>> Step 4/5: Running production open-ended SmolVLM...")
out_open = "/kaggle/working/results/raw/vlm_open"
os.makedirs(out_open, exist_ok=True)

# 2 corruptions (gaussian,swapblend) to fit within 12h session ceiling (~9.6h runtime)
prod_cmd = (
    f"cd {repo_dir} && python -u scripts/run_vlm_open.py "
    f"--model HuggingFaceTB/SmolVLM-Instruct "
    f"--assistant_prefix answer_colon "
    f"--corruptions gaussian,swapblend "
    f"--data_ids results/derived/main_ids.json "
    f"--n 500 --seed 0 --shard_size 25 "
    f"--out {out_open}"
)

t_start = time.time()
subprocess.run(prod_cmd, shell=True, check=True)
print(f"\nSmolVLM Open-Ended finished in {(time.time() - t_start)/60:.1f} min!")

# STEP 5: ZIP OUTPUT ARCHIVE & AUTO-DOWNLOAD
print("\n>>> Step 5/5: Packaging shards into zip and triggering download...")
zip_name = "account1_wave2_smolvlm_open.zip"
zip_path = f"/kaggle/working/{zip_name}"

subprocess.run(
    f"cd /kaggle/working && zip -r -q {zip_name} results/raw/vlm_open",
    shell=True, check=True
)
file_size_mb = os.path.getsize(zip_path) / (1024 * 1024)
print(f"SUCCESS! Archive created: {zip_name} ({file_size_mb:.2f} MB)")

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
    <h3 style="color: #274916; margin: 0 0 8px 0;">🎉 Download Triggered Automatically!</h3>
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

### Wave 2 — Account 2: LLaVA-1.5-7B Open-Ended

```python
import os
import sys
import time
import base64
import subprocess
from IPython.display import HTML, Javascript, display

print("=" * 60)
print("WAVE 2 - ACCOUNT 2: LLAVA-1.5-7B OPEN-ENDED (GAUSSIAN + SWAPBLEND)")
print("=" * 60)

# STEP 1: FIX CUDA CONFLICT & INSTALL DEPENDENCIES
print("\n>>> Step 1/5: Fixing CUDA conflict and installing dependencies...")
subprocess.run("pip uninstall -y torchaudio", shell=True)
subprocess.run(
    "pip install -q -U transformers accelerate datasets pillow torchvision bitsandbytes",
    shell=True, check=True
)

# STEP 2: CLONE OR UPDATE REPOSITORY (EXTENSION BRANCH)
print("\n>>> Step 2/5: Synchronizing GitHub repository (extension branch)...")
repo_dir = "/kaggle/working/reliance-utility"
if not os.path.exists(repo_dir):
    subprocess.run(
        f"git clone https://github.com/Stealth768/reliance-utility.git {repo_dir}",
        shell=True, check=True
    )
else:
    print("Repository exists on disk. Resetting and pulling latest...")
    subprocess.run(f"cd {repo_dir} && git reset --hard HEAD", shell=True, check=True)

subprocess.run(
    f"cd {repo_dir} && git fetch --all --tags && git checkout extension && git pull origin extension",
    shell=True, check=True
)

# Verify GPU
import torch
from transformers import AutoProcessor
assert torch.cuda.is_available(), "GPU accelerator not detected! Enable GPU in Kaggle settings."
print(f"CUDA GPU Verified: {torch.cuda.get_device_name(0)}")

# STEP 3: PRE-FLIGHT CHECK
print("\n>>> Step 3/5: Running pre-flight check...")
check_cmd = f"cd {repo_dir} && python -u scripts/run_vlm_open.py --model llava-hf/llava-1.5-7b-hf --assistant_prefix answer_colon --check"
subprocess.run(check_cmd, shell=True, check=True)

# STEP 4: PRODUCTION RUN
print("\n>>> Step 4/5: Running production open-ended LLaVA-1.5-7B...")
out_open = "/kaggle/working/results/raw/vlm_open"
os.makedirs(out_open, exist_ok=True)

# 2 corruptions (gaussian,swapblend) to fit within 12h session ceiling (~9.6h runtime)
prod_cmd = (
    f"cd {repo_dir} && python -u scripts/run_vlm_open.py "
    f"--model llava-hf/llava-1.5-7b-hf "
    f"--assistant_prefix answer_colon "
    f"--corruptions gaussian,swapblend "
    f"--data_ids results/derived/main_ids.json "
    f"--n 500 --seed 0 --shard_size 25 "
    f"--out {out_open}"
)

t_start = time.time()
subprocess.run(prod_cmd, shell=True, check=True)
print(f"\nLLaVA-1.5-7B Open-Ended finished in {(time.time() - t_start)/60:.1f} min!")

# STEP 5: ZIP OUTPUT ARCHIVE & AUTO-DOWNLOAD
print("\n>>> Step 5/5: Packaging shards into zip and triggering download...")
zip_name = "account2_wave2_llava_open.zip"
zip_path = f"/kaggle/working/{zip_name}"

subprocess.run(
    f"cd /kaggle/working && zip -r -q {zip_name} results/raw/vlm_open",
    shell=True, check=True
)
file_size_mb = os.path.getsize(zip_path) / (1024 * 1024)
print(f"SUCCESS! Archive created: {zip_name} ({file_size_mb:.2f} MB)")

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
    <h3 style="color: #274916; margin: 0 0 8px 0;">🎉 Download Triggered Automatically!</h3>
    <p style="margin: 0 0 10px 0;">If your browser popup blocker prevented the automatic download, click below:</p>
    <a href="data:application/zip;base64,{b64_data}" download="{zip_name}" 
       style="display: inline-block; background: #1890ff; color: white; padding: 8px 16px; border-radius: 4px; text-decoration: none; font-weight: bold;">
       Click to Download {zip_name} ({file_size_mb:.2f} MB)
    </a>
</div>
"""
display(HTML(html_code))
```
