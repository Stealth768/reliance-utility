# Kaggle Notebook Instructions: Reliance-Utility Extension Phase

This document contains executable notebook scripts for the Extension Phase of the reliance-utility research paper across Kaggle GPU environments (GPU T4 x2 or P100 with Internet enabled).

All documentation and terminology strictly adhere to **retained visual reliance** under multimodal degradation (without assuming subjective over-reliance).

---

## Experimental Protocol & Job Allocation

1. **Wave 0 (Pre-Flight Interactive Session)**:
   - A single interactive session on one GPU account to validate model weights, tokenization, assistant prefixes, and pre-flight gates across all model $\times$ dataset configurations.
   - Evaluates mean letter probability mass, clean multimodal accuracy, text-only unimodal accuracy, empty-generation fraction, and per-question inference latency.
   - Outputs a verified benchmark and runtime projection table before launching long-running production sessions.

2. **Wave 1 (Production Batches — 2 Parallel Accounts)**:
   - **Account 1** (`account1_wave1.zip`, ~7.0h): Qwen2-VL Open-Ended (N=500, 3 corruptions) + Extra Swapped Pairings for Qwen, SmolVLM, LLaVA across derangement seeds $\{11, 12, 13\}$ (swapblend only). Total: 240 shards.
   - **Account 2** (`account2_wave1_ai2d.zip`, ~7.7h): AI2D Multiple-Choice full suite: Qwen Main (N=500), Qwen Rep (N=400), SmolVLM Main (N=500), LLaVA Main (N=500). Total: 228 shards.

3. **Wave 2 (Production Batches — 2 Parallel Accounts)**:
   - **Account 1** (`account1_wave2_smolvlm_open.zip`, ~9.6h): SmolVLM Open-Ended on A-OKVQA (N=500, 2 corruptions: `gaussian,swapblend`). Total: 40 shards.
   - **Account 2** (`account2_wave2_llava_open.zip`, ~9.6h): LLaVA-1.5-7B Open-Ended on A-OKVQA (N=500, 2 corruptions: `gaussian,swapblend`). Total: 40 shards.

4. **Pairing Stability Protocol**:
   - Evaluates the range and 95% bootstrap confidence interval of retained terminal reliance $\hat{R}(1)$ across derangement seeds $\{0, 11, 12, 13\}$. No variance test is performed.

5. **Execution & Download Rule**:
   - Waves 1 and 2 **must run with "Save & Run All" (Commit mode)**.
   - Every individual job executes its pre-flight `--check` first and stops immediately if `--check` exits non-zero.
   - Output archives are written directly into `/kaggle/working/` and downloaded from the committed version's **Output** tab.

---

## Dataset Provenance & Licenses

- **AI2D (AI2 Diagrams)**:
  - **Source**: `lmms-lab-encoder/ai2d` via Hugging Face (`default/test/0.parquet`, `default/test/1.parquet`).
  - **Original Reference**: Kembhavi et al., *A Diagram is Worth A Dozen Images*, ECCV 2016. Allen Institute for Artificial Intelligence.
  - **License**: CC BY-SA 4.0 (Creative Commons Attribution-ShareAlike 4.0 International).
  - **Split Specification**: 4-choice multiple-choice diagrams; N=500 main validation split (seed 0, `results/derived/ai2d_main_ids.json`), N=400 disjoint replication split (seed 1, `results/derived/ai2d_rep_ids.json`).

- **A-OKVQA (Augmented OK-VQA)**:
  - **Source**: `HuggingFaceM4/A-OKVQA` (revision: `d1b0efa3a436e9101dfbde3752db7607da696c35`).
  - **Original Reference**: Schwenk et al., *A-OKVQA: A Benchmark for Visual Question Answering using World Knowledge*, ECCV 2022.
  - **License**: CC BY 4.0 (Creative Commons Attribution 4.0 International).
  - **Split Specification**: Validation split; N=500 main validation IDs (`results/derived/main_ids.json`), N=445 disjoint replication IDs (`results/derived/rep_ids.json`).

---

# WAVE 0: Interactive Pre-Flight Diagnostics & Runtime Calibration

Run this cell interactively in a single GPU notebook before starting Wave 1. It executes `--check` on all 6 combinations, gathers exact latency numbers, and prints the runtime projection table.

```python
import os
import sys
import time
import subprocess
import re

print("=" * 70)
print("WAVE 0: INTERACTIVE PRE-FLIGHT DIAGNOSTICS & RUNTIME CALIBRATION")
print("=" * 70)

# 1. Environment & Dependencies
print("\n>>> Setting up environment...")
subprocess.run("pip uninstall -y torchaudio", shell=True)
subprocess.run(
    "pip install -q -U transformers accelerate datasets pillow torchvision bitsandbytes",
    shell=True, check=True
)

repo_dir = "/kaggle/working/reliance-utility"
if not os.path.exists(repo_dir):
    subprocess.run(
        f"git clone https://github.com/Stealth768/reliance-utility.git {repo_dir}",
        shell=True, check=True
    )
else:
    subprocess.run(f"cd {repo_dir} && git reset --hard HEAD", shell=True, check=True)

subprocess.run(
    f"cd {repo_dir} && git fetch --all --tags && git checkout extension && git pull origin extension",
    shell=True, check=True
)

import torch
assert torch.cuda.is_available(), "GPU accelerator required!"
gpu_name = torch.cuda.get_device_name(0)
print(f"CUDA Accelerator Verified: {gpu_name}")

# 2. Check Suite Definitions
checks = [
    {
        "model": "Qwen2-VL-2B",
        "task": "AI2D MC",
        "cmd": f"cd {repo_dir} && python -u scripts/run_vlm_pilot.py --model Qwen/Qwen2-VL-2B-Instruct --dataset ai2d --check",
        "type": "mc"
    },
    {
        "model": "SmolVLM",
        "task": "AI2D MC",
        "cmd": f"cd {repo_dir} && python -u scripts/run_vlm_pilot.py --model HuggingFaceTB/SmolVLM-Instruct --dataset ai2d --assistant_prefix answer_colon --check",
        "type": "mc"
    },
    {
        "model": "LLaVA-1.5-7B",
        "task": "AI2D MC",
        "cmd": f"cd {repo_dir} && python -u scripts/run_vlm_pilot.py --model llava-hf/llava-1.5-7b-hf --dataset ai2d --assistant_prefix answer_colon --check",
        "type": "mc"
    },
    {
        "model": "Qwen2-VL-2B",
        "task": "Open-Ended",
        "cmd": f"cd {repo_dir} && python -u scripts/run_vlm_open.py --model Qwen/Qwen2-VL-2B-Instruct --check",
        "type": "open"
    },
    {
        "model": "SmolVLM",
        "task": "Open-Ended",
        "cmd": f"cd {repo_dir} && python -u scripts/run_vlm_open.py --model HuggingFaceTB/SmolVLM-Instruct --assistant_prefix answer_colon --check",
        "type": "open"
    },
    {
        "model": "LLaVA-1.5-7B",
        "task": "Open-Ended",
        "cmd": f"cd {repo_dir} && python -u scripts/run_vlm_open.py --model llava-hf/llava-1.5-7b-hf --assistant_prefix answer_colon --check",
        "type": "open"
    },
]

results = []
print("\n>>> Executing Pre-Flight Checks (10 items per combination)...")
for c in checks:
    print(f"\n--- Checking {c['model']} [{c['task']}] ---", flush=True)
    t0 = time.time()
    res = subprocess.run(c["cmd"], shell=True, capture_output=True, text=True)
    elapsed = time.time() - t0
    stdout = res.stdout
    print(stdout)
    if res.returncode != 0:
        print(f"[FAIL] Check failed for {c['model']} {c['task']} with exit code {res.returncode}")
        print(res.stderr)
        sys.exit(res.returncode)

    # Parse metrics
    mass_match = re.search(r"Mean Letter Probability Mass\s*:\s*([0-9.]+)", stdout)
    clean_acc_match = re.search(r"Clean Multimodal Accuracy\s*:\s*([0-9.]+)%", stdout)
    text_acc_match = re.search(r"Text-Only Accuracy\s*:\s*([0-9.]+)%", stdout)
    empty_match = re.search(r"Empty answers\s*:\s*[0-9]+/10\s*\(([0-9.]+)%\)", stdout)
    time_q_match = re.search(r"Time per question \(check\)\s*:\s*([0-9.]+)\s*s", stdout)

    time_per_q = float(time_q_match.group(1)) if time_q_match else (elapsed / 10.0)
    mass = f"{float(mass_match.group(1)):.4f}" if mass_match else "N/A"
    clean_acc = f"{float(clean_acc_match.group(1)):.1f}%" if clean_acc_match else "N/A"
    text_acc = f"{float(text_acc_match.group(1)):.1f}%" if text_acc_match else "N/A"
    empty_frac = f"{float(empty_match.group(1)):.1f}%" if empty_match else "0.0%"

    results.append({
        "model": c["model"],
        "task": c["task"],
        "mass": mass,
        "clean_acc": clean_acc,
        "text_acc": text_acc,
        "empty": empty_frac,
        "time_q": time_per_q,
    })

# 3. Print Pre-Flight Diagnostic Summary Table
print("\n" + "=" * 90)
print("PRE-FLIGHT DIAGNOSTIC SUMMARY")
print("=" * 90)
header = f"{'Model':<15} | {'Task':<12} | {'Letter Mass':<11} | {'Clean Acc':<9} | {'Text Acc':<9} | {'Empty Frac':<10} | {'Time/Q (s)':<10} | {'Status'}"
print(header)
print("-" * 90)
for r in results:
    row = f"{r['model']:<15} | {r['task']:<12} | {r['mass']:<11} | {r['clean_acc']:<9} | {r['text_acc']:<9} | {r['empty']:<10} | {r['time_q']:<10.2f} | PASS"
    print(row)
print("-" * 90)

# 4. Projected Runtime Table for Waves 1 & 2
# MC job estimation: N questions * (3 corruptions * 9 grid + 2 nulls) / (check_evals * scaling)
# Empirically on T4: MC questions take ~time_per_q / 2 per forward pass (check did 2 forward passes per item)
# Open-ended: check does ~3 forward passes per item (clean, black, removed), full run does 29 forward passes per item.
print("\n" + "=" * 90)
print("PROJECTED RUNTIMES PER WAVE 1 & WAVE 2 PRODUCTION JOBS")
print("=" * 90)
job_header = f"{'Wave':<6} | {'Account':<9} | {'Job Description':<40} | {'Items / Corruptions':<20} | {'Projected Hours'}"
print(job_header)
print("-" * 90)

lookup = {(r['model'], r['task']): r['time_q'] for r in results}

# Estimates in hours
qwen_open_h = (lookup[('Qwen2-VL-2B', 'Open-Ended')] / 3.0 * 29 * 500) / 3600.0
smol_open_h = (lookup[('SmolVLM', 'Open-Ended')] / 3.0 * 20 * 500) / 3600.0  # 2 corruptions
llava_open_h = (lookup[('LLaVA-1.5-7B', 'Open-Ended')] / 3.0 * 20 * 500) / 3600.0  # 2 corruptions

qwen_ai2d_main_h = (lookup[('Qwen2-VL-2B', 'AI2D MC')] * 14 * 500) / 3600.0
qwen_ai2d_rep_h = (lookup[('Qwen2-VL-2B', 'AI2D MC')] * 14 * 400) / 3600.0
smol_ai2d_main_h = (lookup[('SmolVLM', 'AI2D MC')] * 14 * 500) / 3600.0
llava_ai2d_main_h = (lookup[('LLaVA-1.5-7B', 'AI2D MC')] * 14 * 500) / 3600.0

# Pairings: 3 seeds * 500 items * 1 corruption (9 grid + 2 nulls) = 11 passes per item
pairings_h = 3 * ((lookup[('Qwen2-VL-2B', 'AI2D MC')] + lookup[('SmolVLM', 'AI2D MC')] + lookup[('LLaVA-1.5-7B', 'AI2D MC')]) * 5.5 * 500) / 3600.0

print(f"{'Wave 1':<6} | {'Account 1':<9} | {'Qwen2-VL Open-Ended (3 corruptions)':<40} | {'N=500, 3 corr':<20} | {qwen_open_h:.2f} h")
print(f"{'Wave 1':<6} | {'Account 1':<9} | {'Extra Swapped Pairings (3 models x 3 seeds)':<40} | {'N=500 x 9 jobs':<20} | {pairings_h:.2f} h")
print(f"{'Wave 1':<6} | {'Account 1':<9} | {'--> Account 1 Wave 1 Total Projected Time':<40} | {'':<20} | {(qwen_open_h + pairings_h):.2f} h")
print("-" * 90)
print(f"{'Wave 1':<6} | {'Account 2':<9} | {'Qwen2-VL AI2D Main':<40} | {'N=500, 3 corr':<20} | {qwen_ai2d_main_h:.2f} h")
print(f"{'Wave 1':<6} | {'Account 2':<9} | {'Qwen2-VL AI2D Replication':<40} | {'N=400, 3 corr':<20} | {qwen_ai2d_rep_h:.2f} h")
print(f"{'Wave 1':<6} | {'Account 2':<9} | {'SmolVLM AI2D Main':<40} | {'N=500, 3 corr':<20} | {smol_ai2d_main_h:.2f} h")
print(f"{'Wave 1':<6} | {'Account 2':<9} | {'LLaVA-1.5-7B AI2D Main':<40} | {'N=500, 3 corr':<20} | {llava_ai2d_main_h:.2f} h")
print(f"{'Wave 1':<6} | {'Account 2':<9} | {'--> Account 2 Wave 1 Total Projected Time':<40} | {'':<20} | {(qwen_ai2d_main_h + qwen_ai2d_rep_h + smol_ai2d_main_h + llava_ai2d_main_h):.2f} h")
print("-" * 90)
print(f"{'Wave 2':<6} | {'Account 1':<9} | {'SmolVLM Open-Ended (swapblend+gaussian)':<40} | {'N=500, 2 corr':<20} | {smol_open_h:.2f} h")
print(f"{'Wave 2':<6} | {'Account 2':<9} | {'LLaVA-1.5-7B Open-Ended (swapblend+gaussian)':<40} | {'N=500, 2 corr':<20} | {llava_open_h:.2f} h")
print("=" * 90)
```

---

# WAVE 1

Start both Account 1 and Account 2 in parallel using **"Save & Run All" (Commit mode)**.

---

### Wave 1 — Account 1: Open-Ended Qwen + Extra Swapped Pairings

Paste this cell into Account 1:

```python
import os
import sys
import time
import subprocess

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

import torch
assert torch.cuda.is_available(), "GPU accelerator not detected!"
print(f"CUDA GPU Verified: {torch.cuda.get_device_name(0)}")

# STEP 3 & 4: PRODUCTION RUNS WITH MANDATORY PRE-CHECK GUARD
out_open = "/kaggle/working/results/raw/vlm_open"
out_pair = "/kaggle/working/results/raw/vlm_pairings"
os.makedirs(out_open, exist_ok=True)
os.makedirs(out_pair, exist_ok=True)

# Helper function to execute pre-check before running production
def run_guarded_job(job_name, check_command, prod_command):
    print(f"\n" + "=" * 60)
    print(f"STARTING GUARDED JOB: {job_name}")
    print("=" * 60)
    print(f">> Executing pre-flight check: {check_command}", flush=True)
    check_res = subprocess.run(check_command, shell=True)
    if check_res.returncode != 0:
        print(f"\n[FATAL ERROR] Pre-flight check failed with code {check_res.returncode}!")
        print(f"Stopping job '{job_name}' immediately. Refusing to run full production.")
        sys.exit(check_res.returncode)
    print(f"[CHECK PASSED] Launching production run: {prod_command}", flush=True)
    t0 = time.time()
    subprocess.run(prod_command, shell=True, check=True)
    print(f"[JOB COMPLETED] Finished {job_name} in {(time.time() - t0)/60:.1f} min", flush=True)

t_start = time.time()

# 1. Qwen Open-Ended Run
run_guarded_job(
    job_name="Qwen2-VL Open-Ended (N=500, 3 corruptions)",
    check_command=f"cd {repo_dir} && python -u scripts/run_vlm_open.py --model Qwen/Qwen2-VL-2B-Instruct --check",
    prod_command=(
        f"cd {repo_dir} && python -u scripts/run_vlm_open.py "
        f"--model Qwen/Qwen2-VL-2B-Instruct "
        f"--data_ids results/derived/main_ids.json "
        f"--n 500 --seed 0 --shard_size 25 "
        f"--out {out_open}"
    )
)

# 2. Extra Swapped Pairings for Qwen2-VL, SmolVLM, LLaVA-1.5 across seeds 11, 12, 13
# Run model MC check once per model before executing pairing seeds
models_pairing = [
    {
        "name": "Qwen2-VL-2B",
        "check": f"cd {repo_dir} && python -u scripts/run_vlm_pilot.py --model Qwen/Qwen2-VL-2B-Instruct --dataset aokvqa --check",
        "base_cmd": f"cd {repo_dir} && python -u scripts/run_vlm_pilot.py --model Qwen/Qwen2-VL-2B-Instruct --dataset aokvqa --corruptions swapblend --seed 0 --out {out_pair}"
    },
    {
        "name": "SmolVLM",
        "check": f"cd {repo_dir} && python -u scripts/run_vlm_pilot.py --model HuggingFaceTB/SmolVLM-Instruct --dataset aokvqa --assistant_prefix answer_colon --check",
        "base_cmd": f"cd {repo_dir} && python -u scripts/run_vlm_pilot.py --model HuggingFaceTB/SmolVLM-Instruct --dataset aokvqa --assistant_prefix answer_colon --corruptions swapblend --seed 0 --out {out_pair}"
    },
    {
        "name": "LLaVA-1.5-7B",
        "check": f"cd {repo_dir} && python -u scripts/run_vlm_pilot.py --model llava-hf/llava-1.5-7b-hf --dataset aokvqa --assistant_prefix answer_colon --check",
        "base_cmd": f"cd {repo_dir} && python -u scripts/run_vlm_pilot.py --model llava-hf/llava-1.5-7b-hf --dataset aokvqa --assistant_prefix answer_colon --corruptions swapblend --seed 0 --out {out_pair}"
    },
]

for m in models_pairing:
    print(f"\n>> Pre-flight verification for {m['name']} pairings...")
    c_res = subprocess.run(m["check"], shell=True)
    if c_res.returncode != 0:
        print(f"[FATAL ERROR] Check failed for {m['name']}. Aborting!")
        sys.exit(c_res.returncode)
    for pair_seed in [11, 12, 13]:
        p_cmd = f"{m['base_cmd']} --pairing_seed {pair_seed}"
        print(f">> Running {m['name']} Swapped Pairing Seed {pair_seed}...", flush=True)
        subprocess.run(p_cmd, shell=True, check=True)

print(f"\nAll Wave 1 Account 1 tasks completed in {(time.time() - t_start)/60:.1f} min!")

# STEP 5: ZIP OUTPUT ARCHIVE
print("\n>>> Step 5/5: Packaging shards into zip...")
zip_name = "account1_wave1.zip"
zip_path = f"/kaggle/working/{zip_name}"

subprocess.run(
    f"cd /kaggle/working && zip -r -q {zip_name} results/raw/vlm_open results/raw/vlm_pairings",
    shell=True, check=True
)
file_size_mb = os.path.getsize(zip_path) / (1024 * 1024)
print("=" * 70)
print(f"SUCCESS! Output archive saved to: {zip_path} ({file_size_mb:.2f} MB)")
print("=" * 70)
print("NOTE: Download this archive directly from the 'Output' tab of your committed Kaggle notebook version.")
```

---

### Wave 1 — Account 2: AI2D Multiple-Choice Suite

Paste this cell into Account 2:

```python
import os
import sys
import time
import subprocess

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

import torch
assert torch.cuda.is_available(), "GPU accelerator not detected!"
print(f"CUDA GPU Verified: {torch.cuda.get_device_name(0)}")

# STEP 3 & 4: PRODUCTION RUNS WITH MANDATORY PRE-CHECK GUARD
out_ai2d = "/kaggle/working/results/raw/vlm_ai2d"
os.makedirs(out_ai2d, exist_ok=True)

def run_guarded_job(job_name, check_command, prod_commands):
    print(f"\n" + "=" * 60)
    print(f"STARTING GUARDED JOB: {job_name}")
    print("=" * 60)
    print(f">> Executing pre-flight check: {check_command}", flush=True)
    check_res = subprocess.run(check_command, shell=True)
    if check_res.returncode != 0:
        print(f"\n[FATAL ERROR] Pre-flight check failed with code {check_res.returncode}!")
        print(f"Stopping job '{job_name}' immediately. Refusing to run full production.")
        sys.exit(check_res.returncode)
    for cmd in prod_commands:
        print(f"[CHECK PASSED] Launching production run: {cmd}", flush=True)
        t0 = time.time()
        subprocess.run(cmd, shell=True, check=True)
        print(f"[SUBTASK COMPLETED] in {(time.time() - t0)/60:.1f} min", flush=True)

t_start = time.time()

# 1. Qwen2-VL AI2D Main + Replication
run_guarded_job(
    job_name="Qwen2-VL AI2D Suite (Main N=500 + Rep N=400)",
    check_command=f"cd {repo_dir} && python -u scripts/run_vlm_pilot.py --model Qwen/Qwen2-VL-2B-Instruct --dataset ai2d --check",
    prod_commands=[
        f"cd {repo_dir} && python -u scripts/run_vlm_pilot.py --model Qwen/Qwen2-VL-2B-Instruct --dataset ai2d --data_ids results/derived/ai2d_main_ids.json --n 500 --seed 0 --shard_size 25 --out {out_ai2d}",
        f"cd {repo_dir} && python -u scripts/run_vlm_pilot.py --model Qwen/Qwen2-VL-2B-Instruct --dataset ai2d --data_ids results/derived/ai2d_rep_ids.json --n 400 --seed 1 --shard_size 25 --out {out_ai2d}"
    ]
)

# 2. SmolVLM AI2D Main
run_guarded_job(
    job_name="SmolVLM AI2D Main (N=500)",
    check_command=f"cd {repo_dir} && python -u scripts/run_vlm_pilot.py --model HuggingFaceTB/SmolVLM-Instruct --dataset ai2d --assistant_prefix answer_colon --check",
    prod_commands=[
        f"cd {repo_dir} && python -u scripts/run_vlm_pilot.py --model HuggingFaceTB/SmolVLM-Instruct --dataset ai2d --assistant_prefix answer_colon --data_ids results/derived/ai2d_main_ids.json --n 500 --seed 0 --shard_size 25 --out {out_ai2d}"
    ]
)

# 3. LLaVA-1.5-7B AI2D Main
run_guarded_job(
    job_name="LLaVA-1.5-7B AI2D Main (N=500)",
    check_command=f"cd {repo_dir} && python -u scripts/run_vlm_pilot.py --model llava-hf/llava-1.5-7b-hf --dataset ai2d --assistant_prefix answer_colon --check",
    prod_commands=[
        f"cd {repo_dir} && python -u scripts/run_vlm_pilot.py --model llava-hf/llava-1.5-7b-hf --dataset ai2d --assistant_prefix answer_colon --data_ids results/derived/ai2d_main_ids.json --n 500 --seed 0 --shard_size 25 --out {out_ai2d}"
    ]
)

print(f"\nAll Wave 1 Account 2 tasks completed in {(time.time() - t_start)/60:.1f} min!")

# STEP 5: ZIP OUTPUT ARCHIVE
print("\n>>> Step 5/5: Packaging shards into zip...")
zip_name = "account2_wave1_ai2d.zip"
zip_path = f"/kaggle/working/{zip_name}"

subprocess.run(
    f"cd /kaggle/working && zip -r -q {zip_name} results/raw/vlm_ai2d",
    shell=True, check=True
)
file_size_mb = os.path.getsize(zip_path) / (1024 * 1024)
print("=" * 70)
print(f"SUCCESS! Output archive saved to: {zip_path} ({file_size_mb:.2f} MB)")
print("=" * 70)
print("NOTE: Download this archive directly from the 'Output' tab of your committed Kaggle notebook version.")
```

---

# WAVE 2 (FINAL LOAD-BALANCED PRODUCTION)

Run both notebooks in parallel. Both accounts include **checkpoint zips** after every completed job so shards are preserved even in case of an early exit.

---

### Wave 2 — Account 1: SmolVLM Pairing 13 + SmolVLM Open-Ended (~11.0h)

**Upload**: 
otebooks/wave2_account1_final.ipynb OR paste this cell:

`python
import os
import sys
import time
import subprocess

print("=" * 70)
print("WAVE 2 FINAL - ACCOUNT 1: SMOLVLM PAIRING 13 + SMOLVLM OPEN-ENDED")
print("=" * 70)

# STEP 1: FIX CUDA CONFLICT & INSTALL DEPENDENCIES
print("\n>>> Step 1: Fixing CUDA conflict and installing dependencies...")
subprocess.run("pip uninstall -y torchaudio", shell=True)
subprocess.run(
    "pip install -q -U transformers accelerate datasets pillow torchvision bitsandbytes",
    shell=True, check=True
)

# STEP 2: CLONE OR UPDATE REPOSITORY (EXTENSION BRANCH)
print("\n>>> Step 2: Synchronizing GitHub repository (extension branch)...")
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

import torch
assert torch.cuda.is_available(), "GPU accelerator not detected!"
print(f"CUDA GPU Verified: {torch.cuda.get_device_name(0)}")

out_pair = "/kaggle/working/results/raw/vlm_pairings"
out_open = "/kaggle/working/results/raw/vlm_open"
os.makedirs(out_pair, exist_ok=True)
os.makedirs(out_open, exist_ok=True)

# Helper function to execute pre-check before running production
def run_guarded_job(job_name, check_cmd, prod_cmd):
    print("\n" + "=" * 70)
    print(f"STARTING GUARDED JOB: {job_name}")
    print("=" * 70)
    print(f">> Executing pre-flight check: {check_cmd}", flush=True)
    check_res = subprocess.run(check_cmd, shell=True)
    if check_res.returncode != 0:
        print(f"\n[FATAL ERROR] Pre-flight check failed with code {check_res.returncode}!")
        print(f"Stopping job '{job_name}' immediately. Refusing to run full production.")
        sys.exit(check_res.returncode)
    print(f"[CHECK PASSED] Launching production run: {prod_cmd}", flush=True)
    t0 = time.time()
    subprocess.run(prod_cmd, shell=True, check=True)
    elapsed_min = (time.time() - t0) / 60
    print(f"[JOB COMPLETED] Finished {job_name} in {elapsed_min:.1f} min", flush=True)

t_global_start = time.time()

# ----------------------------------------------------------------------
# JOB 1: SmolVLM A-OKVQA Pairing Seed 13 (~1.5 hours)
# ----------------------------------------------------------------------
run_guarded_job(
    job_name="SmolVLM A-OKVQA Pairing Seed 13",
    check_cmd=f"cd {repo_dir} && python -u scripts/run_vlm_pilot.py --model HuggingFaceTB/SmolVLM-Instruct --dataset aokvqa --assistant_prefix answer_colon --check",
    prod_cmd=(
        f"cd {repo_dir} && python -u scripts/run_vlm_pilot.py "
        f"--model HuggingFaceTB/SmolVLM-Instruct "
        f"--dataset aokvqa "
        f"--assistant_prefix answer_colon "
        f"--corruptions swapblend "
        f"--pairing_seed 13 "
        f"--seed 0 --shard_size 25 "
        f"--out {out_pair}"
    )
)

# Intermediate packaging for Job 1
print("\n>>> Creating intermediate checkpoint zip for pairings...")
subprocess.run(f"cd /kaggle/working && zip -r -q checkpoint_acc1_pair13.zip results/raw/vlm_pairings", shell=True)
print("Checkpoint zip created: checkpoint_acc1_pair13.zip")

# ----------------------------------------------------------------------
# JOB 2: SmolVLM Open-Ended (gaussian, swapblend) (~9.5 hours)
# ----------------------------------------------------------------------
run_guarded_job(
    job_name="SmolVLM Open-Ended (gaussian + swapblend)",
    check_cmd=f"cd {repo_dir} && python -u scripts/run_vlm_open.py --model HuggingFaceTB/SmolVLM-Instruct --assistant_prefix answer_colon --check",
    prod_cmd=(
        f"cd {repo_dir} && python -u scripts/run_vlm_open.py "
        f"--model HuggingFaceTB/SmolVLM-Instruct "
        f"--assistant_prefix answer_colon "
        f"--corruptions gaussian,swapblend "
        f"--data_ids results/derived/main_ids.json "
        f"--n 500 --seed 0 --shard_size 25 "
        f"--out {out_open}"
    )
)

# ----------------------------------------------------------------------
# FINAL PACKAGING
# ----------------------------------------------------------------------
print("\n>>> Step 5: Packaging all completed shards into final zip...")
final_zip = "account1_wave2_final.zip"
subprocess.run(
    f"cd /kaggle/working && zip -r -q {final_zip} results/raw/vlm_pairings results/raw/vlm_open",
    shell=True, check=True
)

zip_path = f"/kaggle/working/{final_zip}"
size_mb = os.path.getsize(zip_path) / (1024 * 1024)
total_hours = (time.time() - t_global_start) / 3600

print("=" * 70)
print(f"SUCCESS! Account 1 completed in {total_hours:.2f} hours.")
print(f"Final archive: {zip_path} ({size_mb:.2f} MB)")
print("=" * 70)
print("NOTE: Download this archive directly from the 'Output' tab of your committed Kaggle notebook version.")

`

---

### Wave 2 — Account 2: LLaVA AI2D + Pairing 13 + Open-Ended (~11.5h)

**Upload**: 
otebooks/wave2_account2_final.ipynb OR paste this cell:

`python
import os
import sys
import time
import subprocess

print("=" * 70)
print("WAVE 2 FINAL - ACCOUNT 2: LLAVA AI2D + PAIRING 13 + OPEN-ENDED")
print("=" * 70)

# STEP 1: FIX CUDA CONFLICT & INSTALL DEPENDENCIES
print("\n>>> Step 1: Fixing CUDA conflict and installing dependencies...")
subprocess.run("pip uninstall -y torchaudio", shell=True)
subprocess.run(
    "pip install -q -U transformers accelerate datasets pillow torchvision bitsandbytes",
    shell=True, check=True
)

# STEP 2: CLONE OR UPDATE REPOSITORY (EXTENSION BRANCH)
print("\n>>> Step 2: Synchronizing GitHub repository (extension branch)...")
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

import torch
assert torch.cuda.is_available(), "GPU accelerator not detected!"
print(f"CUDA GPU Verified: {torch.cuda.get_device_name(0)}")

out_ai2d = "/kaggle/working/results/raw/vlm_ai2d"
out_pair = "/kaggle/working/results/raw/vlm_pairings"
out_open = "/kaggle/working/results/raw/vlm_open"
os.makedirs(out_ai2d, exist_ok=True)
os.makedirs(out_pair, exist_ok=True)
os.makedirs(out_open, exist_ok=True)

def run_guarded_job(job_name, check_cmd, prod_cmd):
    print("\n" + "=" * 70)
    print(f"STARTING GUARDED JOB: {job_name}")
    print("=" * 70)
    print(f">> Executing pre-flight check: {check_cmd}", flush=True)
    check_res = subprocess.run(check_cmd, shell=True)
    if check_res.returncode != 0:
        print(f"\n[FATAL ERROR] Pre-flight check failed with code {check_res.returncode}!")
        print(f"Stopping job '{job_name}' immediately. Refusing to run full production.")
        sys.exit(check_res.returncode)
    print(f"[CHECK PASSED] Launching production run: {prod_cmd}", flush=True)
    t0 = time.time()
    subprocess.run(prod_cmd, shell=True, check=True)
    elapsed_min = (time.time() - t0) / 60
    print(f"[JOB COMPLETED] Finished {job_name} in {elapsed_min:.1f} min", flush=True)

t_global_start = time.time()

# ----------------------------------------------------------------------
# JOB 1: LLaVA-1.5-7B AI2D Remaining (patch + swapblend) (~1.6 hours)
# Note: --corruptions patch,swapblend runs only the remaining corruptions
# ----------------------------------------------------------------------
run_guarded_job(
    job_name="LLaVA-1.5-7B AI2D (patch + swapblend)",
    check_cmd=f"cd {repo_dir} && python -u scripts/run_vlm_pilot.py --model llava-hf/llava-1.5-7b-hf --dataset ai2d --assistant_prefix answer_colon --check",
    prod_cmd=(
        f"cd {repo_dir} && python -u scripts/run_vlm_pilot.py "
        f"--model llava-hf/llava-1.5-7b-hf "
        f"--dataset ai2d "
        f"--assistant_prefix answer_colon "
        f"--corruptions patch,swapblend "
        f"--data_ids results/derived/ai2d_main_ids.json "
        f"--n 500 --seed 0 --shard_size 25 "
        f"--out {out_ai2d}"
    )
)

print("\n>>> Checkpoint zip: packaging completed AI2D shards...")
subprocess.run(f"cd /kaggle/working && zip -r -q checkpoint_acc2_ai2d.zip results/raw/vlm_ai2d", shell=True)
print("Checkpoint zip created: checkpoint_acc2_ai2d.zip")

# ----------------------------------------------------------------------
# JOB 2: LLaVA-1.5-7B A-OKVQA Pairing Seed 13 (~1.4 hours)
# ----------------------------------------------------------------------
run_guarded_job(
    job_name="LLaVA-1.5-7B A-OKVQA Pairing Seed 13",
    check_cmd=f"cd {repo_dir} && python -u scripts/run_vlm_pilot.py --model llava-hf/llava-1.5-7b-hf --dataset aokvqa --assistant_prefix answer_colon --check",
    prod_cmd=(
        f"cd {repo_dir} && python -u scripts/run_vlm_pilot.py "
        f"--model llava-hf/llava-1.5-7b-hf "
        f"--dataset aokvqa "
        f"--assistant_prefix answer_colon "
        f"--corruptions swapblend "
        f"--pairing_seed 13 "
        f"--seed 0 --shard_size 25 "
        f"--out {out_pair}"
    )
)

print("\n>>> Checkpoint zip: packaging completed pairing shards...")
subprocess.run(f"cd /kaggle/working && zip -r -q checkpoint_acc2_pair13.zip results/raw/vlm_pairings", shell=True)
print("Checkpoint zip created: checkpoint_acc2_pair13.zip")

# ----------------------------------------------------------------------
# JOB 3: LLaVA-1.5-7B Open-Ended (gaussian, swapblend) (~8.5 hours)
# ----------------------------------------------------------------------
run_guarded_job(
    job_name="LLaVA-1.5-7B Open-Ended (gaussian + swapblend)",
    check_cmd=f"cd {repo_dir} && python -u scripts/run_vlm_open.py --model llava-hf/llava-1.5-7b-hf --assistant_prefix answer_colon --check",
    prod_cmd=(
        f"cd {repo_dir} && python -u scripts/run_vlm_open.py "
        f"--model llava-hf/llava-1.5-7b-hf "
        f"--assistant_prefix answer_colon "
        f"--corruptions gaussian,swapblend "
        f"--data_ids results/derived/main_ids.json "
        f"--n 500 --seed 0 --shard_size 25 "
        f"--out {out_open}"
    )
)

# ----------------------------------------------------------------------
# FINAL PACKAGING
# ----------------------------------------------------------------------
print("\n>>> Step 5: Packaging all completed shards into final zip...")
final_zip = "account2_wave2_final.zip"
subprocess.run(
    f"cd /kaggle/working && zip -r -q {final_zip} results/raw/vlm_ai2d results/raw/vlm_pairings results/raw/vlm_open",
    shell=True, check=True
)

zip_path = f"/kaggle/working/{final_zip}"
size_mb = os.path.getsize(zip_path) / (1024 * 1024)
total_hours = (time.time() - t_global_start) / 3600

print("=" * 70)
print(f"SUCCESS! Account 2 completed in {total_hours:.2f} hours.")
print(f"Final archive: {zip_path} ({size_mb:.2f} MB)")
print("=" * 70)
print("NOTE: Download this archive directly from the 'Output' tab of your committed Kaggle notebook version.")

`
