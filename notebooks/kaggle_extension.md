# Kaggle Notebook Instructions: Reliance-Utility Extension Phase

Instructions and executable cells for executing Extension Phase experiments (AI2D Second MC Dataset, Extra Swapped Pairings, Open-Ended Free Generation, and Qwen2-VL-7B-Instruct 4-bit) across Kaggle GPU sessions.

---

## Multi-Account Job Matrix (Zero Shard Collisions)

| Job ID | Account / Session | Accelerator | Task / Dataset | Models | Seeds / Shards | Est. Runtime | Output Artifact |
|---|---|---|---|---|---|---|---|
| **EXT-1** | Account A | GPU T4 x2 / P100 | AI2D Multiple-Choice | Qwen2-VL-2B-Instruct | Main N=500 (seed 0) & Rep N=400 (seed 1) | ~4.5 hours | `ai2d_qwen2vl.zip` |
| **EXT-2** | Account B | GPU T4 x2 / P100 | AI2D Multiple-Choice | SmolVLM & LLaVA-1.5-7B | Main N=500 (seed 0) | ~5.5 hours | `ai2d_smol_llava.zip` |
| **EXT-3** | Account C | GPU T4 x2 / P100 | A-OKVQA Extra Pairings | Qwen2-VL, SmolVLM, LLaVA | Pairing seeds 11, 12, 13 (swapped only) | ~4.0 hours | `aokvqa_extra_pairings.zip` |
| **EXT-4** | Account D | GPU T4 x2 / P100 | Open-Ended Generation | Qwen2-VL-2B & SmolVLM | A-OKVQA Main N=500 (seed 0) | ~6.0 hours | `vlm_open_ended.zip` |
| **EXT-5** | Account E (Opt) | GPU T4 x2 / P100 | 7B Scaling Check | Qwen2-VL-7B (4-bit NF4) | A-OKVQA & AI2D checks + shards | ~5.0 hours | `qwen7b_4bit.zip` |

---

### Cell 1: Public Clone & Dependency Setup (No Tokens Required)

```bash
%%bash
git clone https://github.com/Stealth768/reliance-utility.git /kaggle/working/reliance-utility
cd /kaggle/working/reliance-utility
git fetch --all --tags
git checkout extension
pip uninstall -y torchaudio
pip install -q -U transformers accelerate datasets pillow torchvision bitsandbytes
```

---

### Cell 2: GPU Environment Verification

```python
import torch

assert torch.cuda.is_available(), "GPU is not active. Enable GPU T4 x2 or P100 in Kaggle Notebook settings."
print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
print(f"Device Count: {torch.cuda.device_count()}")
print(f"Memory Allocated: {torch.cuda.memory_allocated(0) / (1024**2):.1f} MB")
```

---

### Cell 3: Pre-Flight Check — AI2D Multiple Choice (Qwen2-VL-2B)

```bash
%%bash
cd /kaggle/working/reliance-utility
python scripts/run_vlm_pilot.py \
    --model Qwen/Qwen2-VL-2B-Instruct \
    --dataset ai2d \
    --check
```

---

### Cell 4: Pre-Flight Check — AI2D Multiple Choice (SmolVLM & LLaVA-1.5)

```bash
%%bash
cd /kaggle/working/reliance-utility
echo "=== Checking SmolVLM on AI2D ==="
python scripts/run_vlm_pilot.py \
    --model HuggingFaceTB/SmolVLM-Instruct \
    --dataset ai2d \
    --assistant_prefix answer_colon \
    --check

echo "=== Checking LLaVA-1.5 on AI2D ==="
python scripts/run_vlm_pilot.py \
    --model llava-hf/llava-1.5-7b-hf \
    --dataset ai2d \
    --assistant_prefix answer_colon \
    --check
```

---

### Cell 5: Pre-Flight Check — Open-Ended Generation (Qwen2-VL-2B)

```bash
%%bash
cd /kaggle/working/reliance-utility
python scripts/run_vlm_open.py \
    --model Qwen/Qwen2-VL-2B-Instruct \
    --check
```

---

### Cell 6: Pre-Flight Check — Optional Qwen2-VL-7B 4-bit NF4

```bash
%%bash
cd /kaggle/working/reliance-utility
python scripts/run_vlm_pilot.py \
    --model Qwen/Qwen2-VL-7B-Instruct \
    --dataset ai2d \
    --check
```

---

### Cell 7: Production Run — JOB EXT-1 (AI2D on Qwen2-VL-2B Main & Rep)

```bash
%%bash
cd /kaggle/working/reliance-utility
# 1. AI2D Main validation sample (N=500, seed 0)
python scripts/run_vlm_pilot.py \
    --model Qwen/Qwen2-VL-2B-Instruct \
    --dataset ai2d \
    --data_ids results/derived/ai2d_main_ids.json \
    --n 500 \
    --seed 0 \
    --out /kaggle/working/results/raw/vlm_ai2d

# 2. AI2D Disjoint replication sample (N=400, seed 1)
python scripts/run_vlm_pilot.py \
    --model Qwen/Qwen2-VL-2B-Instruct \
    --dataset ai2d \
    --data_ids results/derived/ai2d_rep_ids.json \
    --n 400 \
    --seed 1 \
    --out /kaggle/working/results/raw/vlm_ai2d
```

---

### Cell 8: Zip Shards — JOB EXT-1

```bash
%%bash
cd /kaggle/working
zip -r ai2d_qwen2vl.zip results/raw/vlm_ai2d/ai2d_Qwen2-VL-2B-Instruct* results/raw/vlm_ai2d/manifest_ai2d_Qwen2-VL-2B-Instruct*
ls -lh ai2d_qwen2vl.zip
```

---

### Cell 9: Base64 Browser Download Link — JOB EXT-1

```python
import base64
from IPython.display import HTML

def create_download_link(filename, title="Download Shard Archive"):
    with open(filename, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    return HTML(f'<a href="data:application/zip;base64,{b64}" download="{filename}" target="_blank"><strong>{title}: {filename}</strong></a>')

create_download_link("ai2d_qwen2vl.zip")
```

---

### Cell 10: Production Run — JOB EXT-2 (AI2D on SmolVLM & LLaVA-1.5)

```bash
%%bash
cd /kaggle/working/reliance-utility
# SmolVLM on AI2D Main
python scripts/run_vlm_pilot.py \
    --model HuggingFaceTB/SmolVLM-Instruct \
    --dataset ai2d \
    --assistant_prefix answer_colon \
    --data_ids results/derived/ai2d_main_ids.json \
    --n 500 \
    --seed 0 \
    --out /kaggle/working/results/raw/vlm_ai2d

# LLaVA-1.5-7B on AI2D Main
python scripts/run_vlm_pilot.py \
    --model llava-hf/llava-1.5-7b-hf \
    --dataset ai2d \
    --assistant_prefix answer_colon \
    --data_ids results/derived/ai2d_main_ids.json \
    --n 500 \
    --seed 0 \
    --out /kaggle/working/results/raw/vlm_ai2d
```

---

### Cell 11: Zip Shards & Base64 Download — JOB EXT-2

```bash
%%bash
cd /kaggle/working
zip -r ai2d_smol_llava.zip results/raw/vlm_ai2d/ai2d_SmolVLM* results/raw/vlm_ai2d/ai2d_llava* results/raw/vlm_ai2d/manifest_ai2d_SmolVLM* results/raw/vlm_ai2d/manifest_ai2d_llava*
ls -lh ai2d_smol_llava.zip
```

```python
import base64
from IPython.display import HTML

with open("ai2d_smol_llava.zip", "rb") as f:
    b64 = base64.b64encode(f.read()).decode()
HTML(f'<a href="data:application/zip;base64,{b64}" download="ai2d_smol_llava.zip" target="_blank"><strong>Download: ai2d_smol_llava.zip</strong></a>')
```

---

### Cell 12: Production Run — JOB EXT-3 (A-OKVQA Extra Pairings: Seeds 11, 12, 13)

```bash
%%bash
cd /kaggle/working/reliance-utility
for pair_seed in 11 12 13; do
    echo "Running Qwen2-VL with pairing seed $pair_seed"
    python scripts/run_vlm_pilot.py \
        --model Qwen/Qwen2-VL-2B-Instruct \
        --dataset aokvqa \
        --corruptions swapblend \
        --pairing_seed $pair_seed \
        --seed 0 \
        --out /kaggle/working/results/raw/vlm_pairings

    echo "Running SmolVLM with pairing seed $pair_seed"
    python scripts/run_vlm_pilot.py \
        --model HuggingFaceTB/SmolVLM-Instruct \
        --dataset aokvqa \
        --corruptions swapblend \
        --assistant_prefix answer_colon \
        --pairing_seed $pair_seed \
        --seed 0 \
        --out /kaggle/working/results/raw/vlm_pairings

    echo "Running LLaVA-1.5 with pairing seed $pair_seed"
    python scripts/run_vlm_pilot.py \
        --model llava-hf/llava-1.5-7b-hf \
        --dataset aokvqa \
        --corruptions swapblend \
        --assistant_prefix answer_colon \
        --pairing_seed $pair_seed \
        --seed 0 \
        --out /kaggle/working/results/raw/vlm_pairings
done
```

---

### Cell 13: Zip Shards & Base64 Download — JOB EXT-3

```bash
%%bash
cd /kaggle/working
zip -r aokvqa_extra_pairings.zip results/raw/vlm_pairings/*
ls -lh aokvqa_extra_pairings.zip
```

```python
import base64
from IPython.display import HTML

with open("aokvqa_extra_pairings.zip", "rb") as f:
    b64 = base64.b64encode(f.read()).decode()
HTML(f'<a href="data:application/zip;base64,{b64}" download="aokvqa_extra_pairings.zip" target="_blank"><strong>Download: aokvqa_extra_pairings.zip</strong></a>')
```

---

### Cell 14: Production Run — JOB EXT-4 (Open-Ended Free Generation)

```bash
%%bash
cd /kaggle/working/reliance-utility
# 1. Qwen2-VL-2B Open-Ended on A-OKVQA Main N=500
python scripts/run_vlm_open.py \
    --model Qwen/Qwen2-VL-2B-Instruct \
    --data_ids results/derived/main_ids.json \
    --n 500 \
    --seed 0 \
    --out /kaggle/working/results/raw/vlm_open

# 2. SmolVLM Open-Ended on A-OKVQA Main N=500
python scripts/run_vlm_open.py \
    --model HuggingFaceTB/SmolVLM-Instruct \
    --assistant_prefix answer_colon \
    --data_ids results/derived/main_ids.json \
    --n 500 \
    --seed 0 \
    --out /kaggle/working/results/raw/vlm_open
```

---

### Cell 15: Zip Shards & Base64 Download — JOB EXT-4

```bash
%%bash
cd /kaggle/working
zip -r vlm_open_ended.zip results/raw/vlm_open/*
ls -lh vlm_open_ended.zip
```

```python
import base64
from IPython.display import HTML

with open("vlm_open_ended.zip", "rb") as f:
    b64 = base64.b64encode(f.read()).decode()
HTML(f'<a href="data:application/zip;base64,{b64}" download="vlm_open_ended.zip" target="_blank"><strong>Download: vlm_open_ended.zip</strong></a>')
```
