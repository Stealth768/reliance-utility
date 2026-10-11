# Reliance-Utility Misalignment in Multimodal AI

This repository investigates **reliance-utility misalignment in multimodal AI**: whether multimodal foundation models continue relying on an increasingly degraded modality (e.g. corrupted visual sensor stream) even after that modality ceases to provide positive utility relative to unimodal null baselines. 

Using Vision-Language Models (VLMs) on Visual Question Answering (A-OKVQA) as our primary empirical benchmark alongside a controlled synthetic toy-fusion benchmark and a modality-dropout case study, we:
1. Formalize information-theoretic reliance $R(S)$ via Jensen-Shannon divergence ($\jsd$) from unimodal null baselines.
2. Trace multimodal utility $U(S) = \text{acc}(S) - \text{acc}_{\text{null}}$ across calibrated corruption severity sweeps.
3. Locate empirical utility crossing points $S^* = \min \{ S \mid U(S) \le 0 \}$.
4. Quantify over-trust via the normalized Misalignment Index ($\mi = \frac{1}{1-S^*} \int_{S^*}^1 \hat{R}(S) \, dS$).

---

## Repository Map

```text
reliance-utility/
├── CITATION.cff              # Citation metadata (TODO: author info)
├── DEVIATIONS.md             # Formal audit log of all design decisions and pre-registered hypotheses
├── FREEZE.md                 # Closed experiment specification (no new models/features)
├── README.md                 # Project overview and quickstart guide
├── RUNBOOK.md                # Multi-account Kaggle GPU orchestration manual
├── SCOPE.md                  # Project boundaries and tier definitions
├── requirements-gpu.txt      # Pinned requirements for GPU evaluation
├── figures/                  # Vector PDF plots generated for the paper
├── notebooks/
│   └── kaggle_main.md        # Kaggle cell-by-cell execution instructions
├── paper/                    # Paper draft sources
│   ├── main.tex              # Top-level article document
│   ├── macros.tex            # Shared LaTeX macros & checknum validator
│   ├── results_notes.md      # Template for recording execution metadata
│   ├── spec.md               # Mathematical specification
│   └── sections/             # Section drafts (abstract, intro, method, synthetic, results, dropout, limitations)
├── results/
│   ├── derived/              # Summaries, bootstrap CIs, and probe results
│   │   ├── main_ids.json     # N=500 disjoint validation question IDs
│   │   ├── synthetic_summary_v04.csv
│   │   └── bev_probe.json    # BEV driving feasibility probe output
│   └── raw/                  # Immutable write-once evaluation outputs
│       ├── synthetic/        # Frozen synthetic validation outputs
│       └── vlm_pilot/        # Frozen pilot study outputs
├── scripts/
│   ├── analyze_fusion.py     # Hierarchical analysis of modality dropout study
│   ├── analyze_vlm_all.py    # Multi-model cross-seed VLM benchmark analysis
│   ├── bev_feasibility_check.py # Hardware and library feasibility probe
│   ├── check_numbers.py      # Automated validator comparing paper numbers to CSVs
│   ├── check_run_complete.py # Coverage validator identifying missing shards
│   ├── hash_results.py       # SHA-256 integrity verifier for raw results
│   ├── make_all.py           # One-command regenerator for all derived tables/plots
│   ├── make_synthetic_figs.py# Synthetic publication figures generator
│   ├── merge_shards.py       # Safe multi-account shard merger
│   ├── pin_requirements.py   # Lockfile generator for environment freeze
│   └── run_vlm_pilot.py      # Primary VLM benchmark runner (GPU)
├── src/
│   ├── corruptions/          # Image degradation operators (Gaussian, patch, swapblend, extra nulls)
│   ├── fusion/               # Late/gated fusion model, datasets, training loops
│   ├── metrics/              # FROZEN metrics: JSD, utility, reliance, S*, MI, bootstrap CI
│   ├── io_utils.py           # Atomic write-once file utilities
│   ├── seed.py               # Deterministic RNG seeding
│   └── vlm/                  # Multi-model adapters (Qwen2-VL, SmolVLM, LLaVA-1.5), scoring, data loader
├── tables/                   # Generated LaTeX booktabs tables
└── tests/                    # 80+ unit and integration tests (full CPU suite)
```

---

## How to Run Everything

### 1. Quick CPU Test Suite
The entire test suite runs on CPU without requiring PyTorch, CUDA, or model checkpoints:

```bash
# Run all tests (all 85+ tests should pass)
python -m pytest -q
```

### 2. Running Benchmark Jobs on Kaggle
Refer to [`RUNBOOK.md`](RUNBOOK.md) for the exact session-by-session commands.
In brief:
1. Open a Kaggle notebook with GPU accelerator (T4 x2 or P100).
2. Clone repo and checkout the frozen tag:
   ```bash
   git checkout v0.9-runner
   ```
3. Run pre-flight check mode:
   ```bash
   python scripts/run_vlm_pilot.py --model Qwen/Qwen2-VL-2B-Instruct --check
   ```
4. Run full benchmark ($N=500$ across 3 corruptions):
   ```bash
   python scripts/run_vlm_pilot.py \
       --model Qwen/Qwen2-VL-2B-Instruct \
       --data_ids results/derived/main_ids.json \
       --n 500 --grid 9 \
       --corruptions gaussian,patch,swapblend \
       --seed 0 --shard_size 25 \
       --out /kaggle/working/results/raw/vlm_main
   ```
5. Zip and download output artifacts immediately after each model completes.

### 3. Local Assembly, Merge & Analysis
Once results are downloaded to your local machine:

```bash
# 1. Merge shards from multiple sessions/accounts (refuses to overwrite)
python scripts/merge_shards.py --src /path/to/downloaded_zips/ --out results/raw/vlm_main

# 2. Check complete shard coverage
python scripts/check_run_complete.py --dir results/raw/vlm_main --model Qwen/Qwen2-VL-2B-Instruct --corruption gaussian --seed 0

# 3. Regenerate all derived summary tables and vector PDF figures
python scripts/make_all.py

# 4. Verify hard-coded numbers in paper/*.tex against generated CSV tables
python scripts/check_numbers.py
```

---

## License
TODO: Specify license (e.g., MIT or Apache-2.0).
