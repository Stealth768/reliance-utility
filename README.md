# Reliance-Utility Misalignment in Multimodal AI

This repository investigates **reliance-utility misalignment in multimodal AI**: whether multimodal architectures continue relying on an increasingly degraded modality (e.g., a corrupted or noisy visual input) even after that modality ceases to provide positive utility relative to unimodal or null baselines. Using Vision-Language Models (VLMs) on Visual Question Answering (VQA) as our primary empirical case study alongside a controlled synthetic toy-fusion benchmark, we formalize information-theoretic reliance via Jensen-Shannon divergence, trace utility degradation across severity sweeps, locate the empirical utility crossing point, and quantify the persistent reliance regime through a normalized misalignment index.

## Directory Structure

```text
reliance-utility/
├── configs/            # Experiment configuration files (YAML/JSON)
├── figures/            # Generated plots and visualizations for the paper
├── notebooks/          # Exploratory analysis and prototype notebooks
├── paper/              # Paper draft, specifications, and LNCS LaTeX sources
│   └── spec.md         # Formal mathematical specification and protocol
├── results/
│   ├── raw/            # Write-once raw evaluation outputs and manifests
│   └── derived/        # Post-processed metrics, summary tables, and CIs
├── scripts/            # Automation, hashing, and manifest generation utilities
├── src/                # Core Python package codebase
│   ├── corruptions/    # Modality degradation operators and corruption grids
│   ├── metrics/        # Divergence, reliance, utility, and misalignment index
│   ├── models/         # Model wrappers, inference harnesses, and null baselines
│   ├── synthetic/      # Synthetic validation benchmarks and gated fusion models
│   ├── io_utils.py     # Atomic, write-once file I/O utilities
│   └── seed.py         # Multi-framework deterministic seed setup
├── tables/             # LaTeX and markdown tables for paper results
├── tests/              # Unit tests and regression test suite
├── DATA_LICENSES.md    # Licensing information for models and datasets
├── DEVIATIONS.md       # Audit log for experimental deviations
├── requirements.txt    # Project dependencies
└── SCOPE.md            # Experimental scope, boundaries, and compute limits
```

## Running Tests

All unit tests are written using `pytest` and can be executed without installing heavy dependencies:

```bash
# Run the entire test suite
pytest

# Run tests with verbose output
pytest -v

# Run metrics-specific tests
pytest tests/test_metrics.py
```
