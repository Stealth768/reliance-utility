# Frozen Experiment Specification

This document defines the closed, immutable scope of experiments for the reliance-utility study.
Any evaluation, model, benchmark, or variant not explicitly listed here is strictly **future work**.

---

## 1. Primary Benchmark: Vision-Language Models (VLM)
- **Dataset**: A-OKVQA ($N=500$ validation items, strictly disjoint from the $N=200$ pilot, `results/derived/main_ids.json`).
- **Models**:
  1. `Qwen/Qwen2-VL-2B-Instruct` (float16)
  2. `HuggingFaceTB/SmolVLM-Instruct` (float16; evaluated with assistant prefix if mass $\ge 0.90$)
  3. `llava-hf/llava-1.5-7b-hf` (4-bit NF4 via BitsAndBytesConfig; evaluated with assistant prefix if mass $\ge 0.90$)
- **Corruptions**:
  1. `gaussian` (Gaussian noise blend over 9 severity levels $S \in \{0.0, 0.125, \dots, 1.0\}$)
  2. `patch` (random square patch occlusion, 28x28 patches)
  3. `swapblend` (deceptive visual blend from fixed seeded derangement)
- **Evaluation Seeds**: Seeds `0`, `1`, `2` across all models and corruptions.
- **Null Baselines**:
  - Primary: `removed` (text-only prompt without image)
  - Secondary controls: `black` (all-zero image), `swapped` (image from derangement donor)
  - Sensitivity controls (optional via `--extra_nulls`): `mean_image` (dataset gray), `noise_image` (standard Gaussian noise)

---

## 2. Controlled Case Study: Modality Dropout Fusion
- **Architectures**: Late fusion network with learned quality-agnostic gate (sigmoid over modality B representation).
- **Datasets**:
  1. `synthetic`: Synthetic two-view classification (dim=64, 4 classes, class overlap=0.3).
  2. `fashionmnist`: Two-view proxy (modality A = left 14 cols, modality B = right 14 cols).
     *(Explicitly documented as a methodological two-view proxy, not a true multimodal benchmark).*
- **Dropout Grid**: $p \in \{0.0, 0.1, 0.25, 0.5\}$ applied to modality A during training.
- **Seeds**: 5 seeds ($0, 1, 2, 3, 4$) per condition.
- **Hypotheses**: Pre-declared H1 (MI non-increasing in $p$), H2 (clean accuracy drop $< 3\%$ at $p=0.25$), H3 (report failure without rescue).

---

## 3. Explicitly Excluded (Future Work)
- **InternVL2-2B**: Dropped due to persistent remote-code incompatibility with `transformers>=4.50` (`all_tied_weights_keys` missing).
- **Autonomous Driving (BEVFusion / nuScenes-mini)**: Feasibility probe verdict is `NO-GO`; driving evaluation is deferred to future work.
- **Open-ended Generation**: Multiple-choice 4-option formulation is frozen.
