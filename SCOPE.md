# Project Scope and Boundaries

This document defines the research scope, tiers of evaluation, and resource constraints for the project on **Reliance-Utility Misalignment in Multimodal AI**.

---

## 1. Core Evaluation: Vision-Language Models (VLMs)
- **Task**: Visual Question Answering (VQA) using VQAv2 and GQA benchmark subsets.
- **Models**: 4–5 open-weights Vision-Language Models (e.g., LLaVA family, Qwen-VL, InternVL variants).
- **Corruptions**: 2–3 systematic visual corruption types applied over calibrated severity grids $S \in [0, 1]$ (e.g., Gaussian noise, Gaussian blur, patch occlusion / pixel dropout).
- **Controls & Null Baselines**:
  - Black image
  - Mean dataset image
  - Pure noise image
  - Modality ablation (image tokens omitted / removed from prompt)

---

## 2. Validation: Synthetic Toy-Fusion Test
- Controlled synthetic benchmark with known data-generating distributions.
- Validates the theoretical behavior of reliance $R(S)$, utility $U(S)$, crossing point $S^*$, and misalignment index $\text{MI}$.
- Success criterion: Gated dynamic-reliance fusion achieves $\text{MI}(\text{gated}) < 0.2$, whereas fixed-weight naive fusion exhibits $\text{MI}(\text{fixed}) > 0.7$, with non-overlapping 95% bootstrap confidence intervals.

---

## 3. Second Case Study: Small Trained Fusion Model
- Compact multimodal classification/regression model trained from scratch.
- Comparison of architectures trained:
  - Without regularization (baseline multimodal fusion)
  - With modality dropout during training
- Measures how architectural gating and training-time regularization affect empirical reliance once a sensor/modality degrades.

---

## 4. Stretch Goal: Autonomous Driving (BEVFusion on nuScenes-mini)
- Pretrained BEVFusion checkpoint evaluated under sensor degradation (e.g., camera noise/fog or LiDAR dropout).
- Single corruption sweep on nuScenes-mini dataset.
- Clear computational limitation statement:
  > "Due to limited computational resources, the driving case study is restricted to nuScenes-mini and pretrained checkpoints, and we treat it as a proof of applicability rather than a full evaluation."
