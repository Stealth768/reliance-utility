# Dataset Licenses and Verification Record

This document records the verified dataset IDs, official revisions, field names, and licensing terms for all visual question answering benchmarks evaluated in the Reliance-Utility research project. All terms have been verified online via official Hugging Face repositories and author documentation.

---

## 1. Primary Benchmark: A-OKVQA

- **Repository**: `HuggingFaceM4/A-OKVQA`
- **Official Revision**: `d1b0efa3a436e9101dfbde3752db7607da696c35`
- **Source Paper**: Schwenk et al., *"A-OKVQA: A Benchmark for Visual Question Answering using World Knowledge"*, ECCV 2022 (arXiv:2206.01718).
- **License**: 
  - Annotations and dataset code: **Apache License 2.0** (Allen Institute for AI).
  - Images: Sourced from MS-COCO, subject to original Flickr Creative Commons licenses.
- **Access / Restrictions**: Open public access. No registration, gated access, or user authentication required.
- **Verified Fields**:
  - `question_id`: `str` / `int` (unique question identifier)
  - `question`: `str` (question text)
  - `choices`: `list[str]` (multiple-choice option strings, length 4)
  - `correct_choice_idx`: `int` (0-based integer index in $[0, 3]$)
  - `image`: `PIL.Image` / `dict` containing image bytes
  - `direct_answers`: `list[str]` (ground-truth free-form human answers, used for open-ended VQA soft accuracy)

---

## 2. Second Multiple-Choice Benchmark: AI2D (AI2 Diagrams)

- **Repository**: `lmms-lab-encoder/ai2d` (mirrored from `lmms-lab/ai2d`)
- **Official Source**: Allen Institute for AI (Ai2), Kembhavi et al., *"A Diagram Is Worth A Dozen Images"*, ECCV 2016.
- **License**: **Creative Commons Attribution-ShareAlike (CC BY-SA 4.0)**.
- **Access / Restrictions**: Open public access. No registration, gated access, or non-commercial-only restriction.
- **Verified Fields**:
  - `question`: `str` (diagram comprehension question text)
  - `options`: `list[str]` (multiple-choice candidate options; exactly 4 choices per question across all 3,088 test items)
  - `answer`: `str` (0-based integer index formatted as string `"0"`, `"1"`, `"2"`, `"3"`)
  - `image`: `dict` with image `bytes` (PIL-decodable diagram image)
- **Sample Partitions**:
  - Main Sample: $N=500$ validation questions deterministically sampled from the test split with seed 0.
  - Replication Sample: $N=400$ disjoint validation questions deterministically sampled with seed 1.
  - Total pool: 3,088 four-option diagram questions available.

---

## 3. Alternative Examined: ScienceQA (Inspected, Not Selected)

- **Repository**: `derek-thomas/ScienceQA`
- **Source Paper**: Lu et al., *"Learn to Explain: Multimodal Reasoning via Thought Chains for Science Question Answering"*, NeurIPS 2022.
- **License**: **CC BY-NC-SA 4.0** (Creative Commons Attribution-NonCommercial-ShareAlike 4.0).
- **Inspection Findings**:
  - Only ~750 questions per split have both an image and exactly 4 choices (many questions have 2, 3, or 5 choices).
  - Substantial portion of multimodal questions in ScienceQA can be answered by pure language models without the image (visual context is often supplemental rather than strictly necessary), posing a risk of violating the diagnostic gate requirement that text-only accuracy be at least 10 percentage points below clean multimodal accuracy.
  - License is strictly Non-Commercial (NC).
- **Selection Decision**: AI2D was chosen over ScienceQA due to its open CC BY-SA license, 100% four-option consistency across 3,088 items, and strict visual grounding where diagrams are mandatory to answer.
