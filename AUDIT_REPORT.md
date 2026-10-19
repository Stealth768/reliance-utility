# Comprehensive Read-Only Audit Report: `reliance-utility`

**Date of Audit:** October 19, 2026  
**Repository Root:** `c:\Users\steal\Desktop\Multimodal Research paper\Code\reliance-utility`  
**Execution Environment:** Windows (PowerShell), Python 3.13, PyTorch CPU in `.venv`  
**Audit Mode:** Strict Read-Only (no files modified under `results/raw`, `src`, `scripts`, `tests`, or `paper`)

---

## SECTION 1: Repository Integrity

### 1.1 Git Status & Current Branch
- **Current Branch:** `main`
- **Branch Tracking:** Up to date with `origin/main`
- **HEAD Commit:** `ac183972adcae0e08d4e5c3fe3c89c718d927edd`
- **HEAD Commit Date:** Mon Oct 12 21:27:42 2026 +0530
- **HEAD Commit Subject:** `fix(fusion): slice test split for Fashion-MNIST to match 392-dim model encoder`
- **Working Tree Status:** Clean with respect to tracked files; 6 untracked staging directories exist under `results/raw/`:
  - `results/raw/fusion_out/` (40 files: 20 models + 20 predictions)
  - `results/raw/llava_rep_out/` (72 files: 36 shards + 36 manifests)
  - `results/raw/llava_rep_out(0)/` (144 files: 72 shards + 72 manifests)
  - `results/raw/qwen_rep_out/` (72 files: 36 shards + 36 manifests)
  - `results/raw/smol_out/` (120 files: 60 shards + 60 manifests)
  - `results/raw/smolvlm_rep_out/` (72 files: 36 shards + 36 manifests)

### 1.2 Git Tags Audit
All local tags were compared against `origin` via `git ls-remote --tags origin`.

- **Total Local Tags:** 30
- **Total Remote Tags on Origin:** 30
- **Local-Only Tags:** None (0 local-only tags; all tags are pushed and synchronized with origin):
  - `v0.0-setup` (`4bad1fd`)
  - `v0.1-synthetic` (`170243e`)
  - `v0.2-synthetic` (`c725415`)
  - `v0.3-synthetic` (`eec6bb0`)
  - `v0.4-synthetic` (`bdb90cb`)
  - `v0.5-vlm-pilot-code` (`a799031`)
  - `v0.6-vlm-pilot-results` (`ee0cc16`)
  - `v0.7-main-code` (`26a42e9`)
  - `v0.7.1-main-code` (`adb7056`)
  - `v0.7.2-main-code` (`4c464c0`)
  - `v0.7.3-main-code` (`8f45a0f`)
  - `v0.8-llava-code` (`be74575`)
  - `v0.8-qwen-results` (`93e60aa`)
  - `v0.9-runner` (`64fe783`)
  - `v0.10-analysis` (`bd54fbc`)
  - `v0.11-dropout-code` (`5501687`)
  - `v0.12-bev-probe` (`6a32785`)
  - `v0.13-ready-to-run` (`674e2c8`)
  - `v0.13.1-ready` (`0565328`)
  - `v0.13.2` (`1038082`)
  - `v0.13.3` (`d13e583`)
  - `v0.13.4` (`0587cf5`)
  - `v0.14-llava-results` (`1553f62`)
  - `v0.14.1` (`082c59f`)
  - `v0.14.2` (`352917f`)
  - `v0.15-rep-ids` (`8004591`)
  - `v0.15.1-qwen-rep` (`007b72c`)
  - `v0.15.2-llava-rep` (`4cb0216`)
  - `v0.16-smol-results` (`eecf6e9`)
  - `v0.16.1-analysis` (`ec0bfed`)

### 1.3 Pytest Suite Execution
- **Command:** `.venv\Scripts\pytest -q`
- **Result:** **98 passed, 0 failed, 0 skipped, 1 warning** (duration: 25.30s)
- **Warning:** `tests/test_metrics.py:159: UserWarning: Bootstrap resampling encountered 82/200 (41.00%) resamples with no crossing / None result.` (expected diagnostic test for flakey crossing).

### 1.4 Raw Data Hash Verification
- **Manifest File:** `results/raw/MANIFEST.sha256`
- **Total Entries in Manifest:** 989 files
- **Files in Manifest Missing from Disk:** 0
- **Files on Disk with Mismatched Hash:** **0** (100% hash integrity for all 989 tracked raw files)
- **Files on Disk not in Manifest:** 520 files, comprising the 6 uncommitted staging directories:
  - `llava_rep_out(0)/` (144 files)
  - `smol_out/` (120 files)
  - `llava_rep_out/` (72 files)
  - `qwen_rep_out/` (72 files)
  - `smolvlm_rep_out/` (72 files)
  - `fusion_out/` (40 files)

---

## SECTION 2: Raw Data Inventory

### 2.1 Shard Counts per Directory, Model, Corruption, and Seed

| Directory | Model Name | Corruption | Seed | Actual Shards | Expected Shards | Gaps / Status |
|---|---|---|---|---|---|---|
| `vlm_main/` | `Qwen/Qwen2-VL-2B-Instruct` | gaussian | 0 (None in json) | 20 | 20 | Complete ($N=500$) |
| `vlm_main/` | `Qwen/Qwen2-VL-2B-Instruct` | patch | 0 (None in json) | 20 | 20 | Complete ($N=500$) |
| `vlm_main/` | `Qwen/Qwen2-VL-2B-Instruct` | swapblend | 0 (None in json) | 20 | 20 | Complete ($N=500$) |
| `vlm_llava/` | `llava-hf/llava-1.5-7b-hf` | gaussian | 0 | 20 | 20 | Complete ($N=500$) |
| `vlm_llava/` | `llava-hf/llava-1.5-7b-hf` | patch | 0 | 20 | 20 | Complete ($N=500$) |
| `vlm_llava/` | `llava-hf/llava-1.5-7b-hf` | swapblend | 0 | 20 | 20 | Complete ($N=500$) |
| `vlm_smol/` | `HuggingFaceTB/SmolVLM-Instruct` | gaussian | 0 | 20 | 20 | Complete ($N=500$) |
| `vlm_smol/` | `HuggingFaceTB/SmolVLM-Instruct` | patch | 0 | 20 | 20 | Complete ($N=500$) |
| `vlm_smol/` | `HuggingFaceTB/SmolVLM-Instruct` | swapblend | 0 | 20 | 20 | Complete ($N=500$) |
| `vlm_rep/` | `Qwen/Qwen2-VL-2B-Instruct` | gaussian | 1 | 18 | 18 | Complete ($N=445$) |
| `vlm_rep/` | `Qwen/Qwen2-VL-2B-Instruct` | swapblend | 1 | 18 | 18 | Complete ($N=445$) |
| `vlm_rep/` | `llava-hf/llava-1.5-7b-hf` | gaussian | 1 | 18 | 18 | Complete ($N=445$) |
| `vlm_rep/` | `llava-hf/llava-1.5-7b-hf` | swapblend | 1 | 18 | 18 | Complete ($N=445$) |
| `vlm_pilot/` | `Qwen/Qwen2-VL-2B-Instruct` | gaussian | 0 (None in json) | 8 | 8 | Complete ($N=200$) |
| `vlm_pilot/` | `Qwen/Qwen2-VL-2B-Instruct` | patch | 0 (None in json) | 8 | 8 | Complete ($N=200$) |
| `synthetic/` | Fixed, Gated, Partial | — | 0..4 | 30 | 30 | Complete (v0.1) |
| `synthetic_v02/`| Fixed, Gated, Partial | — | 0..4 | 45 | 45 | Complete (v0.2) |
| `synthetic_v03/`| Fixed, Gated, Partial, Oracle, Sharp | — | 0..4 | 150 | 150 | Complete (v0.3) |
| `fusion/` | None | — | — | 0 | 20 | Empty (raw data located in `fusion_out/`) |
| `fusion_out/` *(untracked)* | Fashion-MNIST LateFusion | grid | 0..4 | 20 | 20 | Complete ($p \in \{0, 0.1, 0.25, 0.5\} \times 5$ seeds) |
| `smolvlm_rep_out/` *(untracked)* | `HuggingFaceTB/SmolVLM-Instruct` | gaussian | 1 | 18 | 18 | Complete ($N=445$) |
| `smolvlm_rep_out/` *(untracked)* | `HuggingFaceTB/SmolVLM-Instruct` | swapblend | 1 | 18 | 18 | Complete ($N=445$) |

### 2.2 Manifest Audit per Model

| Evaluation Group | Model ID | git_commit | git_tag | Seed | Assistant Prefix | Question IDs File Hash (SHA-256) |
|---|---|---|---|---|---|---|
| **Qwen Main** | `Qwen/Qwen2-VL-2B-Instruct` | `not_a_git_repo` **[FLAGGED]** | `None` | 0 | `None` | `None` (legacy manifest schema) |
| **LLaVA Main** | `llava-hf/llava-1.5-7b-hf` | `0587cf5` | `v0.13.4` | 0 | `None` | `abaf601d2d2409df81054c5f054a0bfaba9458c303b67ed2953220ac94cb956a` |
| **SmolVLM Main** | `HuggingFaceTB/SmolVLM-Instruct` | `8004591` | `v0.15-rep-ids` | 0 | `answer_colon` | `abaf601d2d2409df81054c5f054a0bfaba9458c303b67ed2953220ac94cb956a` |
| **Qwen Rep** | `Qwen/Qwen2-VL-2B-Instruct` | `8004591` | `v0.15-rep-ids` | 1 | `None` | `a55c470423bd18fd8e5a9856b9fb220c2ae3627ff630afed9a923ec6c282ac32` |
| **LLaVA Rep** | `llava-hf/llava-1.5-7b-hf` | `8004591` | `v0.15-rep-ids` | 1 | `None` | `a55c470423bd18fd8e5a9856b9fb220c2ae3627ff630afed9a923ec6c282ac32` |
| **SmolVLM Rep** | `HuggingFaceTB/SmolVLM-Instruct` | `8004591` | `v0.15-rep-ids` | 1 | `answer_colon` | `a55c470423bd18fd8e5a9856b9fb220c2ae3627ff630afed9a923ec6c282ac32` |

> **Flagged Item:** The Qwen main evaluation manifests in `results/raw/vlm_main/` record `"git_commit": "not_a_git_repo"`. This is documented in `DEVIATIONS.md` entry 20: evaluation was launched in a Kaggle environment where the repository was copied as a flat script bundle rather than cloned via git. Code and adapter were verified identical to `v0.7-main-code` and provenance is locked via `results/raw/MANIFEST.sha256` under tag `v0.8-qwen-results`.

### 2.3 Disjointness Verification of Question ID Sets
Verified by direct inspection of `results/derived/pilot_ids.json`, `main_ids.json`, and `rep_ids.json`:
- `pilot_ids.json`: 200 question IDs
- `main_ids.json`: 500 question IDs
- `rep_ids.json`: 445 question IDs
- **Pairwise Overlaps:**
  - $\text{pilot} \cap \text{main} = 0$
  - $\text{pilot} \cap \text{rep} = 0$
  - $\text{main} \cap \text{rep} = 0$
- **Total Unique IDs:** $200 + 500 + 445 = 1145$ (100% of A-OKVQA validation set accounted for, strictly pairwise disjoint).

---

## SECTION 3: Reproducibility of Numbers

### 3.1 Pipeline Execution & Verification
1. `python scripts/analyze_vlm_all.py` executed cleanly, writing to `results/derived/vlm_main_summary.csv`, `tables/vlm_main_summary.tex`, and `figures/`.
2. `python scripts/analyze_vlm_all.py --raw_dir results/raw/vlm_rep --out_dir results/derived/vlm_rep --table_dir tables/vlm_rep --fig_dir figures/vlm_rep --no_siblings` executed cleanly for the replication sample.
3. `python scripts/check_numbers.py` executed cleanly (`[INFO] No \checknum macros found in 9 .tex file(s). All good.`).

### 3.2 Canonical Benchmark Table (Directly from Generated CSVs)

Bootstrap configuration: $B = 1000$ resamples, random seed = 42 (from `analyze_vlm_all.py`).

#### Main Benchmark ($N=500$, Seed 0, `results/derived/vlm_main_summary.csv`)

| Model | Corruption | Clean Acc | Null (Rem) | Null (Blk) | Swap Acc | $U(1)$ [95% CI] | $\hat{R}(1)$ [95% CI] | $S^*$ | Sig $S^*$ | MI | No-Crossing % | Mean Letter Mass |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **SmolVLM** | gaussian | 0.766 | 0.368 | 0.354 | 0.344 | −0.0220 [−0.0680, +0.0220] | 0.3808 [0.3373, 0.4263] | 1.000 (edge) | NaN | undefined | 76.9% | 0.9996 |
| **SmolVLM** | patch | 0.766 | 0.368 | 0.354 | 0.344 | +0.0040 [−0.0320, +0.0420] | 0.2331 [0.2282, 0.2488] | NaN | NaN | undefined | 100.0% | 0.9997 |
| **SmolVLM** | swapblend | 0.766 | 0.368 | 0.354 | 0.344 | −0.0160 [−0.0620, +0.0300] | 0.4932 [0.4477, 0.5479] | 1.000 (edge) | NaN | undefined | 57.1% | 0.9997 |
| **Qwen2-VL** | gaussian | 0.820 | 0.436 | 0.420 | 0.368 | −0.0300 [−0.0760, +0.0160] | 0.1681 [0.1475, 0.1891] | 1.000 (edge) | 0.875 | undefined | 62.3% | 0.9989 |
| **Qwen2-VL** | patch | 0.820 | 0.436 | 0.420 | 0.368 | −0.0080 [−0.0520, +0.0360] | 0.0290 [0.0271, 0.0315] | NaN | NaN | undefined | 100.0% | 0.9990 |
| **Qwen2-VL** | swapblend | 0.820 | 0.436 | 0.420 | 0.368 | −0.0640 [−0.1120, −0.0180] | 0.4872 [0.4368, 0.5433] | 0.875 | 1.000 | 0.4848 | 48.1% | 0.9989 |
| **LLaVA-1.5** | gaussian | 0.794 | 0.484 | 0.476 | 0.428 | −0.0220 [−0.0640, +0.0220] | 0.2947 [0.2544, 0.3365] | 1.000 (edge) | NaN | undefined | 94.2% | 0.9995 |
| **LLaVA-1.5** | patch | 0.794 | 0.484 | 0.476 | 0.428 | −0.0200 [−0.0660, +0.0240] | 0.2585 [0.2311, 0.3029] | NaN | NaN | undefined | 100.0% | 0.9995 |
| **LLaVA-1.5** | swapblend | 0.794 | 0.484 | 0.476 | 0.428 | −0.0480 [−0.0940, −0.0040] | 0.5784 [0.5167, 0.6548] | 1.000 (edge) | 1.000 | undefined | 52.8% | 0.9995 |

#### Replication Benchmark ($N=445$, Seed 1, `results/derived/vlm_rep/vlm_main_summary.csv`)

| Model | Corruption | Clean Acc | Null (Rem) | Null (Blk) | Swap Acc | $U(1)$ [95% CI] | $\hat{R}(1)$ [95% CI] | $S^*$ | Sig $S^*$ | MI | No-Crossing % | Mean Letter Mass |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **Qwen2-VL** | gaussian | 0.827 | 0.4607 | 0.4764 | 0.3663 | −0.0157 [−0.0584, +0.0270] | 0.1688 [0.1447, 0.1961] | 1.000 (edge) | NaN | undefined | 69.2% | 0.9990 |
| **Qwen2-VL** | swapblend | 0.827 | 0.4607 | 0.4764 | 0.3663 | −0.1034 [−0.1551, −0.0517] | 0.4700 [0.4182, 0.5228] | 0.875 | 0.875 | 0.4814 | 1.4% | 0.9990 |
| **LLaVA-1.5** | gaussian | 0.791 | 0.4966 | 0.4764 | 0.4787 | −0.0022 [−0.0427, +0.0360] | 0.2604 [0.2269, 0.3009] | 1.000 (edge) | NaN | undefined | 92.1% | 0.9995 |
| **LLaVA-1.5** | swapblend | 0.791 | 0.4966 | 0.4764 | 0.4787 | −0.0247 [−0.0719, +0.0225] | 0.5634 [0.5058, 0.6422] | 0.875 | NaN | 0.5740 | 44.9% | 0.9995 |

### 3.3 Explanation of Discrepancies vs Previous Assistant Report
The previous assistant response reported:
- Qwen clean accuracy as `0.812` (instead of canonical `0.820`)
- LLaVA clean accuracy as `0.776` (instead of canonical `0.794`)
- LLaVA removed accuracy as `0.478` (instead of canonical `0.484`)

**Audit Explanation:**
1. Direct evaluation of all 500 raw shard items in `results/raw/vlm_main/` confirms exact clean accuracy is $410/500 = \mathbf{0.8200}$ and removed null accuracy is $218/500 = \mathbf{0.4360}$.
2. Direct evaluation of all 500 raw shard items in `results/raw/vlm_llava/` confirms exact clean accuracy is $397/500 = \mathbf{0.7940}$ and removed null accuracy is $242/500 = \mathbf{0.4840}$.
3. The values `0.812` ($406/500$), `0.776` ($388/500$), and `0.478` ($239/500$) were never present in `results/derived/vlm_main_summary.csv` or any git commit. They originated as an ungrounded transcription error in the previous assistant's markdown response table. The CSV files and raw shards remain completely consistent.

### 3.4 SmolVLM Replication Metrics & Explanation of $R(0), R(1), \hat{R}(1)$
The SmolVLM replication raw shards in `results/raw/smolvlm_rep_out/` ($N=445$, seed 1) were evaluated under canonical metrics:

- **Swapblend:**
  - $R(0) = D(P_0, P_{\text{removed}}) = \mathbf{0.350679}$
  - $R(1) = D(P_1, P_{\text{removed}}) = \mathbf{0.193851}$
  - $\hat{R}(1) = \frac{R(1)}{R(0)} = \frac{0.193851}{0.350679} = \mathbf{0.552788}$
  - Clean Acc = $0.7461$, Null (Rem) Acc = $0.3933$, Terminal $S=1$ Acc = $0.3551$, $U(1) = -0.0382$
- **Gaussian:**
  - $R(0) = \mathbf{0.350679}$
  - $R(1) = \mathbf{0.113319}$
  - $\hat{R}(1) = \frac{0.113319}{0.350679} = \mathbf{0.323142}$
  - Clean Acc = $0.7461$, Null (Rem) Acc = $0.3933$, Terminal $S=1$ Acc = $0.3640$, $U(1) = -0.0292$

**Explanation of the "1.10" Artifact:**
In an earlier ad-hoc scratch script, the calculation was erroneously implemented as:
$$\text{ratio} = \frac{\text{mean}(\text{JSD}(P_0, P_1))}{\text{mean}(\text{JSD}(P_0, P_{\text{removed}}))} = \frac{0.3863}{0.3507} = 1.1016$$
This inverted the reference baseline by computing divergence from clean input $P_0$ rather than divergence from unimodal null $P_{\text{null}}$. Under the canonical definition in `src/metrics/metrics.py`, normalized reliance is $R(1)/R(0) = D(P_1, P_{\text{null}}) / D(P_0, P_{\text{null}}) = \mathbf{0.5528}$, strictly bounded within $[0, 1]$.

### 3.5 Letter Mass Audit
- Qwen2-VL: $0.9989$ (main), $0.9990$ (rep)
- LLaVA-1.5: $0.9995$ (main), $0.9995$ (rep)
- SmolVLM: $0.9996$ (main), $0.9996$ (rep)
- **Violation Check:** **No model in any condition has mean letter mass below 0.90.** All models clear the protocol gate with $>0.998$ probability mass on the designated multiple-choice choice tokens.

---

## SECTION 4: Pre-Declared Protocol Compliance

### 4.1 Chronology and Tag Verification in `DEVIATIONS.md`

| Entry # | Date | Tag / Context | Subject | Protocol Compliance Status |
|---|---|---|---|---|
| 1 | 2026-10-10 | `v0.1-synthetic` | Synthetic benchmark failure with zeros/mean null | Logged as deviation |
| 2 | 2026-10-10 | `v0.2-synthetic` | Null changed from zeros/mean to removed | Logged as post hoc fix |
| 3 | 2026-10-10 | `v0.3-synthetic` | Revised criteria, MI clipping, hierarchical bootstrap | Logged as post hoc |
| 4 | 2026-10-10 | `v0.4` | Unclipped MI, S* edge returns None, paired bootstrap | Logged as post hoc |
| 5 | 2026-10-10 | `v0.5-vlm-pilot-code` | VLM format adapted to 4-choice multiple choice | Logged before pilot |
| 6 | 2026-10-11 | `v0.7-main-code` | Swapblend corruption added, S* edge rules formalized | Logged as post hoc |
| 7 | 2026-10-11 | `v0.7.1-main-code` | SmolVLM token variants (bare + space-prefixed) | Logged before benchmark |
| 8 | 2026-10-11 | `7ccaff7` | **Pre-registered swapblend expectation** | **Pre-registered prior to data** |
| 9 | 2026-10-11 | `v0.7.2-main-code` | InternVL2 loading without `device_map="auto"` | Logged before benchmark |
| 10 | 2026-10-11 | `v0.7.3-main-code` | InternVL2 dropped due to transformers>=4.50 error | Logged before benchmark |
| 11 | 2026-10-11 | `v0.7.3-main-code` | SmolVLM assistant prefix (`answer_colon`) added | Logged before benchmark |
| 12 | 2026-10-11 | `v0.8-llava-code` | LLaVA-1.5-7B added as 4-bit replacement | Logged before benchmark |
| 13 | 2026-10-11 | `5501687` | **Modality dropout hypotheses H1, H2, H3** | **Pre-declared in commit** |
| 14 | 2026-10-11 | `v0.13.2` | Qwen main swapblend outcome vs pre-registered criterion | Logged outcome |
| 15 | 2026-10-11 | `v0.13.2` | Qwen evaluation provenance `not_a_git_repo` | Logged environment artifact |
| 16 | 2026-10-11 | `v0.13.3` | Gaussian crossing smoothing lag explanation | Logged post hoc analysis |
| 17 | 2026-10-11 | `v0.13.3` | Swapblend MI sensitivity & reporting $\hat{R}(1)$ | Logged post hoc analysis |
| 18 | 2026-10-11 | `v0.13.3` | Black-null reliance floor table | Logged post hoc analysis |
| 19 | 2026-10-11 | `v0.13.4` | LLaVA token lookup SentencePiece fix | Logged before benchmark |
| 20 | 2026-10-11 | `v0.14.2` | LLaVA main outcome vs pre-registered criterion | Logged outcome |
| 21 | 2026-10-12 | `v0.15.1-qwen-rep`| Qwen replication outcome on held-out questions | Logged outcome |
| 22 | 2026-10-12 | `v0.16.1-analysis`| SmolVLM main outcome & prefix selection | Logged outcome |
| 23 | 2026-10-12 | `v0.16.1-analysis`| LLaVA replication outcome on held-out questions | Logged outcome |

### 4.2 Detailed Compliance Checks
- **Check (a) Pre-registered Swapblend Expectation:**  
  Committed in `7ccaff7da5e0b0d4c99a1e1fb9eb6569d1d69354` on `2026-10-11 11:56:11 +0530`.  
  Qwen main results were committed in `93e60aabf6b69b66a8c977b6b4ff1b8245472000` on `2026-10-11 14:39:35 +0530`.  
  **Result:** **COMPLIANT** (pre-registered 2 hours and 43 minutes before main results were committed).
- **Check (b) Modality Dropout Hypotheses (H1-H3):**  
  Committed in `5501687793be3cae5dd742a93be7cee980c0f748` on `2026-10-11 14:27:20 +0530`.  
  `src/fusion/train.py` was introduced in the **exact same commit** (`5501687`).  
  **Result:** **COMPLIANT** (written alongside the initial code introduction; no fusion training results were produced or committed until Kaggle execution on 2026-10-12).
- **Check (c) Post Hoc Changes:**  
  All protocol alterations (null baseline switch, metric un-clipping, moving-average boundary rule, black null floor) are explicitly tagged with `post hoc` in both the change and reason columns.  
  **Result:** **COMPLIANT**.
- **Check (d) Gate Thresholds:**  
  Letter mass threshold ($0.90$) was set in commit `26a42e9` and remained unaltered throughout. Pilot gate rules and item counts ($N=200$, $N=500$, $N=445$) remained identical.  
  **Result:** **COMPLIANT** (0 threshold violations).

---

## SECTION 5: Fusion Study Status

### 5.1 Datasets and Runs Present
- `results/raw/fusion/`: Empty (`[]`).
- `results/raw/fusion_out/` *(untracked)*: Contains the **complete 20-run grid** for Fashion-MNIST:
  - Dataset: `fashionmnist`
  - $p$-values: `[0.0, 0.1, 0.25, 0.5]`
  - Seeds: `[0, 1, 2, 3, 4]`
  - Total files: 20 `predictions.json` files + 20 `model.pt` checkpoints (40 files).

### 5.2 Empirical Performance of Fashion-MNIST Runs

| Dropout $p$ | Clean Accuracy (S=0) [Chance=0.100] | Null Removed Accuracy | Terminal S=1 Accuracy | H2 Check ($\Delta \text{Clean} < 3\text{ pp}$) |
|---|---|---|---|---|
| $p=0.0$ | **0.8879** | 0.5620 | 0.2761 | Baseline |
| $p=0.1$ | **0.8877** | 0.8396 | 0.4480 | $-0.02$ pp |
| $p=0.25$ | **0.8843** | 0.8477 | 0.5550 | **$-0.36$ pp** ($\ll 3$ pp, **H2 PASSES**) |
| $p=0.5$ | **0.8823** | 0.8584 | 0.7256 | $-0.56$ pp |

- **Clean Accuracy:** Clean accuracy is **88.4%–88.8%**, far above random chance ($10.0\%$).
- **Null Accuracy Elevation:** Training with modality dropout forces the network to learn robust unimodal representations on modality B, raising unimodal null accuracy from $56.2\%$ ($p=0$) to $84.8\%$ ($p=0.25$).
- **Execution of `analyze_fusion.py`:** Has **NOT** been run yet (no derived files in `results/derived/fusion*` or `tables/fusion*`).

### 5.3 What Remains for Full Fusion Grid
1. Move `results/raw/fusion_out/` to canonical `results/raw/fusion/`, run `scripts/hash_results.py`, commit, and run `scripts/analyze_fusion.py` to evaluate H1 (monotonicity of MI in $p$) with bootstrap CIs.
2. Synthetic fusion 20-run grid: Has not been run (only synthetic v0.1/v0.2/v0.3 exist under `results/raw/synthetic*`). Can be run on CPU in ~2.5 hours if desired.

---

## SECTION 6: Paper Status

### 6.1 Word Counts and TODO Markers

| File Path | Word Count | `\TODO` Markers |
|---|---|---|
| `paper/main.tex` | 71 | 0 |
| `paper/macros.tex` | 37 | 0 |
| `paper/results_notes.md` | 196 | 0 |
| `paper/spec.md` | 512 | 0 |
| `paper/sections/abstract.tex` | 129 | 1 |
| `paper/sections/intro.tex` | 134 | 2 |
| `paper/sections/method.tex` | 441 | 0 |
| `paper/sections/synthetic.tex` | 173 | 0 |
| `paper/sections/results.tex` | 123 | 4 |
| `paper/sections/dropout.tex` | 121 | 1 |
| `paper/sections/limitations.tex` | 326 | 0 |
| **Total Section Prose** | **1,467 words** | **8 TODOs** |

### 6.2 Empirical Grounding Audit in Paper Sections
Every section was inspected for hardcoded numerical claims:
- `abstract.tex`: Contains no hardcoded empirical numbers; includes 1 `\TODO` placeholder to summarize findings.
- `intro.tex`: Contains no hardcoded empirical numbers; includes 1 `\TODO` placeholder.
- `method.tex`: Pure mathematical formalization of $U(S), R(S), \hat{R}(S), S^*, \text{MI}$; matches `src/metrics/metrics.py`.
- `synthetic.tex`: Grounded via `\input{tables/synthetic_summary.tex}` and vector figures.
- `results.tex`: Grounded via `\input{tables/vlm_main_summary.tex}`; empirical narrative text is marked with 4 `\TODO` comments.
- `dropout.tex`: States pre-registered hypotheses H1/H2; includes 1 `\TODO` placeholder for `\input{tables/fusion_summary.tex}`.
- `limitations.tex`: Methodological and architectural caveats (6 numbered points); no unbacked numerical claims.
- **Finding:** **Zero ungrounded empirical sentences found.** All presented data is backed by `\input` tables or explicit hypotheses.

### 6.3 Compilation Status
- **LaTeX Toolchain Check:** `pdflatex`, `xelatex`, and `latexmk` are not installed in the Windows environment.
- **Compilation Status:** **UNVERIFIED (no local LaTeX toolchain available)**.

### 6.4 `FREEZE.md` Audit
- **Experiments on `FREEZE.md` that are RUN:**
  - VLM Main ($N=500$, seed 0): Qwen2-VL, SmolVLM, LLaVA-1.5 across gaussian, patch, swapblend.
  - Fashion-MNIST Fusion ($p \in \{0, 0.1, 0.25, 0.5\} \times 5$ seeds): Completed on Kaggle GPU.
  - BEV probe: Feasibility check completed, verdict NO-GO documented.
- **Experiments on `FREEZE.md` that are NOT RUN:**
  - VLM Main Seeds 1 and 2 on the original 500 questions (Seed 1 was executed as an independent replication on 445 *held-out* questions instead).
  - Synthetic Two-View Modality Dropout Fusion 20-run grid.
- **Experiments Run but NOT in original `FREEZE.md`:**
  - Replication VLM Benchmark on $N=445$ held-out questions (seed 1).

---

## SECTION 7: Completion Scorecard

| Area | Status | Evidence |
|---|---|---|
| **Scope & Repo Setup** | **DONE** | `FREEZE.md`, `DEVIATIONS.md`, git configuration, and environment setup locked. |
| **Formal Framework** | **DONE** | Complete $U(S), R(S), \hat{R}(S), S^*, \text{MI}$ implementation in `src/metrics/metrics.py`. |
| **Synthetic Validation** | **DONE** | 150 shards in `results/raw/synthetic_v03/`, summary table and 4 vector figures generated. |
| **VLM Pilot** | **DONE** | 16 shards in `results/raw/vlm_pilot/`, tag `v0.6-vlm-pilot-results`. |
| **Qwen Main** | **DONE** | 60 shards in `results/raw/vlm_main/`, tag `v0.8-qwen-results`. |
| **LLaVA Main** | **DONE** | 60 shards in `results/raw/vlm_llava/`, tag `v0.14-llava-results`. |
| **SmolVLM Main** | **DONE** | 60 shards in `results/raw/vlm_smol/`, tag `v0.16-smol-results`. |
| **Replications** | **PARTIAL** | Qwen & LLaVA committed in `vlm_rep/`; SmolVLM downloaded in `smolvlm_rep_out/` (needs commit). |
| **Fusion Study** | **PARTIAL** | Fashion-MNIST 20 runs completed in `fusion_out/`; needs ingestion and `analyze_fusion.py`. |
| **BEV Probe** | **DONE** | `scripts/bev_feasibility_check.py` executed; NO-GO documented under tag `v0.12-bev-probe`. |
| **Analysis & Tables** | **DONE** | `scripts/analyze_vlm_all.py` generates full tables; replication tables isolated in `tables/vlm_rep/`. |
| **Figures** | **DONE** | Vector PDF curves generated in `figures/` and `figures/vlm_rep/`. |
| **Paper Method** | **DONE** | `paper/sections/method.tex` completely written with exact mathematical formulas. |
| **Paper Results** | **PARTIAL** | Tables inserted, but narrative text contains 4 `\TODO` blocks awaiting final synthesis. |
| **Paper Discussion & Limitations** | **PARTIAL** | `limitations.tex` complete (6 enumerated points); discussion needs final synthesis. |
| **Related Work** | **NOT STARTED**| No `related_work.tex` file exists in `paper/sections/`. |
| **Reproducibility Release** | **PARTIAL** | `CITATION.cff` skeleton present with TODOs; Zenodo deposit not completed. |

**Overall Project Completion:** **11 / 17 rows = 64.7% (unweighted count of DONE rows)**.

---

## SECTION 8: Risks and Next Actions

### 8.1 Top 5 Risks to the Paper

1. **Overclaiming on Point Thresholds ($\hat{R}(1) > 0.50$):**  
   The pre-registered threshold $\hat{R}(1) > 0.50$ under swapblend is strictly cleared only by LLaVA-1.5-7B (main: $0.578$ $[0.517, 0.655]$; rep: $0.563$ $[0.506, 0.642]$). For Qwen2-VL (main: $0.487$ $[0.437, 0.543]$; rep: $0.470$ $[0.418, 0.523]$) and SmolVLM (main: $0.493$ $[0.448, 0.548]$), the 95% CIs straddle 0.50. Furthermore, SmolVLM terminal utility ($U(1) = -0.016$) is not statistically significantly below zero. The paper must frame Qwen and SmolVLM as *borderline/inconclusive* on the strict point threshold, rather than claiming uniform misalignment.
2. **Undefined Misalignment Index ($\text{MI}$) in Benign Corruptions:**  
   In Gaussian blur and patch occlusion, utility rarely crosses zero ($S^* = 1.0$ or $\text{NaN}$), causing 62%–100% of bootstrap resamples to drop. As a result, $\text{MI}$ is undefined for all models under benign corruptions. The paper must report $\hat{R}(1)$ and $U(1)$ as the primary robust metrics and explain why $\text{MI}$ is defined only under genuine utility inversion (swapblend).
3. **Untracked Raw Data in Working Tree:**  
   The 20 Fashion-MNIST fusion runs (`results/raw/fusion_out/`) and 36 SmolVLM replication shards (`results/raw/smolvlm_rep_out/`) represent ~5 hours of GPU compute but are currently untracked files in the working directory. An accidental clean or git reset could destroy this data.
4. **Unverified LaTeX Compilation:**  
   No LaTeX toolchain is present locally. Macro interactions, table sizing (e.g. `vlm_main_summary.tex` has 27 columns), and package dependencies in `paper/main.tex` have not been verified in a live compiler.
5. **Generality Limited to a Single Task / Benchmark:**  
   All VLM experiments use A-OKVQA validation questions. While replication over held-out questions and novel donor pairings confirms the finding does not depend on a specific question sample, it remains restricted to a single visual question-answering dataset.

### 8.2 Ordered List of Remaining Actions

1. **[Laptop-only] Ingest SmolVLM Replication Data:**  
   Move `results/raw/smolvlm_rep_out/*.json` into `results/raw/vlm_rep/`, update `results/raw/MANIFEST.sha256`, commit with message `results(vlm-rep): SmolVLM-Instruct replication, 445 held-out questions, seed 1`, and tag `v0.15.3-smol-rep`.
2. **[Laptop-only] Ingest Fashion-MNIST Fusion Data:**  
   Move `results/raw/fusion_out/*` into `results/raw/fusion/`, update `results/raw/MANIFEST.sha256`, commit with message `results(fusion): Fashion-MNIST late fusion 20-run grid`, and tag `v0.17-fusion-results`.
3. **[Laptop-only] Run Fusion Analysis:**  
   Execute `python scripts/analyze_fusion.py` to generate `results/derived/fusion_summary.csv`, `tables/fusion_summary.tex`, and publication figures evaluating H1 and H2.
4. **[Laptop-only] Re-run Replication Analysis with All 3 Models:**  
   Execute `python scripts/analyze_vlm_all.py --raw_dir results/raw/vlm_rep --out_dir results/derived/vlm_rep --table_dir tables/vlm_rep --fig_dir figures/vlm_rep --no_siblings` to produce the final 3-model replication table.
5. **[Writing] Draft Paper Results Section:**  
   Populate `paper/sections/results.tex` with empirical prose matching the generated tables and figures.
6. **[Writing] Draft Related Work Section:**  
   Author `paper/sections/related_work.tex` covering multimodal fusion failure modes, modality reliance, and post-hoc explanation methods.
7. **[Writing] Complete Abstract, Intro, and Limitations Synthesis:**  
   Resolve remaining `\TODO` markers across `abstract.tex`, `intro.tex`, `dropout.tex`, and update `CITATION.cff`.
8. **[Writing / Compilation] Compile and Format Paper:**  
   Transfer `paper/` to Overleaf or a TeX Live environment to verify compilation, table formatting, and page budgets.

---

## SECTION 9: Items Unsure About or Flagged for User Direction

1. **Synthetic Two-View Fusion Grid:**  
   `FREEZE.md` specified running the 20-run modality-dropout grid on both `fashionmnist` and `synthetic`. The Fashion-MNIST runs are complete. The synthetic fusion runner can be executed locally on CPU in ~2.5 hours (`python scripts/train_fusion.py --dataset synthetic --p 0,0.1,0.25,0.5 --seeds 0,1,2,3,4`), or skipped if Fashion-MNIST is sufficient for the paper's case study.
2. **Main Seeds 1 and 2 vs Replication:**  
   `FREEZE.md` originally specified evaluating seeds 0, 1, and 2 on the main $N=500$ sample. In practice, seed 1 was run on $N=445$ *held-out* questions as an independent replication rather than a re-evaluation of the main 500 questions. This represents a stronger methodological test of generalizability, but deviates from the literal text of `FREEZE.md`.
3. **Table Width in LaTeX:**  
   `tables/vlm_main_summary.tex` contains 27 columns generated directly by `analyze_vlm_all.py`. In standard two-column conference formats (CVPR/ICML/NeurIPS), this table will exceed text width unless formatted as a selected sub-table (as in `scripts/evaluate_all_tasks.py`).
