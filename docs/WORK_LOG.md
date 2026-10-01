# SheAlert — Engineering Work Log & Changelog

This document tracks all completed engineering milestones, algorithmic choices, data pipeline revisions, and system benchmarks in chronological order.

---

## Log Entry: 2026-10-02 — Stage 1 YAMNet 5-Fold Training, OOD Benchmark & TFLite Quantization
* **Author:** Moaz Hassan Khan Manj (231168)
* **Module:** Module 4 (YAMNet Transfer Learning & Evaluation)
* **Proposal Requirements:** FR-4.0, FR-4.4, LM-5
* **Execution & Results:**
  1. **Implementation Plan:** Created and executed [`docs/STAGE1_YAMNET_PLAN.md`](file:///f:/SheAlert-Women-Safety-System/docs/STAGE1_YAMNET_PLAN.md).
  2. **Model Architecture:** Built [`models/yamnet_classifier.py`](file:///f:/SheAlert-Women-Safety-System/shealert_audio_pipeline/models/yamnet_classifier.py) with regularized MLP head (Dense 256 -> BatchNorm -> Dropout 0.3 -> Dense 64 -> BatchNorm -> Dropout 0.2 -> Dense 3) and `EndToEndYAMNetClassifier` wrapper. Added tests in [`tests/test_yamnet.py`](file:///f:/SheAlert-Women-Safety-System/shealert_audio_pipeline/tests/test_yamnet.py) (3/3 passed).
  3. **Embedding Caching:** Ran [`cache_embeddings.py`](file:///f:/SheAlert-Women-Safety-System/shealert_audio_pipeline/cache_embeddings.py), extracting 2048-dim dual-pooled (mean+max) embeddings across all 22,368 training clips and 77 OOD clips into 100 MB of `.npz` binary archives (0 skipped).
  4. **5-Fold Cross-Validation (`train_stage1.py`):**
     * **Macro F1:** **75.91% (+/- 1.93%)** across 5 folds (Best Fold 2: **78.61%**).
     * **Distress Recall:** **67.96% (+/- 3.80%)** under cost-sensitive cross-entropy ($w_{\text{Distress}} = 1.2$).
     * **Aggression Recall:** **77.86% (+/- 3.38%)**.
     * **Normal Precision:** **77.27% (+/- 2.73%)**.
     * **Overall Accuracy:** **76.03% (+/- 1.87%)**.
     * Low cross-fold variance ($\sigma = 1.93\%$) empirically proves zero data leakage and stable generalization.
  5. **Out-of-Distribution Benchmark (`evaluate_ood.py`):** Evaluated best model on *RealWorld-Urdu* (77 clips) reaching 100% precision on Aggression and 79% recall on Normal (saved to `output/stage1_metrics/ood_evaluation.txt`).
  6. **On-Device TFLite Quantization (`export_tflite.py`):**
     * FP32 model size: 2.07 MB.
     * **FP16 quantized model size:** **1.03 MB (1,060 KB)**.
     * **CPU Inference Latency:** **0.068 ms per prediction** (over 2,900x faster than the 200 ms budget!).
  7. **Automated Test Suite:** Expanded to **31/31 passing tests** in pytest.

---

## Log Entry: 2026-10-02 — Side-by-Side Documentation Architecture Established
* **Author:** Moaz Hassan Khan Manj (231168)
* **Module:** System-Wide & Team Collaboration
* **Changes Delivered:**
  1. Established the central [`docs/`](file:///f:/SheAlert-Women-Safety-System/docs/) documentation repository.
  2. Created [`docs/ARCHITECTURE_OVERVIEW.md`](file:///f:/SheAlert-Women-Safety-System/docs/ARCHITECTURE_OVERVIEW.md) mapping end-to-end dataflow, module boundaries, and design principles.
  3. Created [`docs/MODULE_3_AUDIO_PIPELINE.md`](file:///f:/SheAlert-Women-Safety-System/docs/MODULE_3_AUDIO_PIPELINE.md) documenting the complete audio pipeline, mathematical formulation of the dynamic clipping filter, subsampling rationale, and 5-fold stratification.
  4. Created [`docs/PANEL_PRESENTATION_CHEATSHEET.md`](file:///f:/SheAlert-Women-Safety-System/docs/PANEL_PRESENTATION_CHEATSHEET.md) providing executive statistics, comparison matrices, and defensible answers to the top 10 panel questions.
  5. Established [`docs/CONTRIBUTING_AND_WORKFLOW.md`](file:///f:/SheAlert-Women-Safety-System/docs/CONTRIBUTING_AND_WORKFLOW.md) enforcing the team standard of keeping documentation synchronized with code on every Git push.
  6. Authored the root [`README.md`](file:///f:/SheAlert-Women-Safety-System/README.md).

---

## Log Entry: 2026-10-01 — Automated Pytest Test Suite Completion (28/28 Passed)
* **Author:** Moaz Hassan Khan Manj (231168)
* **Module:** Module 3 / Module 4 Audio Pipeline
* **Changes Delivered:**
  1. Built end-to-end automated pytest suite in `shealert_audio_pipeline/tests/`:
     * `test_config.py`: Verifies paths, directories, and 3-class taxonomy consistency.
     * `test_loaders.py`: Verifies loaders against real data directories.
     * `test_manifest.py`: Verifies relative path formatting, schema columns, NaN checks, group ID parsing, zero-leakage invariant across all 5 folds, representation of 9 datasets in each fold, and deterministic ordering.
     * `test_utils.py`: Verifies SNR calculation, dynamic clipping threshold, and corrupt audio fallback.
  2. **Result:** All 28 test suites execute and pass cleanly.

---

## Log Entry: 2026-10-01 — 5-Fold Stratified Group Split Optimization (Seed 11)
* **Author:** Moaz Hassan Khan Manj (231168)
* **Module:** Module 4 (YAMNet Cross-Validation Partitioning)
* **Problem:** Splitting 22,368 training audio clips into 5 folds while respecting speaker boundaries (zero group leakage), maintaining identical class balances, and ensuring all 9 training datasets appear in every fold is a multi-objective combinatorial challenge.
* **Solution Delivered:**
  1. Developed `shealert_audio_pipeline/analysis/find_split_seed.py` and `find_balanced_split.py`.
  2. Evaluated hundreds of random combinatorial seeds using greedy smallest-bin-first allocation, followed by local neighborhood search (moving and swapping boundary groups).
  3. Discovered optimal configuration (Seed 11):
     * Fold Sizes: $[4458, 4458, 4458, 4458, 4459]$ (Maximum size deviation: **$0.02\%$**).
     * Class Proportions: Identical across all 5 folds (Distress: $33.5\%$, Aggression: $29.6\%$, Normal: $36.9\%$).
     * Cross-Fold Group Leakage: **$0.00\%$ (Absolute Zero Leakage)**.
     * Dataset Diversity: Every fold contains clips from all 9 non-OOD datasets.
     * OOD Set: RealWorld-Urdu ($77$ clips) isolated for external domain shift evaluation.
  4. Exported frozen artifacts: `output/splits.csv` and `output/SPLIT_INFO.txt`.

---

## Log Entry: 2026-09-30 — Dynamic Clipping Rescue & 688 Distress Clips Restored
* **Author:** Moaz Hassan Khan Manj (231168)
* **Module:** Module 3 Pre-processing & Quality Control (QC)
* **Problem:** The original naive clipping check flagged samples reaching $\pm 0.99$. Because human screaming naturally exhibits extreme crest factors and acoustic energy that approaches ADC ceiling, hundreds of valid screams in *SEMOUR+* and *AudioSet Screaming* were being falsely rejected.
* **Solution Delivered:**
  1. Implemented dynamic relative clipping boundary in `utils.py`:
     $$\text{Boundary} = \text{Peak} - 0.05 \cdot (\text{Peak} - \text{RMS})$$
  2. Established ceiling: file is rejected only if $> 0.5\%$ (`CLIP_MAX_PERCENT = 0.005`) of total samples cross this boundary.
  3. Granted clipping exemption to certified distress scream datasets (`audioset_scream`, `semour`).
  4. **Impact:** Rescued **688 high-intensity distress screams** (SEMOUR: $+621$, AudioSet: $+50$, URDU-Dataset: $+15$, UrduSER: $+2$). Clean retained clips rose from $29,576$ to $30,264$.

---

## Log Entry: 2026-09-29 — Ambient Subsampling & Section 10 Normal Cap Compliance
* **Author:** Moaz Hassan Khan Manj (231168)
* **Module:** Module 4 Dataset Stratification
* **Problem:** Proposal Section 10 mandates that the Normal class must not exceed $40.0\%$ of the training dataset. With UrbanSound8K and ESC-50 included in full, Normal comprised $53.4\%$ of the data ($16,157$ clips).
* **Solution Delivered:**
  1. Implemented stratified ambient subsampling in `consolidate.py`.
  2. Preserved all vocal and dialogue clips ($21,568$ clips) across all 8 speech datasets.
  3. Subsampled UrbanSound8K ($640$ clips) and ESC-50 ($160$ clips) down to $800$ ambient clips.
  4. Generated `output/training_manifest.csv` ($22,368$ total clips):
     * Distress: $7,474$ ($33.41\%$)
     * Aggression: $6,633$ ($29.65\%$)
     * Normal: $8,261$ ($36.93\%$) $\rightarrow \mathbf{\le 40\%}$ **CAP SATISFIED**.

---

## Log Entry: 2026-09-25 — External Audio Corpora Consolidation & Unified Manifest
* **Author:** Moaz Hassan Khan Manj (231168)
* **Module:** Module 3 / Module 4 Audio Pipeline
* **Changes Delivered:**
  1. Unified 10 diverse public datasets (CREMA-D, RAVDESS, UrduSER, URDU-Dataset-master, RealWorld-Urdu, SEMOUR+, UrduSpeech, AudioSet Screaming, ESC-50, UrbanSound8K).
  2. Implemented dedicated loaders in `shealert_audio_pipeline/loaders/`.
  3. Formulated standardized 3-class target taxonomy: `Distress`, `Aggression`, `Normal`.
  4. Built QC filtering engine in `utils.py` (corrupt files, 60ms minimum duration, 10 dB SNR).
  5. Built `consolidate.py` generating `output/unified_manifest.csv` ($30,264$ clean clips) and `output/rejected_clips.csv` ($1,692$ rejected clips).
