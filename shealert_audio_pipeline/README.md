# SheAlert — Audio Corpus Consolidation Pipeline

Consolidates all 10 external audio corpora into one unified manifest under the 3-class taxonomy (**Distress**, **Aggression**, **Normal**) from **FR-4.0** of the SheAlert specification.

> 📚 **Detailed Engineering Specs:** See [`docs/MODULE_3_AUDIO_PIPELINE.md`](file:///f:/SheAlert-Women-Safety-System/docs/MODULE_3_AUDIO_PIPELINE.md) for full mathematical and algorithmic details, clipping analysis, and split proofs.  
> 🎤 **FYP Panel Reference:** See [`docs/PANEL_PRESENTATION_CHEATSHEET.md`](file:///f:/SheAlert-Women-Safety-System/docs/PANEL_PRESENTATION_CHEATSHEET.md) for panel defense metrics and answers.

---

## What each dataset looks like (Integrated & Verified)

| Dataset | Structure | Clips Post-QC | Loader |
|---|---|---|---|
| **CREMA-D** | flat `AudioMP3/`, emotion in filename, `.mp3` | 2,534 | `loaders/cremad.py` |
| **RAVDESS** | `Actor_NN/`, emotion in filename, `.wav` | 384 | `loaders/ravdess.py` |
| **UrduSER** | folder-per-emotion (`Angry/`, `Fear/`, `Neutral/`) | 1,500 | `loaders/folder_emotion.py` |
| **URDU-Dataset-master** | folder-per-emotion (`Angry/`, `Neutral/`) | 200 | `loaders/folder_emotion.py` |
| **RealWorld-Urdu** | folder-per-emotion (`Aggressive/`, `Normal/`, OOD Set) | 77 | `loaders/folder_emotion.py` |
| **SEMOUR+** | `Actor_N/Emotion/`, nested | 10,435 | `loaders/folder_emotion.py` |
| **UrduSpeech** | pre-sorted into `Aggression/Distress/Normal/` | 5,608 | `loaders/folder_emotion.py` |
| **AudioSet_Screaming** | flat `wav/`, all scream → Distress | 830 | `loaders/audioset_scream.py` |
| **ESC-50** | `audio/` + `meta/esc50.csv`, all → Normal | 1,736 | `loaders/esc50.py` |
| **UrbanSound8K** | metadata-CSV driven, all → Normal (gun_shot excluded) | 6,960 | `loaders/urbansound8k.py` |

---

## Pipeline Features & Invariants

1. **Robust Quality Control Gate:**
   - 60 ms minimum duration threshold.
   - 10 dB minimum estimated SNR.
   - Dynamic relative clipping threshold: $\text{Peak} - 0.05 \cdot (\text{Peak} - \text{RMS})$ with $0.5\%$ sample ceiling. Rescued **688 authentic distress screams** while filtering true square-wave flat-top distortion.
2. **Ambient Subsampling ($\le 40\%$ Normal Class Cap):**
   - Keeps all $21,568$ speech clips from the 8 dialogue datasets.
   - Subsamples UrbanSound8K ($640$ clips) and ESC-50 ($160$ clips) down to $800$ ambient clips.
   - Output: `training_manifest.csv` with **33.41% Distress, 29.65% Aggression, 36.93% Normal**.
3. **5-Fold Stratified Group Split (Seed 11):**
   - Partitioned via greedy smallest-bin-first allocation + local search move/swap refinement.
   - Equal fold sizes ($4,458$ clips each, **0.02% variation**).
   - Identical class distributions ($33.5\%$ Distress, $29.6\%$ Aggression, $36.9\%$ Normal across all 5 folds).
   - **Zero group/speaker leakage** across folds.
   - All 9 non-OOD datasets represented in every fold.
   - RealWorld-Urdu ($77$ clips) isolated as pure out-of-distribution benchmark.

---

## Setup & Running

```bash
# Install dependencies
pip install -r requirements.txt

# Run full pipeline with QC and 5-fold stratification
python consolidate.py

# Run automated test suite
pytest tests/ -v
```

---

## Primary Output Files (`shealert_audio_pipeline/output/`)

* `unified_manifest.csv`: 30,264 clean clips passing QC across all 10 datasets.
* `training_manifest.csv`: 22,368 clips with ambient subsampled to satisfy Section 10 Normal $\le 40\%$ cap.
* `splits.csv`: Deterministic 5-fold group split assignments + OOD split.
* `rejected_clips.csv`: 1,692 rejected audio files with exact failure reasons.
* `SPLIT_INFO.txt`: Mathematical verification and SHA-256 integrity hash of splits.
* `corpus_summary.txt`: Full breakdown of dataset counts, class balances, and fold statistics.
