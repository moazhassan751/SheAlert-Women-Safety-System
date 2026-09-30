# SheAlert — Audio Corpus Consolidation Pipeline

Consolidates all 8 external audio corpora (CREMA-D, Radvess/RAVDESS, UrduSER,
URDU-Dataset-master, RealWorld-Urdu, SEMOUR+, UrduSpeech, AudioSet_Screaming,
ESC-50, UrbanSound8K) into one unified manifest under the 3-class taxonomy
(Distress / Aggression / Normal) from FR-4.0 of the scope doc.

Paths in `config.py` are pre-filled to match your actual Google Drive layout
(`/content/drive/MyDrive/FYP DATASETS/...`) from your dataset scan — should
work as-is if you run this in Colab with Drive mounted. Change `BASE_PATH`
if running locally instead.

This is Stage 1 input only (FR-4.4) — your original Urdu distress recordings
stay out of this pipeline on purpose, so Stage 1 / Stage 2 fine-tuning stay
cleanly separated.

## What each dataset actually looks like (confirmed against your scan)

| Dataset | Structure | Loader |
|---|---|---|
| CREMA-D | flat `AudioMP3/`, emotion in filename, **.mp3** | `cremad.py` |
| Radvess | `Actor_NN/`, emotion in filename | `ravdess.py` |
| UrduSER | folder-per-emotion (`Angry/`, `Fear/`, ...) | `folder_emotion.py` |
| URDU-Dataset-master | folder-per-emotion (**no Fear class**) | `folder_emotion.py` |
| RealWorld-Urdu | folder-per-emotion (**no Fear class**, from seniors' prior FYP) | `folder_emotion.py` |
| SEMOUR+ | `Actor_N/Emotion/`, nested | `folder_emotion.py` |
| UrduSpeech | already pre-sorted into `Aggression/Distress/Normal/` | `folder_emotion.py` |
| AudioSet_Screaming | flat `wav/`, all scream → Distress | `audioset_scream.py` |
| ESC-50 | `audio/` + `meta/esc50.csv`, all → Normal | `esc50.py` |
| UrbanSound8K | metadata-CSV driven, all → Normal (gun_shot excluded) | `urbansound8k.py` |

## Known gaps to fix before running for real

- **ESC-50 / UrbanSound8K metadata paths are guessed** — your scan showed
  the audio folders but not confirmed metadata CSV filenames/locations.
  Fix `esc50_meta` / `urbansound8k_meta` in `config.py` if they don't match.
- **UrbanSound8K shows 0 files in your scan** — not populated on Drive yet.
  The loader just skips cleanly with a warning until it is.
- **SEMOUR+ has some corrupt audio files** (you confirmed this) — the QC
  gate catches these automatically and reports the count in
  `corpus_summary.txt` under "unreadable". No action needed, just check
  the count looks reasonable once you run it for real.

## Setup

```bash
pip install -r requirements.txt
```

## Run

```bash
python consolidate.py              # light QC (corrupt/empty + clipping only)
python consolidate.py --no-qc      # skip QC entirely (fastest first look)
python consolidate.py --resample   # also writes 16kHz-mono copies to output/audio_16k/
```

Output:
- `output/unified_manifest.csv` — filepath, label, source_dataset, speaker_id
  for every clip that passed QC
- `output/corpus_summary.txt` — class balance, per-source counts, rejection
  reasons (also printed to stdout)

## Why QC is light here, not the full Section-10 gate

Section 10's "reject clips <3s / SNR <10dB" rule is written for the team's
own original Urdu recordings. Several public corpora are short by design
(SEMOUR+ clips average ~1-1.5s, CREMA-D ~2.2s) — applying that floor here
would gut both for no real reason, since they're already peer-reviewed,
published datasets. This stage only filters genuinely broken/corrupt files
and hard clipping. The strict 3s/SNR/clipping gate belongs to a separate
script for the original-recording pipeline once Module 4 collection is
underway.

## Structure

```
config.py            dataset paths (pre-filled for your Drive layout) + label taxonomy + QC thresholds
utils.py              duration/clipping/SNR QC gate (mp3-safe via librosa fallback)
loaders/
  cremad.py            filename-parsed, .mp3-aware
  ravdess.py            filename-parsed
  folder_emotion.py     generic loader for UrduSER/URDU-Dataset-master/RealWorld-Urdu/SEMOUR+/UrduSpeech
  esc50.py               metadata-CSV driven, all -> Normal
  urbansound8k.py         metadata-CSV driven, all -> Normal (gun_shot excluded)
  audioset_scream.py     flat folder, all -> Distress
consolidate.py         runs everything, applies QC, writes manifest + summary
```

## Tested

Verified end-to-end against a synthetic mirror of your actual Drive tree
(built from your dataset_scan.json), including a deliberately corrupted
file to reproduce the SEMOUR+ issue: all 5 folder-based datasets correctly
mapped labels per-folder, CREMA-D and RAVDESS filename parsing confirmed
correct, class-not-in-taxonomy folders (Happy/Sad/Boredom/etc.) correctly
skipped, and the corrupt file was correctly caught and flagged rather than
silently dropped or silently crashing the run.
