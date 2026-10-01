"""
config.py
---------
Paths and label mappings, filled in directly from your dataset_scan.json —
these should work as-is if you run this in the same Colab session with
Drive mounted at /content/drive/MyDrive. If you're running locally instead,
just change BASE_PATH.
"""

import os
from pathlib import Path

# -----------------------------------------------------------------------
# 1. UNIFIED LABEL TAXONOMY  (matches FR-4.0 class mapping in the scope doc)
# -----------------------------------------------------------------------
LABELS = ["Distress", "Aggression", "Normal"]

# -----------------------------------------------------------------------
# 2. BASE PATH — Auto-detect local workspace, environment variable, or Colab
# -----------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
LOCAL_DATA_DIR = REPO_ROOT / "audio data set fyp"
ALT_LOCAL_DIR = Path("F:/FYP DATASETS")
COLAB_DATA_DIR = Path("/content/drive/MyDrive/FYP DATASETS")

if os.environ.get("SHEALERT_DATA_DIR") and Path(os.environ["SHEALERT_DATA_DIR"]).exists():
    BASE_PATH = Path(os.environ["SHEALERT_DATA_DIR"])
elif os.environ.get("FYP_DATASETS_DIR") and Path(os.environ["FYP_DATASETS_DIR"]).exists():
    BASE_PATH = Path(os.environ["FYP_DATASETS_DIR"])
elif LOCAL_DATA_DIR.exists():
    BASE_PATH = LOCAL_DATA_DIR
elif ALT_LOCAL_DIR.exists():
    BASE_PATH = ALT_LOCAL_DIR
elif COLAB_DATA_DIR.exists():
    BASE_PATH = COLAB_DATA_DIR
else:
    BASE_PATH = LOCAL_DATA_DIR  # fallback default

# Resolve UrbanSound8K CSV location dynamically
urbansound8k_meta_path = BASE_PATH / "UrbanSound8K" / "UrbanSound8K.csv"
if not urbansound8k_meta_path.exists():
    urbansound8k_meta_path = BASE_PATH / "UrbanSound8K" / "data" / "UrbanSound8K.csv"
if not urbansound8k_meta_path.exists():
    urbansound8k_meta_path = BASE_PATH / "UrbanSound8K" / "metadata" / "UrbanSound8K.csv"

DATASET_ROOTS = {
    # Cross-lingual acoustic anchors (filename-parsed loaders)
    "cremad":       BASE_PATH / "CREMA-D" / "AudioMP3",
    "ravdess":      BASE_PATH / "Radvess",

    # Folder-per-emotion corpora (folder_emotion loader)
    "urduser":              BASE_PATH / "UrduSER A Dataset for Urdu Speech Emotion Recognition",
    "urdu_dataset_master":  BASE_PATH / "URDU-Dataset-master",
    "realworld_urdu":       BASE_PATH / "RealWorld-Urdu",
    "semour":                BASE_PATH / "SEMOUR+",
    # UrduSpeech
    "urduspeech":             BASE_PATH / "UrduSpeech",
    "urduspeech_meta":        BASE_PATH / "sheAlert_urduspeech_std_manifest.csv",

    # Ambient / environmental
    "audioset_scream":       BASE_PATH / "AudioSet_Screaming" / "wav",
    "esc50_audio":            BASE_PATH / "ESC-50" / "audio",
    "esc50_meta":             BASE_PATH / "ESC-50" / "meta" / "esc50.csv",
    "urbansound8k_audio":     BASE_PATH / "UrbanSound8K" / "wav_by_fold",
    "urbansound8k_meta":      urbansound8k_meta_path,
}

# -----------------------------------------------------------------------
# 3. FOLDER-NAME -> UNIFIED LABEL MAPS  (for the folder_emotion loader)
# -----------------------------------------------------------------------
# UrduSER: Angry, Boredom, Disgust, Fear, Happy, Neutral, Sad
URDUSER_FOLDER_MAP = {"Fear": "Distress", "Angry": "Aggression", "Neutral": "Normal"}

# URDU-Dataset-master: Angry, Happy, Neutral, Sad  (NO fear/distress class at all)
URDU_MASTER_FOLDER_MAP = {"Angry": "Aggression", "Neutral": "Normal"}

# RealWorld-Urdu (obtained from seniors' prior FYP): Angry, Happy, Neutral, Sad (also no fear class)
REALWORLD_URDU_FOLDER_MAP = {"Angry": "Aggression", "Neutral": "Normal"}

# SEMOUR+: Anger, Boredom, Disgust, Fearful, Happiness, Neutral, Sadness, Surprise
SEMOUR_FOLDER_MAP = {"Fearful": "Distress", "Anger": "Aggression", "Neutral": "Normal"}

# UrduSpeech: already pre-sorted into the exact target taxonomy — identity map
URDUSPEECH_FOLDER_MAP = {"Aggression": "Aggression", "Distress": "Distress", "Normal": "Normal"}

# -----------------------------------------------------------------------
# 4. QUALITY-CONTROL THRESHOLDS
# -----------------------------------------------------------------------
# NOTE: Section 10's "reject <3s" rule is written for the TEAM'S OWN Urdu
# recordings (FR-4.1), not for these public corpora. Several of them are
# short by design (SEMOUR+ clips average ~1-1.5s, CREMA-D ~2.2s) — applying
# a blanket 3s floor here would gut both down to near-zero for no good
# reason, since they're already vetted, published, peer-reviewed datasets.
# This stage only filters out genuinely broken/corrupt/empty files (relevant
# given the SEMOUR+ sync issue) plus hard clipping. Full 3s/SNR QC belongs
# to the separate original-recording pipeline once Module 4 data collection
# is underway — that's a different script, not this one.
MIN_DURATION_SEC = 0.3
TARGET_SR = 16000          # Hz, matches FR-4.3 / Android AudioRecord spec
MIN_SNR_DB = 3.0            # loose — this stage isn't the strict QC gate
MAX_CLASS_FRACTION = 0.40  # no class exceeds 40% of the corpus (original clips only, per Section 10)

# Clipping thresholds and exemptions
CLIPPING_THRESHOLD = 0.999
CLIPPING_MAX_FRACTION = 0.001
CLIP_EXEMPT_DATASETS = ["semour", "urduser", "urdu_dataset_master", "audioset_scream"]

# -----------------------------------------------------------------------
# 5. OUTPUT
# -----------------------------------------------------------------------
PIPELINE_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = PIPELINE_ROOT / "output"
OUTPUT_MANIFEST = OUTPUT_DIR / "unified_manifest.csv"
OUTPUT_SUMMARY = OUTPUT_DIR / "corpus_summary.txt"
OUTPUT_REJECTED_CLIPS = OUTPUT_DIR / "rejected_clips.csv"
OUTPUT_TRAINING_MANIFEST = OUTPUT_DIR / "training_manifest.csv"
OUTPUT_SPLITS = OUTPUT_DIR / "splits.csv"
OUTPUT_SPLIT_INFO = OUTPUT_DIR / "SPLIT_INFO.txt"

# -----------------------------------------------------------------------
# 6. TRAINING SPLIT & AMBIENT SAMPLING CONFIGURATION
# -----------------------------------------------------------------------
RANDOM_SEED = 42
BALANCED_SPLIT_SEED = 11
AMBIENT_SAMPLE_TOTAL = 800
AMBIENT_DATASETS = ["urbansound8k", "esc50"]
N_SPLITS = 5
OOD_DATASET = "realworld_urdu"


