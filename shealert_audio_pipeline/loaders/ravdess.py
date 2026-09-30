"""
RAVDESS loader.

Filename convention (fixed, documented by the dataset authors), 7 fields
separated by hyphens:
    Modality-VocalChannel-Emotion-Intensity-Statement-Repetition-Actor.wav
    e.g. 03-01-06-01-02-01-12.wav

Modality:      03 = audio-only          (keep)
VocalChannel:  01 = speech              (keep; skip 02=song)
Emotion:       01 neutral 02 calm 03 happy 04 sad 05 angry 06 fearful
               07 disgust 08 surprised
Actor (last field, odd/even): used as speaker id.

Per FR-4.0, only the angry/fearful audio-only speech subset (~200 clips)
is used, supplementing CREMA-D.
"""

from pathlib import Path

EMOTION_MAP = {"05": "Aggression", "06": "Distress"}  # only angry / fearful kept


def load(root_dir: Path) -> list[dict]:
    root_dir = Path(root_dir)
    records = []
    if not root_dir.exists():
        print(f"[ravdess] WARNING: path not found: {root_dir} — skipping")
        return records

    for wav_path in root_dir.rglob("*.wav"):
        fields = wav_path.stem.split("-")
        if len(fields) != 7:
            continue
        modality, vocal_channel, emotion, _intensity, _stmt, _rep, actor = fields
        if modality != "03" or vocal_channel != "01":
            continue  # keep audio-only speech; drop song and video/AV files
        label = EMOTION_MAP.get(emotion)
        if label is None:
            continue
        records.append({
            "filepath": str(wav_path),
            "label": label,
            "source_dataset": "ravdess",
            "speaker_id": f"ravdess_{actor}",
        })
    print(f"[ravdess] loaded {len(records)} clips (angry+fearful, audio-only speech)")
    return records
