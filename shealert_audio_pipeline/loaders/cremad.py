"""
CREMA-D loader.

Filename convention (fixed, documented by the dataset authors):
    ActorID_Sentence_Emotion_EmotionLevel.mp3
    e.g. 1001_DFA_ANG_XX.mp3

NOTE: ships as .mp3 in AudioMP3/ — AudioWAV/ is empty on most Drive copies
(confirmed against your dataset scan). If you convert to wav yourself,
this loader still works since it globs both extensions.

Emotion codes: ANG=Angry, DIS=Disgust, FEA=Fear, HAP=Happy, NEU=Neutral, SAD=Sad
Per FR-4.0 only the Fear and Anger subsets (~2,400 clips) are used as
cross-lingual acoustic anchors.
"""

from pathlib import Path

EMOTION_MAP = {"FEA": "Distress", "ANG": "Aggression"}  # only these two are kept


def load(root_dir: Path) -> list[dict]:
    root_dir = Path(root_dir)
    records = []
    if not root_dir.exists():
        print(f"[cremad] WARNING: path not found: {root_dir} — skipping")
        return records

    audio_files = list(root_dir.glob("*.mp3")) + list(root_dir.glob("*.wav"))
    for wav_path in audio_files:
        parts = wav_path.stem.split("_")
        if len(parts) < 3:
            continue
        actor_id, _sentence, emotion_code = parts[0], parts[1], parts[2]
        label = EMOTION_MAP.get(emotion_code)
        if label is None:
            continue  # not Fear/Anger — CREMA-D used only as an acoustic anchor for these two
        records.append({
            "filepath": str(wav_path),
            "label": label,
            "source_dataset": "cremad",
            "speaker_id": f"cremad_{actor_id}",
        })
    print(f"[cremad] loaded {len(records)} clips (Fear+Anger only)")
    return records
