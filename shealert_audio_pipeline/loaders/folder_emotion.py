"""
Generic folder-per-emotion loader.

Covers UrduSER, URDU-Dataset-master, RealWorld-Urdu, SEMOUR+, and
UrduSpeech — all five ship as recursive directory trees where a file's
IMMEDIATE PARENT FOLDER NAME is its emotion label:

    UrduSER:               root/Fear/*.wav, root/Angry/*.wav, ...
    URDU-Dataset-master:    root/Angry/*.wav, root/Neutral/*.wav, ...
    RealWorld-Urdu:         root/Angry/*.wav, root/Neutral/*.wav, ...
    SEMOUR+:                root/Actor_N/Fearful/*.wav, root/Actor_N/Anger/*.wav, ...
                            (parent = emotion, grandparent = actor — we only
                            need the parent, so actor nesting doesn't matter)
    UrduSpeech:             root/Aggression/*.wav, root/Distress/*.wav, root/Normal/*.wav
                            (already pre-sorted into the target taxonomy)

One loader handles all five because the only thing that differs between
them is which folder names map to which of the 3 unified labels — that's
supplied as `folder_map` from config.py.
"""

from pathlib import Path

AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".m4a"}


def load(root_dir: Path, folder_map: dict, source_name: str) -> list[dict]:
    root_dir = Path(root_dir)
    records = []
    if not root_dir.exists():
        print(f"[{source_name}] WARNING: path not found: {root_dir} — skipping")
        return records

    skipped_unmapped = 0
    for audio_path in root_dir.rglob("*"):
        if audio_path.suffix.lower() not in AUDIO_EXTENSIONS:
            continue
        emotion_folder = audio_path.parent.name
        label = folder_map.get(emotion_folder)
        if label is None:
            skipped_unmapped += 1
            continue  # e.g. Happy/Sad/Disgust/Boredom/Surprise — not in the 3-class taxonomy

        # Speaker/actor id: grandparent folder if it looks like an actor dir
        # (e.g. SEMOUR+'s Actor_N), else None — harmless either way, just
        # useful for the eventual speaker-disjoint split.
        grandparent = audio_path.parent.parent.name
        speaker_id = f"{source_name}_{grandparent}" if grandparent.lower().startswith("actor") else None

        records.append({
            "filepath": str(audio_path),
            "label": label,
            "source_dataset": source_name,
            "speaker_id": speaker_id,
        })

    print(f"[{source_name}] loaded {len(records)} clips "
          f"({skipped_unmapped} skipped — folder not in taxonomy)")
    return records
