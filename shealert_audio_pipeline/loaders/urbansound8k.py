"""
UrbanSound8K loader.

Metadata: metadata/UrbanSound8K.csv, columns include:
    slice_file_name, fold, classID, class

Audio files live under audio/fold{N}/{slice_file_name}.

Per the scope doc, UrbanSound8K supplements the Normal class with urban
ambient backgrounds. "gun_shot" is excluded here — it is not a benign
ambient sound and risks contaminating the Normal class semantically;
everything else maps to Normal. Remove the exclusion below if you'd
rather keep gun_shot in (e.g. as a hard-negative Normal example).
"""

from pathlib import Path
import pandas as pd

EXCLUDED_CLASSES = {"gun_shot"}


def load(audio_dir: Path, meta_csv: Path) -> list[dict]:
    audio_dir, meta_csv = Path(audio_dir), Path(meta_csv)
    records = []
    if not meta_csv.exists():
        candidates = [
            meta_csv.parent / "UrbanSound8K.csv",
            audio_dir.parent / "UrbanSound8K.csv",
            audio_dir.parent / "data" / "UrbanSound8K.csv",
            audio_dir.parent / "metadata" / "UrbanSound8K.csv",
        ]
        for c in candidates:
            if c.exists():
                meta_csv = c
                break
        else:
            print(f"[urbansound8k] WARNING: metadata not found: {meta_csv} — skipping")
            return records

    df = pd.read_csv(meta_csv)
    for _, row in df.iterrows():
        if row["class"] in EXCLUDED_CLASSES:
            continue
        wav_path = audio_dir / f"fold{row['fold']}" / row["slice_file_name"]
        if not wav_path.exists():
            continue
        records.append({
            "filepath": str(wav_path),
            "label": "Normal",
            "source_dataset": "urbansound8k",
            "speaker_id": None,
        })
    excluded_note = f", excluded {EXCLUDED_CLASSES}" if EXCLUDED_CLASSES else ""
    print(f"[urbansound8k] loaded {len(records)} clips (mapped to Normal{excluded_note})")
    return records
