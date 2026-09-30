"""
ESC-50 loader.

Metadata is a single CSV (meta/esc50.csv) with columns:
    filename, fold, target, category, esc10, src_file, take

Per the scope doc, ESC-50 supplements the Normal class with urban/ambient
backgrounds. All 50 categories are environmental/non-emotional-speech, so
every clip maps to Normal — no filtering needed.
"""

from pathlib import Path
import pandas as pd


def load(audio_dir: Path, meta_csv: Path) -> list[dict]:
    audio_dir, meta_csv = Path(audio_dir), Path(meta_csv)
    records = []
    if not meta_csv.exists():
        print(f"[esc50] WARNING: metadata not found: {meta_csv} — skipping")
        return records

    df = pd.read_csv(meta_csv)
    for _, row in df.iterrows():
        wav_path = audio_dir / row["filename"]
        if not wav_path.exists():
            continue
        records.append({
            "filepath": str(wav_path),
            "label": "Normal",
            "source_dataset": "esc50",
            "speaker_id": None,  # not applicable — environmental audio, not speech
        })
    print(f"[esc50] loaded {len(records)} clips (all mapped to Normal)")
    return records
