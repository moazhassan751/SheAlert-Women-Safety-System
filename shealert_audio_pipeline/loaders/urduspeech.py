"""
UrduSpeech manifest-driven loader.
Uses sheAlert_urduspeech_std_manifest.csv as the single source of truth for
UrduSpeech clips, labels, and speakers.
"""

from pathlib import Path
import re
import pandas as pd


def load(audio_root_dir: Path, manifest_csv: Path) -> tuple[list[dict], dict]:
    audio_root_dir, manifest_csv = Path(audio_root_dir), Path(manifest_csv)
    records = []
    stats = {
        "raw_rows": 0,
        "exact_duplicates_dropped": 0,
        "missing_on_disk": 0,
        "loaded": 0,
    }

    if not manifest_csv.exists():
        print(f"[urduspeech] WARNING: manifest not found: {manifest_csv} — skipping")
        return records, stats

    df = pd.read_csv(manifest_csv, encoding="utf-8")
    stats["raw_rows"] = len(df)

    # Deduplicate exact duplicate rows
    dup_cols = ["category_dir", "clip_filename", "repo_audio_path"]
    dup_mask = df.duplicated(subset=dup_cols, keep="first")
    stats["exact_duplicates_dropped"] = int(dup_mask.sum())
    df_dedup = df[~dup_mask].copy()

    # Build filename to path index on disk for fast lookup
    disk_index = {p.name: p for p in audio_root_dir.rglob("*.wav")}

    for _, row in df_dedup.iterrows():
        fname = str(row["clip_filename"]).strip()
        label = str(row["sheAlert_class"]).strip()

        # Look up on disk: check expected folder first, then index
        target_path = audio_root_dir / label / fname
        if not target_path.exists():
            target_path = disk_index.get(fname)

        if target_path is None or not target_path.exists():
            stats["missing_on_disk"] += 1
            continue

        # Extract speaker id: e.g. SPEAKER_0001 from SPEAKER_0001_COMEDY_SHOW_000006.wav
        match = re.match(r"^(SPEAKER_\d+)_", fname)
        speaker_id = f"urduspeech_{match.group(1)}" if match else None

        records.append({
            "filepath": str(target_path.resolve()),
            "label": label,
            "source_dataset": "urduspeech",
            "speaker_id": speaker_id,
            "duration": None,
        })

    stats["loaded"] = len(records)
    print(f"[urduspeech] loaded {len(records)} clips from manifest "
          f"({stats['exact_duplicates_dropped']} exact duplicates dropped, "
          f"{stats['missing_on_disk']} missing from disk)")
    return records, stats
