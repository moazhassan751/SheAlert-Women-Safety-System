"""
AudioSet scream-subset loader.

AudioSet itself ships as YouTube segment references (id, start_sec, end_sec),
not audio files, so extraction (yt-dlp + ffmpeg trim on the "Screaming"
ontology class, /m/03qc9zr) is assumed to already be done and the ~600-800
resulting clips dropped into a flat folder of wav files. Point
DATASET_ROOTS["audioset_scream"] at that folder.

If you haven't extracted it yet and want a script for that step
specifically, ask and I'll write the yt-dlp/ffmpeg extraction script —
it's a separate concern from consolidation and worth keeping isolated
since it depends on network access and can partially fail per-clip.
"""

from pathlib import Path


def load(root_dir: Path) -> list[dict]:
    root_dir = Path(root_dir)
    records = []
    if not root_dir.exists():
        print(f"[audioset_scream] WARNING: path not found: {root_dir} — skipping")
        return records

    for wav_path in root_dir.glob("*.wav"):
        records.append({
            "filepath": str(wav_path),
            "label": "Distress",
            "source_dataset": "audioset_scream",
            "speaker_id": None,  # language-agnostic screams, no speaker identity tracked
        })
    print(f"[audioset_scream] loaded {len(records)} clips (all Distress)")
    return records
