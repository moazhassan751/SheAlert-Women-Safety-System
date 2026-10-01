"""
inspect_datasets.py
-------------------
Standalone dataset inspector for SheAlert FYP.
Inspects all 10 audio dataset folders under BASE_PATH:
- Exact subfolder names and casing
- File counts by extension
- 3 example file paths per dataset
- Empty or unreadable (0-byte / corrupt header) files
- UrduSpeech disk vs manifest comparison
"""

import sys
import os
from pathlib import Path
from collections import Counter
import soundfile as sf
import pandas as pd

# Set console encoding to UTF-8
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE_PATH = Path(r"F:\SheAlert-Women-Safety-System\audio data set fyp")

EXPECTED_FOLDERS = [
    "AudioSet_Screaming",
    "CREMA-D",
    "ESC-50",
    "Radvess",
    "RealWorld-Urdu",
    "SEMOUR+",
    "UrbanSound8K",
    "URDU-Dataset-master",
    "UrduSER A Dataset for Urdu Speech Emotion Recognition",
    "UrduSpeech",
]

print("=" * 80)
print("STEP 1: BASE PATH AND FOLDER VERIFICATION")
print("=" * 80)
print(f"Base Path: {BASE_PATH}")
print(f"Base Path Exists: {BASE_PATH.exists()}\n")

folder_status = {}
for name in EXPECTED_FOLDERS:
    p = BASE_PATH / name
    exists = p.exists() and p.is_dir()
    status = "FOUND" if exists else "MISSING"
    folder_status[name] = status
    print(f"[{status:7s}] {name}")

loose_files = [
    "sheAlert_urduspeech_std_manifest.csv",
    "UrbanSound8K.tar.gz"
]
print("\nLoose files:")
for lf in loose_files:
    p = BASE_PATH / lf
    if p.exists():
        size_kb = p.stat().st_size / 1024
        print(f"[FOUND  ] {lf} ({size_kb:.1f} KB)")
    else:
        print(f"[MISSING] {lf}")

print("\n" + "=" * 80)
print("STEP 2: DEEP INSPECTION OF EVERY DATASET")
print("=" * 80)

def probe_file(p: Path):
    """Check if file is empty (0 bytes) or soundfile readable."""
    try:
        size = p.stat().st_size
        if size == 0:
            return False, "0 bytes"
        # Quick header check for wav/flac
        if p.suffix.lower() in [".wav", ".flac"]:
            try:
                info = sf.info(str(p))
                if info.frames == 0:
                    return False, "0 audio frames"
            except Exception as e:
                return False, f"header corrupt: {e}"
        return True, "ok"
    except Exception as e:
        return False, f"stat/read error: {e}"

unreadable_report = {}

for name in EXPECTED_FOLDERS:
    folder = BASE_PATH / name
    print(f"\n--- DATASET: {name} ---")
    if not folder.exists():
        print("  Folder does not exist!")
        continue

    # Direct subdirectories
    subdirs = sorted([d.name for d in folder.iterdir() if d.is_dir()])
    print(f"  Direct subfolders ({len(subdirs)}): {subdirs[:15]}{' ...' if len(subdirs) > 15 else ''}")

    # Walk files
    all_files = [f for f in folder.rglob("*") if f.is_file()]
    ext_counts = Counter(f.suffix.lower() for f in all_files)
    print(f"  Total files: {len(all_files)}")
    print(f"  Counts by extension: {dict(ext_counts)}")

    # 3 Example paths
    audio_files = [f for f in all_files if f.suffix.lower() in [".wav", ".mp3", ".flac", ".m4a"]]
    examples = audio_files[:3] if audio_files else all_files[:3]
    print(f"  Example file paths ({len(examples)} shown):")
    for ex in examples:
        rel = ex.relative_to(folder)
        print(f"    - {rel}")

    # Check for empty / unreadable audio files
    unreadable = []
    for af in audio_files:
        ok, reason = probe_file(af)
        if not ok:
            unreadable.append((str(af.relative_to(folder)), reason))
    unreadable_report[name] = unreadable
    if unreadable:
        print(f"  WARNING: {len(unreadable)} unreadable/empty audio files found! (First 5: {unreadable[:5]})")
    else:
        print(f"  Audio integrity check: All {len(audio_files)} audio files are non-empty and have readable headers.")

print("\n" + "=" * 80)
print("STEP 3: URDUSPEECH MANIFEST RE-VERIFICATION")
print("=" * 80)

manifest_path = BASE_PATH / "sheAlert_urduspeech_std_manifest.csv"
if manifest_path.exists():
    df = pd.read_csv(manifest_path, encoding="utf-8")
    print(f"Manifest total rows: {len(df)}")
    print(f"Columns: {list(df.columns)}")
    print("\nsheAlert_class distribution (raw):")
    print(df["sheAlert_class"].value_counts())

    # Check exact duplicates
    dup_cols = ["category_dir", "clip_filename", "repo_audio_path"]
    exact_dups = df.duplicated(subset=dup_cols, keep=False)
    dup_count = df.duplicated(subset=dup_cols, keep="first").sum()
    print(f"\nExact duplicate rows (by {dup_cols}): {dup_count}")
    df_dedup = df.drop_duplicates(subset=dup_cols, keep="first")
    print(f"Rows after deduplication: {len(df_dedup)}")

    # Speaker ID extraction
    # Pattern: SPEAKER_XXXX_...
    speaker_extracted = df_dedup["clip_filename"].str.extract(r"^(SPEAKER_\d+)_")[0]
    print(f"Extracted unique speakers: {speaker_extracted.nunique()} (nulls: {speaker_extracted.isna().sum()})")

    # Transcription unique count
    print(f"Unique transcriptions: {df['transcription'].nunique()} / {len(df)}")

    # Check paths on disk
    urduspeech_dir = BASE_PATH / "UrduSpeech"
    disk_files = {f.name: f for f in urduspeech_dir.rglob("*.wav")}
    print(f"\nUrduSpeech files actually on disk: {len(disk_files)}")
    subfolders_disk = Counter(f.parent.name for f in disk_files.values())
    print(f"Disk files by folder name: {dict(subfolders_disk)}")

    # Cross reference deduplicated manifest with disk files
    found_on_disk = sum(1 for fname in df_dedup["clip_filename"] if fname in disk_files)
    missing_from_disk = [fname for fname in df_dedup["clip_filename"] if fname not in disk_files]
    print(f"Deduplicated manifest clips found on disk: {found_on_disk} / {len(df_dedup)}")
    print(f"Clips missing from disk: {len(missing_from_disk)}")

    # Spot checks: 10 Distress, 10 Aggression
    print("\n--- SPOT CHECK: 10 Distress clips ---")
    distress_sample = df_dedup[df_dedup["sheAlert_class"] == "Distress"].sample(10, random_state=42)
    for _, r in distress_sample.iterrows():
        disk_path = disk_files.get(r["clip_filename"], "NOT FOUND ON DISK")
        print(f"  Category: {r['category_dir']} | File: {r['clip_filename']}")
        print(f"    Emotion: {r['emotion_raw']}")
        print(f"    Disk Path: {disk_path}")

    print("\n--- SPOT CHECK: 10 Aggression clips ---")
    agg_sample = df_dedup[df_dedup["sheAlert_class"] == "Aggression"].sample(10, random_state=42)
    for _, r in agg_sample.iterrows():
        disk_path = disk_files.get(r["clip_filename"], "NOT FOUND ON DISK")
        print(f"  Category: {r['category_dir']} | File: {r['clip_filename']}")
        print(f"    Emotion: {r['emotion_raw']}")
        print(f"    Disk Path: {disk_path}")
else:
    print("sheAlert_urduspeech_std_manifest.csv NOT FOUND!")

print("\nInspection complete.")
