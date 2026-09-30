"""
consolidate.py
---------------
Runs every dataset loader against the real folder structure from your
dataset_scan.json, applies a light QC gate (corrupt/empty + hard-clipping
only — see config.py's note on why the 3s/SNR floor doesn't apply here),
and writes one unified manifest CSV covering all external corpora.

Usage (run from Colab with Drive mounted, or locally with BASE_PATH changed):
    python consolidate.py                # manifest only, light QC applied
    python consolidate.py --no-qc        # skip QC entirely (fastest first look)
    python consolidate.py --resample     # also write 16kHz-mono copies to output/audio_16k/

This does NOT touch your original Urdu recordings (Module 4's Stage 2
data) — those get their own manifest once recording + labelling is done,
kept separate on purpose so Stage 1 / Stage 2 stay cleanly split per
FR-4.4.
"""

import argparse
from collections import Counter
from pathlib import Path

import pandas as pd
import soundfile as sf
import librosa

import config
from utils import passes_qc
from loaders import cremad, ravdess, esc50, urbansound8k, audioset_scream, folder_emotion


def collect_all_records() -> list[dict]:
    records = []

    # Filename-parsed (fixed public conventions)
    records += cremad.load(config.DATASET_ROOTS["cremad"])
    records += ravdess.load(config.DATASET_ROOTS["ravdess"])

    # Folder-per-emotion corpora
    records += folder_emotion.load(
        config.DATASET_ROOTS["urduser"], config.URDUSER_FOLDER_MAP, "urduser"
    )
    records += folder_emotion.load(
        config.DATASET_ROOTS["urdu_dataset_master"], config.URDU_MASTER_FOLDER_MAP, "urdu_dataset_master"
    )
    records += folder_emotion.load(
        config.DATASET_ROOTS["realworld_urdu"], config.REALWORLD_URDU_FOLDER_MAP, "realworld_urdu"
    )
    records += folder_emotion.load(
        config.DATASET_ROOTS["semour"], config.SEMOUR_FOLDER_MAP, "semour"
    )
    records += folder_emotion.load(
        config.DATASET_ROOTS["urduspeech"], config.URDUSPEECH_FOLDER_MAP, "urduspeech"
    )

    # Ambient / environmental
    records += audioset_scream.load(config.DATASET_ROOTS["audioset_scream"])
    records += esc50.load(config.DATASET_ROOTS["esc50_audio"], config.DATASET_ROOTS["esc50_meta"])
    records += urbansound8k.load(
        config.DATASET_ROOTS["urbansound8k_audio"], config.DATASET_ROOTS["urbansound8k_meta"]
    )

    return records


def apply_qc(records: list[dict]) -> tuple[list[dict], Counter]:
    passed, rejection_reasons = [], Counter()
    for rec in records:
        ok, reason = passes_qc(rec["filepath"], config.MIN_DURATION_SEC, config.MIN_SNR_DB)
        if ok:
            passed.append(rec)
        else:
            rejection_reasons[reason.split(" ")[0].split(":")[0]] += 1  # bucket by reason type
    return passed, rejection_reasons


def resample_copy(records: list[dict], out_root: Path) -> None:
    """Write a 16kHz-mono copy of every record, organized as out_root/<label>/<source>_<n>.wav"""
    out_root.mkdir(parents=True, exist_ok=True)
    counters = Counter()
    failed = 0
    for rec in records:
        label_dir = out_root / rec["label"]
        label_dir.mkdir(exist_ok=True)
        counters[rec["label"]] += 1
        dest = label_dir / f"{rec['source_dataset']}_{counters[rec['label']]:06d}.wav"
        try:
            y, sr = librosa.load(rec["filepath"], sr=config.TARGET_SR, mono=True)
            sf.write(str(dest), y, config.TARGET_SR)
            rec["resampled_path"] = str(dest)
        except Exception as e:
            failed += 1
            rec["resampled_path"] = None
    if failed:
        print(f"  ({failed} files failed to resample and were left out of output/audio_16k/)")


def write_summary(records: list[dict], rejected_count: int, rejection_reasons: Counter, path: Path) -> None:
    label_counts = Counter(r["label"] for r in records)
    source_counts = Counter(r["source_dataset"] for r in records)
    total = len(records)

    lines = [
        "SheAlert — External Corpus Consolidation Summary",
        "=" * 50,
        f"Total clips retained: {total}",
        f"Total clips rejected by QC: {rejected_count}",
        "",
        "Rejection reasons:",
    ]
    for reason, count in rejection_reasons.most_common():
        lines.append(f"  {reason}: {count}")

    lines += ["", "Class balance:"]
    for label in config.LABELS:
        count = label_counts.get(label, 0)
        fraction = count / total if total else 0
        flag = " <-- exceeds 40% cap (Section 10 rule — matters more once original clips are added)" \
            if fraction > config.MAX_CLASS_FRACTION else ""
        lines.append(f"  {label}: {count} ({fraction:.1%}){flag}")

    lines += ["", "By source dataset:"]
    for source, count in source_counts.most_common():
        lines.append(f"  {source}: {count}")

    summary_text = "\n".join(lines)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(summary_text)
    print("\n" + summary_text)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-qc", action="store_true", help="skip the QC gate entirely")
    parser.add_argument("--resample", action="store_true", help="also write 16kHz mono copies")
    args = parser.parse_args()

    print("Collecting records from all loaders...\n")
    records = collect_all_records()
    print(f"\nTotal collected (pre-QC): {len(records)}")

    rejected_count, rejection_reasons = 0, Counter()
    if not args.no_qc:
        print("\nApplying light QC gate (corrupt/empty + hard clipping)...")
        before = len(records)
        records, rejection_reasons = apply_qc(records)
        rejected_count = before - len(records)

    if args.resample:
        print("\nResampling to 16kHz mono...")
        resample_copy(records, Path("output/audio_16k"))

    config.OUTPUT_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(config.OUTPUT_MANIFEST, index=False)
    print(f"\nManifest written to {config.OUTPUT_MANIFEST}")

    write_summary(records, rejected_count, rejection_reasons, config.OUTPUT_SUMMARY)


if __name__ == "__main__":
    main()
