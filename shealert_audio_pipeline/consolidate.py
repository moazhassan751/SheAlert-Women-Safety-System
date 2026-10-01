"""
consolidate.py
---------------
Runs every dataset loader against the real folder structure from your
dataset scan, applies deduplication by resolved file path, applies a light QC
gate (corrupt/empty + hard-clipping + duration + SNR floor), records rejection
details, and writes:
  1. output/unified_manifest.csv (all retained clips, relative paths, group_id)
  2. output/rejected_clips.csv (all rejected clips with standardized reason)
  3. output/training_manifest.csv (curated clips with 800 sampled ambient)
  4. output/splits.csv (5-fold stratified group split + out-of-distribution set)
  5. output/corpus_summary.txt (comprehensive statistics report)

Usage:
    python consolidate.py                # manifest with light QC applied
    python consolidate.py --no-qc        # skip waveform QC (fastest run)
    python consolidate.py --resample     # also write 16kHz-mono copies to output/audio_16k/
"""

import argparse
import hashlib
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import librosa
from analysis.find_balanced_split import assign_balanced_split, build_group_table

import config
from utils import (
    passes_qc,
    get_duration_sec,
    to_relative_path,
    resolve_relative_path,
    get_rejection_category,
    extract_group_id,
)
from loaders import (
    cremad,
    ravdess,
    esc50,
    urbansound8k,
    audioset_scream,
    folder_emotion,
    urduspeech,
)


def collect_all_records() -> tuple[list[dict], dict]:
    records = []
    dropped_stats = {}

    # 1. Filename-parsed loaders
    records += cremad.load(config.DATASET_ROOTS["cremad"])
    records += ravdess.load(config.DATASET_ROOTS["ravdess"])

    # 2. Folder-per-emotion corpora
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

    # 3. UrduSpeech (Manifest-driven, single source of truth)
    u_records, u_stats = urduspeech.load(
        config.DATASET_ROOTS["urduspeech"], config.DATASET_ROOTS["urduspeech_meta"]
    )
    records += u_records
    dropped_stats["urduspeech_manifest_exact_duplicates"] = u_stats.get("exact_duplicates_dropped", 0)

    # 4. Ambient / environmental
    records += audioset_scream.load(config.DATASET_ROOTS["audioset_scream"])
    records += esc50.load(config.DATASET_ROOTS["esc50_audio"], config.DATASET_ROOTS["esc50_meta"])
    records += urbansound8k.load(
        config.DATASET_ROOTS["urbansound8k_audio"], config.DATASET_ROOTS["urbansound8k_meta"]
    )

    return records, dropped_stats


def deduplicate_by_path(records: list[dict]) -> tuple[list[dict], Counter]:
    """Deduplicate records by real resolved absolute file path."""
    seen_paths = set()
    unique_records = []
    dropped_by_source = Counter()

    for rec in records:
        resolved_path = str(Path(rec["filepath"]).resolve())
        if resolved_path in seen_paths:
            dropped_by_source[rec["source_dataset"]] += 1
        else:
            seen_paths.add(resolved_path)
            rec["filepath"] = resolved_path
            unique_records.append(rec)

    return unique_records, dropped_by_source


def _check_record_qc(rec: dict) -> tuple[dict, bool, str, float, float]:
    ok, reason, duration, clip_frac = passes_qc(
        rec["filepath"],
        config.MIN_DURATION_SEC,
        config.MIN_SNR_DB,
        return_duration=True,
        source_dataset=rec["source_dataset"],
        return_clip_frac=True,
    )
    return rec, ok, reason, duration, clip_frac


def apply_qc(records: list[dict]) -> tuple[list[dict], list[dict], Counter, Counter]:
    passed = []
    rejected_records = []
    rejection_reasons = Counter()
    rejections_by_source = Counter()

    print(f"Applying light QC gate to {len(records)} records (multi-threaded)...")
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(_check_record_qc, r) for r in records]
        for i, future in enumerate(as_completed(futures), 1):
            if i % 5000 == 0 or i == len(records):
                print(f"  Processed {i}/{len(records)} clips...")
            rec, ok, reason, duration, clip_frac = future.result()
            rec["duration"] = round(float(duration), 3)
            rec["clip_frac"] = round(float(clip_frac), 6)
            if ok:
                passed.append(rec)
            else:
                cat = get_rejection_category(reason)
                rejection_reasons[cat] += 1
                rejections_by_source[rec["source_dataset"]] += 1
                rejected_records.append({
                    "filepath": to_relative_path(rec["filepath"]),
                    "source_dataset": rec["source_dataset"],
                    "reason": cat,
                })

    return passed, rejected_records, rejection_reasons, rejections_by_source


def sample_ambient_clips(
    records: list[dict],
    target_total: int = 800,
    random_seed: int = 42,
) -> list[dict]:
    """
    Subsamples ambient datasets (UrbanSound8K and ESC-50) proportionally to their
    retained size, while spreading the sample evenly across original sound classes.
    Deterministic: sorts within each sound class by filepath before calling sample(),
    and breaks class ties by class name.
    """
    us8k_recs = sorted([r for r in records if r["source_dataset"] == "urbansound8k"], key=lambda r: str(r["filepath"]))
    esc50_recs = sorted([r for r in records if r["source_dataset"] == "esc50"], key=lambda r: str(r["filepath"]))
    total_ambient = len(us8k_recs) + len(esc50_recs)
    if total_ambient == 0:
        return []

    # Proportional targets
    n_us8k = round(target_total * len(us8k_recs) / total_ambient)
    n_esc50 = target_total - n_us8k

    def _sample_stratified_by_class(recs, n_target, class_extractor, seed):
        df_sub = pd.DataFrame(recs)
        df_sub["_sound_class"] = df_sub["filepath"].apply(class_extractor)
        
        # Count per sound class, sort by (-count, _sound_class) for deterministic tie-breaking
        counts = df_sub["_sound_class"].value_counts()
        class_df = pd.DataFrame({"_sound_class": counts.index, "count": counts.values})
        class_df = class_df.sort_values(by=["count", "_sound_class"], ascending=[False, True]).reset_index(drop=True)
        
        targets = (class_df["count"] / len(df_sub) * n_target).round().astype(int)
        class_df["target"] = targets
        diff = n_target - class_df["target"].sum()
        if diff != 0:
            for idx in class_df.index[:abs(diff)]:
                class_df.loc[idx, "target"] += 1 if diff > 0 else -1

        target_map = dict(zip(class_df["_sound_class"], class_df["target"]))
        
        sampled = []
        for cls in sorted(target_map.keys()):
            cnt = target_map[cls]
            sub = df_sub[df_sub["_sound_class"] == cls].copy()
            # Sort by filepath before calling sample() to guarantee deterministic index order
            sub = sub.sort_values("filepath", ascending=True).reset_index(drop=True)
            sampled.append(sub.sample(n=min(cnt, len(sub)), random_state=seed))
            
        res_df = pd.concat(sampled, ignore_index=True).drop(columns=["_sound_class"])
        res_df = res_df.sort_values("filepath", ascending=True).reset_index(drop=True)
        return res_df.to_dict(orient="records")

    # UrbanSound8K: slice_file_name e.g. 101415-3-0-2.wav -> field 1 is classID
    sampled_us8k = _sample_stratified_by_class(
        us8k_recs,
        n_us8k,
        lambda p: Path(p).stem.split("-")[1] if len(Path(p).stem.split("-")) >= 2 else "unknown",
        random_seed,
    )

    # ESC-50: 1-100032-A-0.wav -> field 3 is category target
    sampled_esc50 = _sample_stratified_by_class(
        esc50_recs,
        n_esc50,
        lambda p: Path(p).stem.split("-")[3] if len(Path(p).stem.split("-")) >= 4 else "unknown",
        random_seed,
    )

    combined = sampled_us8k + sampled_esc50
    combined.sort(key=lambda r: str(r["filepath"]))
    return combined


def create_splits(
    training_df: pd.DataFrame,
    n_splits: int = 5,
    random_seed: int = 42,
    ood_dataset: str = "realworld_urdu",
) -> pd.DataFrame:
    """
    Creates balanced group splits for training_manifest:
    - OOD dataset (realworld_urdu) gets fold = 'ood'
    - All other rows get fold 0 to n_splits - 1 via greedy balanced assignment
      (smallest-bin-first + local search) grouped by group_id.
    """
    splits_df = training_df[["filepath", "label", "source_dataset", "group_id"]].copy()
    splits_df["fold"] = None

    is_ood = splits_df["source_dataset"] == ood_dataset
    splits_df.loc[is_ood, "fold"] = "ood"

    # Build group table from non-OOD rows
    non_ood = splits_df[~is_ood].copy()
    group_rows = []
    for gid, sub in non_ood.groupby("group_id"):
        src = sub["source_dataset"].iloc[0]
        total = len(sub)
        d = int((sub["label"] == "Distress").sum())
        a = int((sub["label"] == "Aggression").sum())
        n = int((sub["label"] == "Normal").sum())
        group_rows.append({"group_id": gid, "source_dataset": src,
                           "total": total, "Distress": d, "Aggression": a, "Normal": n})
    gt = pd.DataFrame(group_rows)

    # Run balanced assignment
    assignment, _, _, _, _ = assign_balanced_split(gt, random_seed, n_splits)

    # Map group_id -> fold for all non-OOD rows
    for gid, fold_idx in assignment.items():
        splits_df.loc[splits_df["group_id"] == gid, "fold"] = str(fold_idx)

    splits_df = splits_df.sort_values("filepath", ascending=True).reset_index(drop=True)
    return splits_df


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
            abs_p = resolve_relative_path(rec["filepath"])
            y, sr = librosa.load(str(abs_p), sr=config.TARGET_SR, mono=True)
            sf.write(str(dest), y, config.TARGET_SR)
            rec["resampled_path"] = str(dest)
        except Exception:
            failed += 1
            rec["resampled_path"] = None
    if failed:
        print(f"  ({failed} files failed to resample and were left out of output/audio_16k/)")


def compute_clip_frac_summary(records: list[dict]) -> str:
    df = pd.DataFrame(records)
    lines = ["=== CLIP_FRAC SUMMARY PER DATASET (RETAINED CLIPS) ==="]
    lines.append(f"{'dataset':20s} | {'clips':6s} | {'median':8s} | {'p99':8s} | {'clips > 0.001':13s}")
    lines.append("-" * 65)
    for src in sorted(df["source_dataset"].unique()):
        sub = df[df["source_dataset"] == src]
        fracs = sub["clip_frac"].values
        med = float(np.median(fracs))
        p99 = float(np.percentile(fracs, 99))
        n_gt_001 = int(np.sum(fracs > 0.001))
        lines.append(f"{src:20s} | {len(sub):6d} | {med:8.6f} | {p99:8.6f} | {n_gt_001:13d}")
    return "\n".join(lines)


def compare_with_backup(new_manifest_df: pd.DataFrame, new_splits_df: pd.DataFrame) -> str:
    backup_dir = config.PIPELINE_ROOT / "output_backup_before_clip_change"
    lines = ["=== BEFORE VS AFTER COMPARISON (AGAINST BACKUP) ==="]
    backup_manifest_path = backup_dir / "unified_manifest.csv"
    backup_splits_path = backup_dir / "splits.csv"

    if backup_manifest_path.exists():
        old_m_df = pd.read_csv(backup_manifest_path)
        old_counts = old_m_df["source_dataset"].value_counts().to_dict()
        new_counts = new_manifest_df["source_dataset"].value_counts().to_dict()
        all_srcs = sorted(set(list(old_counts.keys()) + list(new_counts.keys())))

        lines.append("\nRetained Clips Count per Dataset (Before vs After):")
        lines.append(f"{'dataset':20s} | {'Before':7s} | {'After':7s} | {'Diff':7s}")
        lines.append("-" * 50)
        for s in all_srcs:
            b = old_counts.get(s, 0)
            a = new_counts.get(s, 0)
            diff = a - b
            diff_str = f"+{diff}" if diff > 0 else str(diff)
            lines.append(f"{s:20s} | {b:7d} | {a:7d} | {diff_str:7s}")
        lines.append("-" * 50)
        tot_b = len(old_m_df)
        tot_a = len(new_manifest_df)
        tot_diff = tot_a - tot_b
        lines.append(f"{'TOTAL':20s} | {tot_b:7d} | {tot_a:7d} | {'+' + str(tot_diff) if tot_diff > 0 else str(tot_diff):7s}")

    if backup_splits_path.exists():
        old_s_df = pd.read_csv(backup_splits_path)
        merged = pd.merge(old_s_df, new_splits_df, on="filepath", suffixes=("_old", "_new"))
        same_fold = int((merged["fold_old"].astype(str) == merged["fold_new"].astype(str)).sum())
        diff_fold = int((merged["fold_old"].astype(str) != merged["fold_new"].astype(str)).sum())
        new_rows = len(new_splits_df) - len(merged)
        lines.append("\nSplit Fold Migration Analysis (splits.csv):")
        lines.append(f"  Total rows in backup splits: {len(old_s_df)}")
        lines.append(f"  Total rows in new splits   : {len(new_splits_df)}")
        lines.append(f"  Common rows analyzed       : {len(merged)}")
        lines.append(f"  Unchanged fold assignments : {same_fold}")
        lines.append(f"  Changed fold assignments   : {diff_fold}")
        lines.append(f"  Newly added rows in splits : {new_rows}")

    return "\n".join(lines)


def write_summary(
    records: list[dict],
    pre_dedup_count: int,
    duplicates_dropped: Counter,
    rejected_count: int,
    rejection_reasons: Counter,
    rejections_by_source: Counter,
    rejection_table_str: str,
    group_report_str: str,
    training_summary_str: str,
    split_summary_str: str,
    clip_frac_summary_str: str,
    comparison_str: str,
    path: Path,
) -> None:
    label_counts = Counter(r["label"] for r in records)
    source_counts = Counter(r["source_dataset"] for r in records)
    total = len(records)

    lines = [
        "SheAlert — External Corpus Consolidation Summary",
        "=" * 60,
        f"Total clips pre-deduplication: {pre_dedup_count}",
        f"Duplicates dropped by resolved path: {sum(duplicates_dropped.values())}",
    ]
    for src, cnt in duplicates_dropped.items():
        lines.append(f"  - {src}: {cnt}")

    lines += [
        "",
        f"Total clips retained after QC: {total}",
        f"Total clips rejected by QC: {rejected_count}",
        "",
        "Rejection reasons breakdown:",
    ]
    for reason, count in rejection_reasons.most_common():
        lines.append(f"  {reason}: {count}")

    if rejections_by_source:
        lines += ["", "Rejections by source dataset:"]
        for src, count in rejections_by_source.most_common():
            lines.append(f"  {src}: {count}")

    lines += [
        "",
        "Rejection Crosstab (source_dataset x reason):",
        rejection_table_str,
        "",
        "Group ID Report (Speaker / Grouping):",
        group_report_str,
        "",
        "Class balance (Overall Full Corpus):",
    ]
    for label in config.LABELS:
        count = label_counts.get(label, 0)
        fraction = count / total if total else 0
        flag = " <-- exceeds 40% cap (Section 10 rule — matters more once original clips are added)" \
            if fraction > config.MAX_CLASS_FRACTION else ""
        lines.append(f"  {label}: {count} ({fraction:.1%}){flag}")

    lines += ["", "By source dataset (Retained clips):"]
    for source, count in source_counts.most_common():
        lines.append(f"  {source:20s}: {count}")

    lines += ["", "Class distribution per dataset:"]
    by_src_label = {}
    for r in records:
        by_src_label.setdefault(r["source_dataset"], Counter())[r["label"]] += 1
    for src in sorted(by_src_label.keys()):
        cnts = by_src_label[src]
        dist = ", ".join(f"{lbl}: {cnts.get(lbl, 0)}" for lbl in config.LABELS)
        lines.append(f"  {src:20s} -> {dist}")

    lines += [
        "",
        "Training Manifest & Splits Summary:",
        training_summary_str,
        "",
        split_summary_str,
        "",
        clip_frac_summary_str,
        "",
        comparison_str,
    ]

    summary_text = "\n".join(lines)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(summary_text, encoding="utf-8")
    print("\n" + summary_text)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-qc", action="store_true", help="skip the QC gate entirely")
    parser.add_argument("--resample", action="store_true", help="also write 16kHz mono copies")
    args = parser.parse_args()

    print("Collecting records from all loaders...\n")
    records, dropped_stats = collect_all_records()
    pre_dedup_count = len(records)
    print(f"\nTotal collected (pre-deduplication): {pre_dedup_count}")

    # Deduplicate by real resolved path
    records, duplicates_dropped = deduplicate_by_path(records)
    print(f"Total after path deduplication: {len(records)} (dropped {sum(duplicates_dropped.values())})")

    rejected_count = 0
    rejection_reasons = Counter()
    rejections_by_source = Counter()
    rejected_records = []
    rejection_table_str = ""

    if not args.no_qc:
        print("\nApplying light QC gate (duration, clipping, SNR)...")
        before_qc = len(records)
        records, rejected_records, rejection_reasons, rejections_by_source = apply_qc(records)
        rejected_count = before_qc - len(records)

        # Sort retained and rejected records by filepath (ascending, plain string order)
        records.sort(key=lambda r: str(r["filepath"]))
        rejected_records.sort(key=lambda r: str(r["filepath"]))

        # Write rejected_clips.csv
        df_rej = pd.DataFrame(rejected_records, columns=["filepath", "source_dataset", "reason"])
        df_rej = df_rej.sort_values("filepath", ascending=True).reset_index(drop=True)
        config.OUTPUT_REJECTED_CLIPS.parent.mkdir(parents=True, exist_ok=True)
        df_rej.to_csv(config.OUTPUT_REJECTED_CLIPS, index=False, encoding="utf-8")
        print(f"Rejected clips written to {config.OUTPUT_REJECTED_CLIPS} ({len(df_rej)} rows)")

        # Generate crosstab table
        crosstab_df = pd.crosstab(df_rej["source_dataset"], df_rej["reason"], margins=True)
        rejection_table_str = crosstab_df.to_string()
        print("\n=== REJECTION TABLE (source_dataset x reason) ===")
        print(rejection_table_str)
    else:
        print("\nPopulating audio duration (QC skipped)...")
        for rec in records:
            if rec.get("duration") is None:
                rec["duration"] = round(get_duration_sec(rec["filepath"]), 3)

    # STEP 2 & 3: Assign group_id, record fallbacks, convert to relative paths
    fallback_counts = Counter()
    distinct_groups_per_dataset = {}
    for rec in records:
        group_id, is_fallback = extract_group_id(rec["filepath"], rec["source_dataset"])
        rec["group_id"] = group_id
        if is_fallback:
            fallback_counts[rec["source_dataset"]] += 1
        rec["filepath"] = to_relative_path(rec["filepath"])

    # Ensure records are sorted by filepath (relative string order)
    records.sort(key=lambda r: str(r["filepath"]))

    # Group ID reporting
    group_report_lines = []
    df_temp = pd.DataFrame(records)
    for src in sorted(df_temp["source_dataset"].unique()):
        sub = df_temp[df_temp["source_dataset"] == src]
        n_groups = sub["group_id"].nunique()
        fb = fallback_counts.get(src, 0)
        distinct_groups_per_dataset[src] = n_groups
        group_report_lines.append(f"  {src:20s}: {n_groups:5d} distinct groups | per-file fallback rows: {fb}")
    group_report_str = "\n".join(group_report_lines)
    print("\n=== GROUP ID REPORT ===")
    print(group_report_str)

    if args.resample:
        print("\nResampling to 16kHz mono...")
        resample_copy(records, config.OUTPUT_DIR / "audio_16k")

    # STEP 2: Save full unified manifest (relative paths, with group_id, clip_frac)
    columns = ["filepath", "label", "source_dataset", "duration", "speaker_id", "group_id", "clip_frac"]
    df = pd.DataFrame(records)
    for col in columns:
        if col not in df.columns:
            df[col] = None
    df = df[columns].sort_values("filepath", ascending=True).reset_index(drop=True)

    config.OUTPUT_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.OUTPUT_MANIFEST, index=False, encoding="utf-8")
    print(f"\nUnified manifest written to {config.OUTPUT_MANIFEST} with {len(df)} rows.")

    # STEP 4: Training manifest (keep all non-ambient, 800 sampled ambient)
    non_ambient_recs = [r for r in records if r["source_dataset"] not in config.AMBIENT_DATASETS]
    sampled_ambient_recs = sample_ambient_clips(
        records,
        target_total=config.AMBIENT_SAMPLE_TOTAL,
        random_seed=config.RANDOM_SEED,
    )
    training_records = non_ambient_recs + sampled_ambient_recs
    training_df = pd.DataFrame(training_records)[columns].sort_values("filepath", ascending=True).reset_index(drop=True)

    config.OUTPUT_TRAINING_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    training_df.to_csv(config.OUTPUT_TRAINING_MANIFEST, index=False, encoding="utf-8")
    print(f"Training manifest written to {config.OUTPUT_TRAINING_MANIFEST} with {len(training_df)} rows.")

    # Calculate training manifest class balance
    train_total = len(training_df)
    train_label_counts = Counter(training_df["label"])
    train_summary_lines = [
        f"Total training manifest clips: {train_total}",
        f"Retained non-ambient: {len(non_ambient_recs)}",
        f"Subsampled ambient (urbansound8k + esc50): {len(sampled_ambient_recs)}",
        "",
        "Training manifest overall class balance:",
    ]
    for lbl in config.LABELS:
        cnt = train_label_counts.get(lbl, 0)
        frac = cnt / train_total if train_total else 0
        train_summary_lines.append(f"  {lbl:12s}: {cnt:5d} ({frac:.2%})")

    normal_fraction = train_label_counts.get("Normal", 0) / train_total if train_total else 0
    if normal_fraction > config.MAX_CLASS_FRACTION:
        train_summary_lines.append(
            f"  WARNING: Normal is still above {config.MAX_CLASS_FRACTION:.0%}: {normal_fraction:.2%}"
        )
    else:
        train_summary_lines.append(
            f"  Normal fraction ({normal_fraction:.2%}) is <= {config.MAX_CLASS_FRACTION:.0%} cap!"
        )

    train_summary_lines.append("\nTraining manifest class distribution per dataset:")
    for src in sorted(training_df["source_dataset"].unique()):
        sub = training_df[training_df["source_dataset"] == src]
        cnts = Counter(sub["label"])
        dist = ", ".join(f"{lbl}: {cnts.get(lbl, 0)}" for lbl in config.LABELS)
        train_summary_lines.append(f"  {src:20s} -> {dist}")

    training_summary_str = "\n".join(train_summary_lines)
    print("\n=== TRAINING MANIFEST CLASS BALANCE ===")
    print(training_summary_str)

    # STEP 5: Create split file
    splits_df = create_splits(
        training_df,
        n_splits=config.N_SPLITS,
        random_seed=config.BALANCED_SPLIT_SEED,
        ood_dataset=config.OOD_DATASET,
    )
    config.OUTPUT_SPLITS.parent.mkdir(parents=True, exist_ok=True)
    splits_df.to_csv(config.OUTPUT_SPLITS, index=False, encoding="utf-8")
    print(f"Splits written to {config.OUTPUT_SPLITS} with {len(splits_df)} rows.")

    # Write SPLIT_INFO.txt
    splits_data = config.OUTPUT_SPLITS.read_bytes()
    splits_sha256 = hashlib.sha256(splits_data).hexdigest()

    non_ood_df = splits_df[splits_df["fold"] != "ood"]
    n_non_ood = len(non_ood_df)
    mean_fold_sz = n_non_ood / config.N_SPLITS
    fold_lens = [len(splits_df[splits_df["fold"] == str(i)]) for i in range(config.N_SPLITS)]
    calc_size_dev = max(abs(s - mean_fold_sz) / mean_fold_sz for s in fold_lens)

    overall_class_shares = {l: (non_ood_df["label"] == l).sum() / n_non_ood for l in config.LABELS}
    calc_class_dev = 0.0
    calc_missing = 0
    non_ood_datasets = sorted(non_ood_df["source_dataset"].unique())

    for i in range(config.N_SPLITS):
        f_df = splits_df[splits_df["fold"] == str(i)]
        f_len = len(f_df)
        for l in config.LABELS:
            f_share = (f_df["label"] == l).sum() / f_len if f_len else 0
            dev = abs(f_share - overall_class_shares[l]) * 100.0
            if dev > calc_class_dev:
                calc_class_dev = dev
        f_datasets = set(f_df["source_dataset"].unique())
        for d in non_ood_datasets:
            if d not in f_datasets:
                calc_missing += 1

    split_info_content = [
        "SheAlert Audio Pipeline - Frozen Split Information",
        "=" * 55,
        f"Generated Date : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Method         : Greedy smallest-bin-first + local search (move/swap)",
        f"Chosen Seed    : {config.BALANCED_SPLIT_SEED}",
        f"Size Dev       : {calc_size_dev:.4f} ({calc_size_dev:.2%})",
        f"Class Dev (pp) : {calc_class_dev:.2f} percentage points",
        f"Missing Sets   : {calc_missing}",
        f"SHA-256 Splits : {splits_sha256}",
        f"Fold Sizes     : {fold_lens}",
    ]
    split_info_path = getattr(config, "OUTPUT_SPLIT_INFO", config.OUTPUT_DIR / "SPLIT_INFO.txt")
    split_info_path.write_text("\n".join(split_info_content) + "\n", encoding="utf-8")
    print(f"Split info written to {split_info_path}")

    # Verify no group_id leakage
    fold_by_group = splits_df.groupby("group_id")["fold"].nunique()
    leaked_groups = int((fold_by_group > 1).sum())
    assert leaked_groups == 0, f"DATA LEAKAGE DETECTED: {leaked_groups} groups appear in multiple folds!"

    split_lines = [
        "=== SPLIT VERIFICATION REPORT ===",
        f"Group leakage across folds: {leaked_groups} (PASS - zero overlap)",
        "",
        "Per-fold row counts and class distribution:",
    ]
    all_folds = [str(i) for i in range(config.N_SPLITS)] + ["ood"]
    all_datasets = sorted(training_df["source_dataset"].unique())

    for fld in all_folds:
        f_df = splits_df[splits_df["fold"] == fld]
        tot = len(f_df)
        cnts = Counter(f_df["label"])
        dist_parts = []
        for l in config.LABELS:
            c = cnts.get(l, 0)
            pct = c / tot if tot else 0
            flag = " [!] < 5%" if pct < 0.05 and fld != "ood" else ""
            dist_parts.append(f"{l}: {c} ({pct:.1%}){flag}")
        split_lines.append(f"  Fold {fld:3s}: {tot:5d} clips -> {', '.join(dist_parts)}")

    split_lines.append("\nPer-fold dataset representation:")
    for fld in [str(i) for i in range(config.N_SPLITS)]:
        f_df = splits_df[splits_df["fold"] == fld]
        present = set(f_df["source_dataset"])
        missing = [d for d in all_datasets if d != config.OOD_DATASET and d not in present]
        flag = f" [!] MISSING: {missing}" if missing else " (All 9 datasets present)"
        split_lines.append(f"  Fold {fld}: {len(present)} datasets{flag}")

    ood_df = splits_df[splits_df["fold"] == "ood"]
    split_lines.append(f"\nOOD Evaluation Set ({config.OOD_DATASET}):")
    split_lines.append(f"  Total clips: {len(ood_df)}")
    for l in config.LABELS:
        c = (ood_df["label"] == l).sum()
        pct = c / len(ood_df) if len(ood_df) else 0
        split_lines.append(f"    {l:12s}: {c:3d} ({pct:.1%})")
    split_lines.append(
        "  Note: RealWorld-Urdu has only 77 clips (0 Distress, 38 Aggression, 39 Normal). "
        "It lacks Distress examples entirely and is too small alone for a stable multi-class F1 metric. "
        "It serves as a supplemental cross-dataset evaluation check."
    )

    split_summary_str = "\n".join(split_lines)
    print("\n" + split_summary_str)

    # Summaries
    clip_frac_summary_str = compute_clip_frac_summary(records)
    print("\n" + clip_frac_summary_str)

    comparison_str = compare_with_backup(df, splits_df)
    print("\n" + comparison_str)

    # Write summary text
    write_summary(
        records,
        pre_dedup_count,
        duplicates_dropped,
        rejected_count,
        rejection_reasons,
        rejections_by_source,
        rejection_table_str,
        group_report_str,
        training_summary_str,
        split_summary_str,
        clip_frac_summary_str,
        comparison_str,
        config.OUTPUT_SUMMARY,
    )



if __name__ == "__main__":
    main()
