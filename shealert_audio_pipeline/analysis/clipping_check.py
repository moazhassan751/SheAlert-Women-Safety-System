"""
clipping_check.py
-----------------
Detailed empirical investigation of clipping rejections in SheAlert audio QC.
Measures real audio sample distributions at native bit depth and sample rate.
"""

from pathlib import Path
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import soundfile as sf

import sys
# Add parent directory to path so config and utils can be imported
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from utils import resolve_relative_path


def get_longest_run(mask: np.ndarray) -> int:
    """Returns the length of the longest consecutive sequence of True values."""
    if not np.any(mask):
        return 0
    padded = np.concatenate(([False], mask, [False]))
    diffs = np.diff(padded.astype(np.int8))
    starts = np.where(diffs == 1)[0]
    ends = np.where(diffs == -1)[0]
    if len(starts) == 0:
        return 0
    return int(np.max(ends - starts))


def measure_clip(filepath: Path, full_path: Path):
    try:
        info = sf.info(str(full_path))
        subtype = info.subtype

        # Read native data
        if subtype == "PCM_16":
            data, sr = sf.read(str(full_path), dtype="int16")
            scale = 32768.0
        elif subtype in ("PCM_24", "PCM_32"):
            data, sr = sf.read(str(full_path), dtype="int32")
            scale = 2147483648.0
        elif subtype in ("FLOAT", "DOUBLE"):
            data, sr = sf.read(str(full_path), dtype="float64")
            scale = 1.0
        else:
            data, sr = sf.read(str(full_path), dtype="float64")
            scale = 1.0

        # Downmix if multi-channel (consistent with passes_qc mono evaluation)
        if data.ndim > 1:
            mono = data.mean(axis=1)
        else:
            mono = data

        abs_y = np.abs(mono.astype(np.float64)) / scale
        total_samples = len(abs_y)
        peak_abs = float(np.max(abs_y)) if total_samples else 0.0

        # Ceiling threshold 0.999 of full scale
        ceiling_mask = (abs_y >= 0.999)
        n_samples_at_ceiling = int(np.sum(ceiling_mask))
        frac_at_ceiling = float(n_samples_at_ceiling / total_samples) if total_samples else 0.0
        longest_run = get_longest_run(ceiling_mask)

        return {
            "filepath": str(filepath).replace("\\", "/"),
            "full_path": str(full_path),
            "format": info.format,
            "subtype": subtype,
            "sr": sr,
            "channels": info.channels,
            "total_samples": total_samples,
            "peak_abs": peak_abs,
            "n_samples_at_ceiling": n_samples_at_ceiling,
            "frac_at_ceiling": frac_at_ceiling,
            "longest_run": longest_run,
        }
    except Exception as e:
        return {
            "filepath": str(filepath).replace("\\", "/"),
            "error": str(e),
            "total_samples": 0,
            "peak_abs": 0.0,
            "n_samples_at_ceiling": 0,
            "frac_at_ceiling": 0.0,
            "longest_run": 0,
        }


def extract_metadata(row):
    fp = Path(row["filepath"])
    src = row["source_dataset"]
    actor = None
    emotion = None
    speaker = None

    if src == "semour":
        for part in fp.parts:
            if re.match(r"^actor_\d+$", part, re.IGNORECASE):
                actor = part
                break
        emotion = fp.parent.name
    elif src == "urdu_dataset_master":
        parts = fp.stem.split("_")
        speaker = parts[0] if len(parts) >= 1 else None
        emotion = fp.parent.name
    elif src == "urduser":
        parts = fp.stem.split("_")
        actor = f"actor_{parts[0]}" if len(parts) >= 1 else None
        emotion = fp.parent.name

    return actor, emotion, speaker


def main():
    print("Loading rejected_clips.csv and unified_manifest.csv...")
    df_rej = pd.read_csv(config.OUTPUT_REJECTED_CLIPS)
    df_manifest = pd.read_csv(config.OUTPUT_MANIFEST)

    clipped_rej = df_rej[df_rej["reason"] == "clipped"].copy()
    print(f"Total clipped rejections: {len(clipped_rej)}")

    # Comparison groups: 300 retained SEMOUR+ and 300 retained UrduSpeech
    retained_semour = df_manifest[df_manifest["source_dataset"] == "semour"].sample(300, random_state=42).copy()
    retained_semour["group"] = "retained_semour_sample"

    retained_urduspeech = df_manifest[df_manifest["source_dataset"] == "urduspeech"].sample(300, random_state=42).copy()
    retained_urduspeech["group"] = "retained_urduspeech_sample"

    # Assign groups to rejected clips
    def assign_group(row):
        src = row["source_dataset"]
        fp = Path(row["filepath"])
        if src == "semour":
            emotion = fp.parent.name
            if emotion.lower() in ("anger", "angry"):
                return "semour_rejected_anger"
            else:
                return "semour_rejected_other"
        elif src == "urdu_dataset_master":
            return "urdu_dataset_master_rejected"
        elif src == "esc50":
            return "esc50_rejected"
        elif src == "audioset_scream":
            return "audioset_rejected"
        elif src == "urduser":
            return "urduser_rejected"
        elif src == "realworld_urdu":
            return "realworld_urdu_rejected"
        elif src == "urbansound8k":
            return "urbansound8k_rejected"
        return f"{src}_rejected"

    clipped_rej["group"] = clipped_rej.apply(assign_group, axis=1)

    # Combine all clips to measure
    all_targets = pd.concat([clipped_rej, retained_semour, retained_urduspeech], ignore_index=True)
    print(f"Measuring {len(all_targets)} audio clips...")

    tasks = []
    for idx, row in all_targets.iterrows():
        rel_p = row["filepath"]
        full_p = resolve_relative_path(rel_p)
        tasks.append((idx, rel_p, full_p))

    results = [None] * len(tasks)
    with ThreadPoolExecutor(max_workers=8) as executor:
        future_to_idx = {
            executor.submit(measure_clip, rel_p, full_p): idx
            for idx, rel_p, full_p in tasks
        }
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            results[idx] = future.result()

    metrics_df = pd.DataFrame(results)
    for col in ["source_dataset", "group", "label"]:
        if col in all_targets.columns:
            metrics_df[col] = all_targets[col]

    # Add actor, emotion, speaker metadata
    meta = [extract_metadata(row) for _, row in metrics_df.iterrows()]
    metrics_df["actor"] = [m[0] for m in meta]
    metrics_df["emotion"] = [m[1] for m in meta]
    metrics_df["speaker"] = [m[2] for m in meta]

    print("\n" + "=" * 80)
    print("STEP 2: REAL CLIPPING MEASUREMENTS PER GROUP")
    print("=" * 80)

    target_groups = [
        "semour_rejected_anger",
        "semour_rejected_other",
        "urdu_dataset_master_rejected",
        "esc50_rejected",
        "audioset_rejected",
        "urbansound8k_rejected",
        "urduser_rejected",
        "realworld_urdu_rejected",
        "retained_semour_sample",
        "retained_urduspeech_sample",
    ]

    group_rows = []
    for grp in target_groups:
        sub = metrics_df[metrics_df["group"] == grp]
        if len(sub) == 0:
            continue
        n_clips = len(sub)
        fracs = sub["frac_at_ceiling"].values
        runs = sub["longest_run"].values

        med_frac = float(np.median(fracs))
        p90_frac = float(np.percentile(fracs, 90))
        p99_frac = float(np.percentile(fracs, 99))
        med_run = float(np.median(runs))

        # Rejection threshold boundary checks: rule was frac > 0.001
        n_below_001 = int(np.sum(fracs < 0.001))
        n_below_0001 = int(np.sum(fracs < 0.0001))
        n_eq_trigger = int(np.sum(np.isclose(fracs, 0.001, atol=1e-5)))

        group_rows.append({
            "group": grp,
            "clips": n_clips,
            "median_frac": med_frac,
            "p90_frac": p90_frac,
            "p99_frac": p99_frac,
            "median_run": med_run,
            "frac < 0.001": n_below_001,
            "frac < 0.0001": n_below_0001,
            "frac == 0.001": n_eq_trigger,
        })

    summary_df = pd.DataFrame(group_rows)
    print(summary_df.to_string(index=False))

    print("\n" + "=" * 80)
    print("STEP 3: WHERE THE REJECTIONS SIT")
    print("=" * 80)

    # 1. SEMOUR+ rejection distribution
    semour_all_rej = metrics_df[metrics_df["source_dataset"] == "semour"]
    semour_clipped_rej = semour_all_rej[semour_all_rej["group"].str.startswith("semour_rejected")]
    print(f"\nSEMOUR+ Clipped Rejections by Emotion:")
    print(semour_clipped_rej["emotion"].value_counts().to_string())

    print(f"\nSEMOUR+ Clipped Rejections by Actor x Emotion Crosstab:")
    semour_crosstab = pd.crosstab(semour_clipped_rej["actor"], semour_clipped_rej["emotion"], margins=True)
    print(semour_crosstab.to_string())

    # Calculate SEMOUR+ rejection rate per actor and per emotion against total loaded
    # Count total loaded from raw SEMOUR+ directory
    semour_root = config.DATASET_ROOTS["semour"]
    loaded_counts = Counter()
    actor_counts = Counter()
    emotion_counts = Counter()
    for wav_path in semour_root.rglob("*.wav"):
        em = wav_path.parent.name
        if em in config.SEMOUR_FOLDER_MAP:
            for part in wav_path.parts:
                if re.match(r"^actor_\d+$", part, re.IGNORECASE):
                    act = part
                    break
            else:
                act = "unknown"
            loaded_counts[(act, em)] += 1
            actor_counts[act] += 1
            emotion_counts[em] += 1

    print("\nSEMOUR+ Rejection Rates per Emotion (Clipped Rejections / Loaded Clips):")
    for em in sorted(emotion_counts.keys()):
        rej_cnt = len(semour_clipped_rej[semour_clipped_rej["emotion"] == em])
        tot_cnt = emotion_counts[em]
        pct = (rej_cnt / tot_cnt * 100) if tot_cnt else 0
        print(f"  {em:12s}: {rej_cnt:4d} / {tot_cnt:4d} rejected ({pct:.1f}%)")

    print("\nTop 5 SEMOUR+ Actors with Highest Clipping Rejections:")
    actor_rej_summary = []
    for act in sorted(actor_counts.keys()):
        rej_cnt = len(semour_clipped_rej[semour_clipped_rej["actor"] == act])
        tot_cnt = actor_counts[act]
        pct = (rej_cnt / tot_cnt * 100) if tot_cnt else 0
        actor_rej_summary.append({"actor": act, "rejected": rej_cnt, "loaded": tot_cnt, "pct_rejected": pct})
    actor_rej_df = pd.DataFrame(actor_rej_summary).sort_values("rejected", ascending=False)
    print(actor_rej_df.head(10).to_string(index=False))

    # 2. URDU-Dataset-master speaker distribution
    udm_rej = metrics_df[metrics_df["group"] == "urdu_dataset_master_rejected"]
    print(f"\nURDU-Dataset-master 15 Clipped Rejections by Speaker and Emotion:")
    udm_spk = udm_rej.groupby(["speaker", "emotion"]).size().reset_index(name="count")
    print(udm_spk.to_string(index=False))
    print(f"Unique speakers in UDM clipped rejections: {udm_rej['speaker'].unique().tolist()}")

    print("\n" + "=" * 80)
    print("STEP 4: WHAT-IF SCENARIOS (MEASURED COUNTS, NO MANIFEST CHANGES)")
    print("=" * 80)

    # 1,016 clipped clips
    clipped_clips_data = metrics_df[metrics_df["group"].str.contains("rejected")].copy()
    total_clipped = len(clipped_clips_data)
    print(f"Total evaluated clipped rejections: {total_clipped}")

    # Baseline manifest retained numbers
    # Full manifest: Distress: 7,402, Aggression: 6,039, Normal: 16,135 (Total: 29,576)
    base_counts = df_manifest["label"].value_counts().to_dict()

    # Determine unified label for each clipped clip
    def get_clip_unified_label(row):
        src = row["source_dataset"]
        em = row["emotion"]
        if src == "semour":
            return config.SEMOUR_FOLDER_MAP.get(em, "Unknown")
        elif src == "urdu_dataset_master":
            return config.URDU_MASTER_FOLDER_MAP.get(em, "Unknown")
        elif src == "urduser":
            return config.URDUSER_FOLDER_MAP.get(em, "Unknown")
        elif src in ("urbansound8k", "esc50"):
            return "Normal"
        elif src == "audioset_scream":
            return "Distress"
        elif src == "realworld_urdu":
            return config.REALWORLD_URDU_FOLDER_MAP.get(em, "Normal")
        return "Unknown"

    clipped_clips_data["unified_label"] = clipped_clips_data.apply(get_clip_unified_label, axis=1)

    scenarios = [
        ("a) frac_at_ceiling > 0.001 to reject (keep if frac <= 0.001)",
         lambda r: r["frac_at_ceiling"] <= 0.001),
        ("b) frac_at_ceiling > 0.005 to reject (keep if frac <= 0.005)",
         lambda r: r["frac_at_ceiling"] <= 0.005),
        ("c) longest_run >= 5 to reject (keep if longest_run < 5)",
         lambda r: r["longest_run"] < 5),
        ("d) skip clipping rule for semour, urduser, urdu_dataset_master",
         lambda r: r["source_dataset"] in ("semour", "urduser", "urdu_dataset_master")),
    ]

    for name, keep_fn in scenarios:
        kept_mask = clipped_clips_data.apply(keep_fn, axis=1)
        kept_df = clipped_clips_data[kept_mask]
        n_kept = len(kept_df)
        n_still_rej = total_clipped - n_kept

        print(f"\n--- Scenario {name} ---")
        print(f"Clips salvaged/kept: {n_kept} | Still rejected: {n_still_rej}")
        print("Kept clips by source_dataset:")
        print(kept_df["source_dataset"].value_counts().to_string())

        semour_kept = kept_df[kept_df["source_dataset"] == "semour"]
        if len(semour_kept):
            print("SEMOUR+ kept clips by emotion:")
            print(semour_kept["emotion"].value_counts().to_string())

        # Projected full corpus class balance
        added_labels = Counter(kept_df["unified_label"])
        new_distress = base_counts.get("Distress", 0) + added_labels.get("Distress", 0)
        new_aggression = base_counts.get("Aggression", 0) + added_labels.get("Aggression", 0)
        new_normal = base_counts.get("Normal", 0) + added_labels.get("Normal", 0)
        new_total = new_distress + new_aggression + new_normal

        print("Projected Full Corpus Class Balance:")
        print(f"  Distress   : {new_distress:5d} ({new_distress / new_total:.2%})")
        print(f"  Aggression : {new_aggression:5d} ({new_aggression / new_total:.2%})")
        print(f"  Normal     : {new_normal:5d} ({new_normal / new_total:.2%})")
        print(f"  Total Clips: {new_total:5d}")

    print("\n" + "=" * 80)
    print("STEP 5: LISTEN-CHECK SAMPLES")
    print("=" * 80)

    # 10 random rejected SEMOUR+ Anger clips (seed 42)
    semour_anger_rej = clipped_clips_data[
        (clipped_clips_data["source_dataset"] == "semour") &
        (clipped_clips_data["emotion"].str.lower().isin(["anger", "angry"]))
    ]
    semour_sample_10 = semour_anger_rej.sample(min(10, len(semour_anger_rej)), random_state=42)
    print("10 Sample Rejected SEMOUR+ Anger Clips (seed 42):")
    cols_to_print = ["filepath", "peak_abs", "frac_at_ceiling", "longest_run"]
    for idx, r in semour_sample_10[cols_to_print].iterrows():
        print(f"  {r['filepath']:55s} | peak={r['peak_abs']:.4f} | frac={r['frac_at_ceiling']:.6f} | run={r['longest_run']:2d}")

    # 5 random URDU-Dataset-master Neutral clips (seed 42)
    udm_neutral_rej = clipped_clips_data[
        (clipped_clips_data["source_dataset"] == "urdu_dataset_master") &
        (clipped_clips_data["emotion"].str.lower() == "neutral")
    ]
    udm_sample_5 = udm_neutral_rej.sample(min(5, len(udm_neutral_rej)), random_state=42)
    print("\n5 Sample Rejected URDU-Dataset-master Neutral Clips (seed 42):")
    for idx, r in udm_sample_5[cols_to_print].iterrows():
        print(f"  {r['filepath']:55s} | peak={r['peak_abs']:.4f} | frac={r['frac_at_ceiling']:.6f} | run={r['longest_run']:2d}")


if __name__ == "__main__":
    main()
