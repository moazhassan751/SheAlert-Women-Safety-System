"""
label_check/audit_urduspeech.py
Audit UrduSpeech label noise without audio playback.
Outputs written strictly to label_check/
"""

import sys
from pathlib import Path
import pandas as pd
import hashlib

def run_audit():
    manifest_path = Path(r"F:\SheAlert-Women-Safety-System\audio data set fyp\sheAlert_urduspeech_std_manifest.csv")
    print(f"Loading UrduSpeech manifest: {manifest_path}")
    raw_df = pd.read_csv(manifest_path)
    total_raw = len(raw_df)
    print(f"B1. Total rows in raw manifest: {total_raw}")

    # Deduplicate on clip_filename
    dedup_df = raw_df.drop_duplicates(subset=["clip_filename"]).copy().reset_index(drop=True)
    n_dedup = len(dedup_df)
    print(f"B1. Deduplicated rows (unique clips): {n_dedup}")

    # Define keyword lists
    DISTRESS_WORDS = ["fear", "anxious", "scared", "panic", "terrified", "distress"]
    AGGRESSION_WORDS = ["anger", "angry", "aggress", "threat", "rage", "furious"]
    NORMAL_WORDS = ["neutral", "calm"]

    # Keyword matching
    dedup_df["em_str"] = dedup_df["emotion_raw"].fillna("").astype(str)
    dedup_df["em_lower"] = dedup_df["em_str"].str.lower()

    dedup_df["matched_distress"] = dedup_df["em_lower"].apply(
        lambda s: [w for w in DISTRESS_WORDS if w in s]
    )
    dedup_df["matched_aggression"] = dedup_df["em_lower"].apply(
        lambda s: [w for w in AGGRESSION_WORDS if w in s]
    )
    dedup_df["matched_normal"] = dedup_df["em_lower"].apply(
        lambda s: [w for w in NORMAL_WORDS if w in s]
    )

    has_dist = dedup_df["matched_distress"].apply(lambda x: len(x) > 0)
    has_aggr = dedup_df["matched_aggression"].apply(lambda x: len(x) > 0)
    has_norm = dedup_df["matched_normal"].apply(lambda x: len(x) > 0)

    # Multi-match
    both_da = dedup_df[has_dist & has_aggr].copy()
    both_dn = dedup_df[has_dist & has_norm].copy()
    both_an = dedup_df[has_aggr & has_norm].copy()

    multi_match_mask = (has_dist.astype(int) + has_aggr.astype(int) + has_norm.astype(int)) >= 2
    multi_match_df = dedup_df[multi_match_mask].copy()

    # Save multi_match_rows.csv
    multi_csv_path = Path("label_check/multi_match_rows.csv")
    multi_match_df.to_csv(multi_csv_path, index=False, encoding="utf-8")
    print(f"\nSaved {len(multi_match_df)} multi-match rows to {multi_csv_path}")

    # B2 a) Both Distress and Aggression
    print("\n" + "=" * 70)
    print(f"B2 a) Rows matching both Distress and Aggression words: {len(both_da)}")
    print("=" * 70)
    print(f"{'No.':3s} | {'clip_filename':32s} | {'sheAlert_class':12s} | emotion_raw")
    print("-" * 70)
    for idx, (_, r) in enumerate(both_da.head(15).iterrows(), 1):
        print(f"{idx:3d} | {r['clip_filename']:32s} | {r['sheAlert_class']:12s} | {r['emotion_raw']}")

    # B2 b) Distress & Normal, Aggression & Normal
    print("\n" + "=" * 70)
    print(f"B2 b) Rows matching Distress and Normal: {len(both_dn)}")
    print("=" * 70)
    print(f"{'No.':3s} | {'clip_filename':32s} | {'sheAlert_class':12s} | emotion_raw")
    print("-" * 70)
    for idx, (_, r) in enumerate(both_dn.head(10).iterrows(), 1):
        print(f"{idx:3d} | {r['clip_filename']:32s} | {r['sheAlert_class']:12s} | {r['emotion_raw']}")

    print("\n" + "=" * 70)
    print(f"B2 b) Rows matching Aggression and Normal: {len(both_an)}")
    print("=" * 70)
    print(f"{'No.':3s} | {'clip_filename':32s} | {'sheAlert_class':12s} | emotion_raw")
    print("-" * 70)
    for idx, (_, r) in enumerate(both_an.head(10).iterrows(), 1):
        print(f"{idx:3d} | {r['clip_filename']:32s} | {r['sheAlert_class']:12s} | {r['emotion_raw']}")

    # B2 c) Top 25 emotion_raw per class
    print("\n" + "=" * 70)
    print("B2 c) 25 Most Frequent emotion_raw values per sheAlert_class:")
    print("=" * 70)
    for cls in ["Distress", "Aggression", "Normal"]:
        sub = dedup_df[dedup_df["sheAlert_class"] == cls]
        print(f"\n--- Class: {cls} (total clips = {len(sub)}) ---")
        top25 = sub["emotion_raw"].value_counts().head(25)
        for rank, (val, cnt) in enumerate(top25.items(), 1):
            print(f"  {rank:2d}. {val:50s} : {cnt:4d}")

    # B2 d)
    dist_only_anx = dedup_df[(dedup_df["sheAlert_class"] == "Distress") & dedup_df["matched_distress"].apply(lambda x: x == ["anxious"])].copy()
    aggr_only_thr = dedup_df[(dedup_df["sheAlert_class"] == "Aggression") & dedup_df["matched_aggression"].apply(lambda x: x == ["threat"])].copy()

    print("\n" + "=" * 70)
    print(f"B2 d) Distress clips matching ONLY 'anxious': {len(dist_only_anx)}")
    print(f"B2 d) Aggression clips matching ONLY 'threat':  {len(aggr_only_thr)}")
    print("=" * 70)

    # B2 e) Empty emotion_raw
    empty_em = dedup_df[dedup_df["em_str"].str.strip() == ""]
    print(f"\nB2 e) Rows with empty emotion_raw: {len(empty_em)}")

    # B2 f) Normal with Distress or Aggression word
    norm_with_da = dedup_df[(dedup_df["sheAlert_class"] == "Normal") & (has_dist | has_aggr)].copy()
    print("\n" + "=" * 70)
    print(f"B2 f) Rows labelled Normal whose emotion_raw contains any Distress or Aggression word: {len(norm_with_da)}")
    print("=" * 70)
    if len(norm_with_da) > 0:
        for idx, (_, r) in enumerate(norm_with_da.head(15).iterrows(), 1):
            print(f"{idx:3d} | {r['clip_filename']:32s} | {r['sheAlert_class']:12s} | {r['emotion_raw']}")
    else:
        print("  None found (0 rows).")

    # B2 g) Distress with no Distress word, Aggression with no Aggression word
    dist_no_d = dedup_df[(dedup_df["sheAlert_class"] == "Distress") & (~has_dist)].copy()
    aggr_no_a = dedup_df[(dedup_df["sheAlert_class"] == "Aggression") & (~has_aggr)].copy()

    print("\n" + "=" * 70)
    print(f"B2 g) Rows labelled Distress with no Distress word: {len(dist_no_d)}")
    print("=" * 70)
    if len(dist_no_d) > 0:
        for idx, (_, r) in enumerate(dist_no_d.head(15).iterrows(), 1):
            print(f"{idx:3d} | {r['clip_filename']:32s} | {r['sheAlert_class']:12s} | {r['emotion_raw']}")
    else:
        print("  None found (0 rows).")

    print("\n" + "=" * 70)
    print(f"B2 g) Rows labelled Aggression with no Aggression word: {len(aggr_no_a)}")
    print("=" * 70)
    if len(aggr_no_a) > 0:
        for idx, (_, r) in enumerate(aggr_no_a.head(15).iterrows(), 1):
            print(f"{idx:3d} | {r['clip_filename']:32s} | {r['sheAlert_class']:12s} | {r['emotion_raw']}")
    else:
        print("  None found (0 rows).")

    # Suspect rows compilation from d, f, g
    dist_only_anx["suspect_reason"] = "distress_only_anxious"
    aggr_only_thr["suspect_reason"] = "aggression_only_threat"
    norm_with_da["suspect_reason"] = "normal_contains_distress_or_aggression"
    dist_no_d["suspect_reason"] = "distress_no_distress_word"
    aggr_no_a["suspect_reason"] = "aggression_no_aggression_word"

    suspect_parts = [dist_only_anx, aggr_only_thr, norm_with_da, dist_no_d, aggr_no_a]
    suspect_df = pd.concat(suspect_parts, ignore_index=True)
    suspect_df = suspect_df.drop_duplicates(subset=["clip_filename"]).reset_index(drop=True)

    suspect_csv_path = Path("label_check/suspect_rows.csv")
    suspect_df.to_csv(suspect_csv_path, index=False, encoding="utf-8")
    print(f"\nSaved {len(suspect_df)} suspect rows to {suspect_csv_path}")

    # Suspect rates
    total_suspect = len(suspect_df)
    pct_suspect = (total_suspect / n_dedup) * 100.0

    print("\n" + "=" * 70)
    print(f"B2 Summary: Total distinct suspect rows = {total_suspect} ({pct_suspect:.2f}% of {n_dedup})")
    print("Suspect rate per sheAlert_class:")
    for cls in ["Distress", "Aggression", "Normal"]:
        cls_total = (dedup_df["sheAlert_class"] == cls).sum()
        cls_susp = (suspect_df["sheAlert_class"] == cls).sum()
        rate = (cls_susp / cls_total) * 100.0 if cls_total else 0.0
        print(f"  {cls:12s}: {cls_susp:4d} / {cls_total:4d} ({rate:.2f}%)")
    print("=" * 70)

    # B3 Matching with output/unified_manifest.csv
    print("\n" + "=" * 70)
    print("B3: Matching UrduSpeech rows in output/unified_manifest.csv to source manifest")
    print("=" * 70)
    src_filenames_lower = raw_df["clip_filename"].astype(str).str.lower()
    dup_counts = src_filenames_lower.value_counts()
    n_mult_appear = (dup_counts > 1).sum()

    uni_df = pd.read_csv("output/unified_manifest.csv")
    us_uni = uni_df[uni_df["source_dataset"] == "urduspeech"].copy()
    us_uni["basename_lower"] = us_uni["filepath"].apply(lambda p: Path(p).name.lower())

    src_unique_set = set(src_filenames_lower.unique())
    matched_count = us_uni["basename_lower"].isin(src_unique_set).sum()
    unmatched_count = (~us_uni["basename_lower"].isin(src_unique_set)).sum()

    print(f"Total UrduSpeech rows in unified_manifest.csv : {len(us_uni)}")
    print(f"Matched count                                 : {matched_count}")
    print(f"Unmatched count                               : {unmatched_count}")
    print(f"Clip filenames appearing >1x in source manifest: {n_mult_appear} distinct filenames ({len(raw_df) - src_filenames_lower.nunique()} duplicate rows)")

    # B4 Verify output/ unchanged
    print("\n" + "=" * 70)
    print("B4: Output Integrity Check (comparing against output_frozen_split_v1)")
    print("=" * 70)
    for fn in ["unified_manifest.csv", "training_manifest.csv", "splits.csv"]:
        h_cur = hashlib.sha256(Path("output", fn).read_bytes()).hexdigest()
        h_frz = hashlib.sha256(Path("output_frozen_split_v1", fn).read_bytes()).hexdigest()
        match_str = "MATCH" if h_cur == h_frz else "MISMATCH"
        print(f"  {fn:25s}: {h_cur} [{match_str}]")

if __name__ == "__main__":
    run_audit()
