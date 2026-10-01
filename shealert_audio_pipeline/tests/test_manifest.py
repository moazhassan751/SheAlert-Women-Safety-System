import pytest
import pandas as pd
from pathlib import Path
import config
from utils import to_relative_path, resolve_relative_path, extract_group_id


def test_manifest_exists():
    assert config.OUTPUT_MANIFEST.exists(), f"Manifest file does not exist at {config.OUTPUT_MANIFEST}"


def test_manifest_structure_and_integrity():
    df = pd.read_csv(config.OUTPUT_MANIFEST)
    assert len(df) > 0, "Manifest is empty"

    # Expected columns: filepath, label, source_dataset, duration, speaker_id, group_id, clip_frac
    expected_cols = ["filepath", "label", "source_dataset", "duration", "speaker_id", "group_id", "clip_frac"]
    assert list(df.columns) == expected_cols

    # No NaN in critical columns
    assert df["filepath"].notna().all(), "Found NaN in filepath column"
    assert df["label"].notna().all(), "Found NaN in label column"
    assert df["source_dataset"].notna().all(), "Found NaN in source_dataset column"
    assert df["group_id"].notna().all(), "Found NaN in group_id column"
    assert df["clip_frac"].notna().all(), "Found NaN in clip_frac column"

    # All labels belong to taxonomy
    invalid_labels = set(df["label"].unique()) - set(config.LABELS)
    assert len(invalid_labels) == 0, f"Found invalid labels: {invalid_labels}"

    # Verify paths are relative and forward-slashed
    for p_str in df["filepath"].sample(min(50, len(df)), random_state=42):
        assert "\\" not in p_str, f"Path contains backslash instead of forward slash: {p_str}"
        assert not Path(p_str).is_absolute(), f"Path is absolute instead of relative: {p_str}"

    # Check a random sample of files exist on disk when resolved
    sample = df.sample(min(100, len(df)), random_state=42)
    for path_str in sample["filepath"]:
        abs_p = resolve_relative_path(path_str)
        assert abs_p.exists(), f"Sampled audio file not found on disk: {abs_p}"


def test_relative_path_helpers():
    sample_rel = "UrduSpeech/Distress/test.wav"
    abs_p = resolve_relative_path(sample_rel)
    assert abs_p.is_absolute()
    assert str(config.BASE_PATH).replace("\\", "/") in str(abs_p).replace("\\", "/")

    # round-trip check
    rel_back = to_relative_path(abs_p)
    assert rel_back == sample_rel
    assert "\\" not in rel_back


def test_group_id_extraction_rules():
    cases = [
        ("UrduSpeech/Distress/SPEAKER_0169_DRAMA_001273.wav", "urduspeech", "urduspeech_SPEAKER_0169", False),
        ("SEMOUR+/Actor_3/Fearful/clip1.wav", "semour", "semour_actor_3", False),
        ("CREMA-D/AudioMP3/1001_DFA_ANG_XX.mp3", "cremad", "cremad_1001", False),
        ("Radvess/Actor_12/03-01-06-01-02-01-12.wav", "ravdess", "ravdess_actor_12", False),
        ("UrduSER/Fear/10_1_1_01.wav", "urduser", "urduser_actor_10", False),
        ("URDU-Dataset-master/Angry/SM1_F4_A04.wav", "urdu_dataset_master", "urdu_dataset_master_SM1", False),
        ("AudioSet_Screaming/wav/-20uudT97E0_30.00_40.00.wav", "audioset_scream", "audioset_scream_-20uudT97E0", False),
        ("ESC-50/audio/1-100032-A-0.wav", "esc50", "esc50_100032", False),
        ("UrbanSound8K/wav_by_fold/fold1/101415-3-0-2.wav", "urbansound8k", "urbansound8k_101415", False),
        ("RealWorld-Urdu/Angry/WhatsApp Audio 2025-12-20 at 11.30.19_c308edfd.wav", "realworld_urdu", "realworld_urdu_WhatsApp Audio 2025-12-20 at 11.30.19_c308edfd", True),
    ]
    for rel_path, src, expected_gid, expected_fallback in cases:
        gid, is_fb = extract_group_id(rel_path, src)
        assert gid == expected_gid, f"Expected {expected_gid}, got {gid} for {rel_path}"
        assert is_fb == expected_fallback, f"Expected fallback={expected_fallback} for {rel_path}"


def test_rejected_clips():
    assert config.OUTPUT_REJECTED_CLIPS.exists(), f"Missing {config.OUTPUT_REJECTED_CLIPS}"
    df_rej = pd.read_csv(config.OUTPUT_REJECTED_CLIPS)
    assert len(df_rej) > 0, "Rejected clips CSV is empty"
    assert list(df_rej.columns) == ["filepath", "source_dataset", "reason"]
    valid_reasons = {"low_snr", "clipped", "too_short", "unreadable"}
    assert set(df_rej["reason"].unique()).issubset(valid_reasons)


def test_exempt_datasets_not_rejected_for_clipping():
    assert config.OUTPUT_REJECTED_CLIPS.exists()
    df_rej = pd.read_csv(config.OUTPUT_REJECTED_CLIPS)
    clipped_rejs = df_rej[df_rej["reason"] == "clipped"]

    # None of the exempt datasets should appear in clipping rejections
    exempt_in_clipped = set(clipped_rejs["source_dataset"]).intersection(set(config.CLIP_EXEMPT_DATASETS))
    assert len(exempt_in_clipped) == 0, f"Found exempt datasets in clipping rejections: {exempt_in_clipped}"

    # Clipping rejections should only be in esc50, urbansound8k, realworld_urdu
    allowed_clipped = {"esc50", "urbansound8k", "realworld_urdu"}
    actual_clipped_datasets = set(clipped_rejs["source_dataset"])
    assert actual_clipped_datasets.issubset(allowed_clipped), f"Unexpected datasets in clipping rejections: {actual_clipped_datasets}"


def test_training_manifest():
    assert config.OUTPUT_TRAINING_MANIFEST.exists(), f"Missing {config.OUTPUT_TRAINING_MANIFEST}"
    df_train = pd.read_csv(config.OUTPUT_TRAINING_MANIFEST)
    expected_cols = ["filepath", "label", "source_dataset", "duration", "speaker_id", "group_id", "clip_frac"]
    assert list(df_train.columns) == expected_cols

    # Exactly 800 ambient clips
    ambient_count = df_train["source_dataset"].isin(["urbansound8k", "esc50"]).sum()
    assert ambient_count == 800, f"Expected 800 ambient clips, got {ambient_count}"

    # Normal class fraction <= 40%
    normal_pct = (df_train["label"] == "Normal").sum() / len(df_train)
    assert normal_pct <= config.MAX_CLASS_FRACTION, f"Normal fraction {normal_pct:.2%} exceeds 40%"


def test_clip_frac_in_both_manifests():
    df_u = pd.read_csv(config.OUTPUT_MANIFEST)
    df_t = pd.read_csv(config.OUTPUT_TRAINING_MANIFEST)

    assert "clip_frac" in df_u.columns, "clip_frac missing from unified manifest"
    assert "clip_frac" in df_t.columns, "clip_frac missing from training manifest"

    assert df_u["clip_frac"].notna().all(), "NaN found in unified manifest clip_frac"
    assert df_t["clip_frac"].notna().all(), "NaN found in training manifest clip_frac"

    assert (df_u["clip_frac"] >= 0.0).all(), "Negative clip_frac found in unified manifest"
    assert (df_t["clip_frac"] >= 0.0).all(), "Negative clip_frac found in training manifest"


def test_splits_and_leakage():
    assert config.OUTPUT_SPLITS.exists(), f"Missing {config.OUTPUT_SPLITS}"
    df_splits = pd.read_csv(config.OUTPUT_SPLITS)
    expected_cols = ["filepath", "label", "source_dataset", "group_id", "fold"]
    assert list(df_splits.columns) == expected_cols

    # Group leakage across folds (folds 0-4 and ood)
    group_folds = df_splits.groupby("group_id")["fold"].nunique()
    leaked = (group_folds > 1).sum()
    assert leaked == 0, f"Detected {leaked} groups appearing in multiple folds"

    # OOD fold contains exactly realworld_urdu
    ood_df = df_splits[df_splits["fold"] == "ood"]
    assert (ood_df["source_dataset"] == "realworld_urdu").all()
    assert (df_splits[df_splits["source_dataset"] == "realworld_urdu"]["fold"] == "ood").all()

    # Folds 0-4 have valid distributions
    for fld in range(5):
        f_df = df_splits[df_splits["fold"] == str(fld)]
        assert len(f_df) > 0, f"Fold {fld} is empty"
        for lbl in config.LABELS:
            pct = (f_df["label"] == lbl).sum() / len(f_df)
            assert pct >= 0.05, f"Fold {fld} class {lbl} fraction {pct:.2%} is below 5%"


def test_deterministic_output_order():
    for out_path in [config.OUTPUT_MANIFEST, config.OUTPUT_TRAINING_MANIFEST, config.OUTPUT_SPLITS, config.OUTPUT_REJECTED_CLIPS]:
        assert out_path.exists(), f"File {out_path} does not exist"
        df = pd.read_csv(out_path)
        assert df["filepath"].is_monotonic_increasing, f"Filepaths in {out_path.name} are not strictly sorted"


def test_zero_group_leakage():
    df_splits = pd.read_csv(config.OUTPUT_SPLITS)
    group_folds = df_splits.groupby("group_id")["fold"].nunique()
    assert (group_folds == 1).all(), "Group leakage detected: one or more groups appear in multiple folds"


def test_every_non_ood_dataset_present_in_every_fold():
    df_splits = pd.read_csv(config.OUTPUT_SPLITS)
    non_ood_df = df_splits[df_splits["fold"] != "ood"]
    expected_datasets = set(non_ood_df["source_dataset"].unique())

    for fld in range(5):
        f_df = df_splits[df_splits["fold"] == str(fld)]
        fold_datasets = set(f_df["source_dataset"].unique())
        missing = expected_datasets - fold_datasets
        assert len(missing) == 0, f"Fold {fld} is missing datasets: {missing}"


def test_split_info_exists():
    split_info_path = getattr(config, "OUTPUT_SPLIT_INFO", config.OUTPUT_DIR / "SPLIT_INFO.txt")
    assert split_info_path.exists(), f"Missing {split_info_path}"
    content = split_info_path.read_text(encoding="utf-8")
    assert "Chosen Seed" in content
    assert str(config.BALANCED_SPLIT_SEED) in content
    assert "SHA-256 Splits" in content


def test_summary_exists():
    assert config.OUTPUT_SUMMARY.exists(), f"Summary file does not exist at {config.OUTPUT_SUMMARY}"
    content = config.OUTPUT_SUMMARY.read_text(encoding="utf-8")
    assert "SheAlert" in content
    assert "Total clips retained" in content
    assert "Rejection Crosstab" in content
    assert "Group ID Report" in content
    assert "SPLIT VERIFICATION REPORT" in content
    assert "CLIP_FRAC SUMMARY PER DATASET" in content
    assert "BEFORE VS AFTER COMPARISON" in content
