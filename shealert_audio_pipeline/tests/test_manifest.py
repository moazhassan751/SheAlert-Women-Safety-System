import pytest
import pandas as pd
from pathlib import Path
import config

def test_manifest_exists():
    assert config.OUTPUT_MANIFEST.exists(), f"Manifest file does not exist at {config.OUTPUT_MANIFEST}"

def test_manifest_structure_and_integrity():
    df = pd.read_csv(config.OUTPUT_MANIFEST)
    assert len(df) > 0, "Manifest is empty"
    
    # Check expected columns
    expected_cols = ["filepath", "label", "source_dataset", "speaker_id"]
    assert list(df.columns) == expected_cols

    # No NaN in critical columns
    assert df["filepath"].notna().all(), "Found NaN in filepath column"
    assert df["label"].notna().all(), "Found NaN in label column"
    assert df["source_dataset"].notna().all(), "Found NaN in source_dataset column"

    # All labels belong to taxonomy
    invalid_labels = set(df["label"].unique()) - set(config.LABELS)
    assert len(invalid_labels) == 0, f"Found invalid labels: {invalid_labels}"

    # Check a random sample of files exist on disk
    sample = df.sample(min(100, len(df)), random_state=42)
    for path_str in sample["filepath"]:
        assert Path(path_str).exists(), f"Sampled audio file not found: {path_str}"

def test_summary_exists():
    assert config.OUTPUT_SUMMARY.exists(), f"Summary file does not exist at {config.OUTPUT_SUMMARY}"
    content = config.OUTPUT_SUMMARY.read_text()
    assert "SheAlert" in content
    assert "Total clips retained" in content
    for label in config.LABELS:
        assert label in content
