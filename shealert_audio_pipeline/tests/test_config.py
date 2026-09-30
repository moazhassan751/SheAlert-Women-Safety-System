import pytest
from pathlib import Path
import config

def test_labels_taxonomy():
    assert config.LABELS == ["Distress", "Aggression", "Normal"]

def test_base_path_exists():
    assert config.BASE_PATH.exists(), f"BASE_PATH not found: {config.BASE_PATH}"

def test_dataset_roots_exist():
    for name, path in config.DATASET_ROOTS.items():
        assert path.exists(), f"Dataset root for '{name}' does not exist: {path}"

def test_output_paths_anchored():
    assert config.OUTPUT_DIR.name == "output"
    assert config.OUTPUT_MANIFEST.name == "unified_manifest.csv"
    assert config.OUTPUT_SUMMARY.name == "corpus_summary.txt"
