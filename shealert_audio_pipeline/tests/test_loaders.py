import pytest
from pathlib import Path
import config
from loaders import cremad, ravdess, esc50, urbansound8k, audioset_scream, folder_emotion

REQUIRED_KEYS = {"filepath", "label", "source_dataset", "speaker_id"}

def validate_records(records, expected_source):
    assert len(records) > 0, f"No records returned for {expected_source}"
    for r in records[:50]:  # check first 50 sample records
        assert REQUIRED_KEYS.issubset(r.keys()), f"Missing keys in record: {r}"
        assert r["label"] in config.LABELS, f"Invalid label: {r['label']}"
        assert r["source_dataset"] == expected_source
        assert Path(r["filepath"]).exists(), f"Audio file does not exist: {r['filepath']}"

def test_loader_cremad():
    records = cremad.load(config.DATASET_ROOTS["cremad"])
    validate_records(records, "cremad")

def test_loader_ravdess():
    records = ravdess.load(config.DATASET_ROOTS["ravdess"])
    validate_records(records, "ravdess")

def test_loader_esc50():
    records = esc50.load(config.DATASET_ROOTS["esc50_audio"], config.DATASET_ROOTS["esc50_meta"])
    validate_records(records, "esc50")

def test_loader_urbansound8k():
    records = urbansound8k.load(config.DATASET_ROOTS["urbansound8k_audio"], config.DATASET_ROOTS["urbansound8k_meta"])
    validate_records(records, "urbansound8k")

def test_loader_audioset_scream():
    records = audioset_scream.load(config.DATASET_ROOTS["audioset_scream"])
    validate_records(records, "audioset_scream")

def test_loader_urduser():
    records = folder_emotion.load(config.DATASET_ROOTS["urduser"], config.URDUSER_FOLDER_MAP, "urduser")
    validate_records(records, "urduser")

def test_loader_urduspeech():
    records = folder_emotion.load(config.DATASET_ROOTS["urduspeech"], config.URDUSPEECH_FOLDER_MAP, "urduspeech")
    validate_records(records, "urduspeech")
