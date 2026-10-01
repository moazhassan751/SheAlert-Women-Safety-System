"""Automated tests for SheAlert YAMNet model architecture and end-to-end wrapper."""

import pytest
import numpy as np
import tensorflow as tf
from models.yamnet_classifier import (
    build_classifier_head,
    load_yamnet_backbone,
    EndToEndYAMNetClassifier,
    LABEL_TO_INDEX,
    INDEX_TO_LABEL,
    NUM_CLASSES,
    EMBEDDING_DIM,
)


def test_labels_and_classes():
    assert len(LABEL_TO_INDEX) == NUM_CLASSES
    assert set(LABEL_TO_INDEX.keys()) == {"Distress", "Aggression", "Normal"}
    for label, idx in LABEL_TO_INDEX.items():
        assert INDEX_TO_LABEL[idx] == label


def test_classifier_head_architecture():
    head = build_classifier_head(embedding_dim=EMBEDDING_DIM, num_classes=NUM_CLASSES)
    assert head.input_shape == (None, EMBEDDING_DIM)
    assert head.output_shape == (None, NUM_CLASSES)

    # Test inference with dummy embeddings
    batch_size = 8
    dummy_embeddings = np.random.randn(batch_size, EMBEDDING_DIM).astype(np.float32)
    predictions = head(dummy_embeddings).numpy()

    assert predictions.shape == (batch_size, NUM_CLASSES)
    # Probabilities should sum to 1.0 across classes
    row_sums = np.sum(predictions, axis=1)
    np.testing.assert_allclose(row_sums, np.ones(batch_size), atol=1e-5)
    assert (predictions >= 0.0).all() and (predictions <= 1.0).all()


def test_end_to_end_yamnet_inference():
    backbone = load_yamnet_backbone()
    head = build_classifier_head()
    e2e_model = EndToEndYAMNetClassifier(backbone, head)

    # Synthetic 1.5s sine wave at 16 kHz
    sr = 16000
    duration = 1.5
    t = np.linspace(0, duration, int(sr * duration), endpoint=False, dtype=np.float32)
    sine_wave = 0.5 * np.sin(2 * np.pi * 440 * t)

    result = e2e_model(tf.constant(sine_wave, dtype=tf.float32))
    assert "probabilities" in result
    assert "patch_probabilities" in result

    probs = result["probabilities"].numpy()
    assert probs.shape == (NUM_CLASSES,)
    np.testing.assert_allclose(np.sum(probs), 1.0, atol=1e-5)
    assert (probs >= 0.0).all() and (probs <= 1.0).all()

    patch_probs = result["patch_probabilities"].numpy()
    assert patch_probs.ndim == 2
    assert patch_probs.shape[1] == NUM_CLASSES
    assert patch_probs.shape[0] >= 1
