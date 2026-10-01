"""SheAlert Audio Pipeline Models Package."""

from .yamnet_classifier import (
    load_yamnet_backbone,
    build_classifier_head,
    EndToEndYAMNetClassifier,
    LABEL_TO_INDEX,
    INDEX_TO_LABEL,
    NUM_CLASSES,
    EMBEDDING_DIM,
)

__all__ = [
    "load_yamnet_backbone",
    "build_classifier_head",
    "EndToEndYAMNetClassifier",
    "LABEL_TO_INDEX",
    "INDEX_TO_LABEL",
    "NUM_CLASSES",
    "EMBEDDING_DIM",
]
