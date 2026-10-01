"""SheAlert YAMNet Model Architecture & Transfer Learning Head.

Provides:
- load_yamnet_backbone(): Retrieves Google's pre-trained YAMNet model via kagglehub / cache.
- build_classifier_head(): Constructs the customized 3-class safety classification head.
- build_end_to_end_model(): Combines YAMNet feature extractor with the custom head
  into a single computational graph taking raw audio waveform -> 3-class safety predictions.
"""

from typing import Tuple, Optional
from pathlib import Path
import numpy as np
import tensorflow as tf
import kagglehub

LABEL_TO_INDEX = {"Distress": 0, "Aggression": 1, "Normal": 2}
INDEX_TO_LABEL = {0: "Distress", 1: "Aggression", 2: "Normal"}
NUM_CLASSES = 3
EMBEDDING_DIM = 1024


def load_yamnet_backbone(local_path: Optional[str] = None) -> tf.saved_model:
    """Loads the pre-trained YAMNet SavedModel from local cache or Kaggle Hub.
    
    Returns:
        Loaded tf.saved_model instance.
    """
    if local_path and Path(local_path).exists():
        model_path = str(local_path)
    else:
        model_path = kagglehub.model_download("google/yamnet/tensorFlow2/yamnet")
    
    return tf.saved_model.load(model_path)


def build_classifier_head(
    embedding_dim: int = EMBEDDING_DIM,
    num_classes: int = NUM_CLASSES,
    l2_reg: float = 1e-4,
    dropout_rate_1: float = 0.3,
    dropout_rate_2: float = 0.2,
    name: str = "shealert_safety_head",
) -> tf.keras.Model:
    """Builds a high-accuracy, regularized MLP classification head.
    
    Takes 1024-dimensional YAMNet embedding vectors and outputs
    calibrated class probabilities over [Distress, Aggression, Normal].
    """
    inputs = tf.keras.Input(shape=(embedding_dim,), name="embedding_input")
    
    # Layer 1
    x = tf.keras.layers.Dense(
        256,
        activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(l2_reg),
        name="dense_256",
    )(inputs)
    x = tf.keras.layers.BatchNormalization(name="bn_1")(x)
    x = tf.keras.layers.Dropout(dropout_rate_1, name="dropout_1")(x)
    
    # Layer 2
    x = tf.keras.layers.Dense(
        64,
        activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(l2_reg),
        name="dense_64",
    )(x)
    x = tf.keras.layers.BatchNormalization(name="bn_2")(x)
    x = tf.keras.layers.Dropout(dropout_rate_2, name="dropout_2")(x)
    
    # Output Layer
    outputs = tf.keras.layers.Dense(
        num_classes,
        activation="softmax",
        dtype="float32",
        name="safety_probabilities",
    )(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name=name)
    return model


class EndToEndYAMNetClassifier(tf.Module):
    """End-to-End inference graph combining YAMNet backbone with the SheAlert head.
    
    Accepts raw PCM audio waveform of variable length (float32 array, 16 kHz)
    and outputs aggregated 3-class safety probabilities.
    Designed for direct export to TFLite (LM-5).
    """

    def __init__(self, yamnet_backbone: tf.saved_model, classifier_head: tf.keras.Model):
        super().__init__()
        self.yamnet = yamnet_backbone
        self.classifier_head = classifier_head

    @tf.function(input_signature=[tf.TensorSpec(shape=[None], dtype=tf.float32, name="waveform")])
    def __call__(self, waveform: tf.Tensor):
        # 1. Extract YAMNet predictions & embeddings: (N_patches, 1024)
        _, embeddings, _ = self.yamnet(waveform)
        
        # 2. Pass embeddings through safety head: (N_patches, 3)
        patch_probs = self.classifier_head(embeddings)
        
        # 3. Aggregate across patches via Confidence-Weighted Max-Pooling:
        # In assault/distress scenarios, a scream may only last 1 second inside a multi-second clip.
        # Max-pooling ensures transient high-threat signatures dominate the decision.
        clip_prob = tf.reduce_max(patch_probs, axis=0)
        
        # Re-normalize to ensure sum to 1.0
        normalized_prob = clip_prob / tf.reduce_sum(clip_prob)
        
        return {
            "probabilities": normalized_prob,
            "patch_probabilities": patch_probs,
        }
