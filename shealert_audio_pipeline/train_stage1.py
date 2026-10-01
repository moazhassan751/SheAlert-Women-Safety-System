"""SheAlert Stage 1 5-Fold Cross-Validation Training Engine (Module 4).

Trains and evaluates the customized 3-class safety classifier head on YAMNet
embeddings across all 5 balanced, zero-leakage folds from splits.csv.

Features:
- Dual Embedding Representation: Combines mean-pooled (background/tone) and
  max-pooled (transient scream peak) embeddings into a 2048-dim feature vector.
- Cost-Sensitive Loss: Applies safety-weighted cross-entropy (w_Distress = 1.2)
  to aggressively penalize False Negatives (missed emergency events).
- Comprehensive Metrics: Calculates per-fold Confusion Matrix, Distress Recall,
  Aggression Recall, Normal Precision, Macro F1, and cross-fold variance.
"""

import sys
import os
import argparse
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, f1_score, recall_score, precision_score

import config
from models.yamnet_classifier import (
    build_classifier_head,
    LABEL_TO_INDEX,
    INDEX_TO_LABEL,
    NUM_CLASSES,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train Stage 1 YAMNet safety head across 5 folds.")
    parser.add_argument(
        "--cached-dir",
        type=Path,
        default=config.OUTPUT_DIR / "cached_embeddings",
        help="Directory containing per-fold .npz embedding files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=config.OUTPUT_DIR / "stage1_metrics",
        help="Directory to save evaluation reports and confusion matrices.",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=config.OUTPUT_DIR / "models",
        help="Directory to save trained Keras model checkpoints.",
    )
    parser.add_argument(
        "--feature-type",
        choices=["concat", "mean", "max"],
        default="concat",
        help="Embedding pooling type: 'concat' (mean+max, 2048-d), 'mean' (1024-d), or 'max' (1024-d).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=40,
        help="Maximum training epochs per fold (early stopping will halt earlier).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for training.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Initial learning rate for Adam optimizer.",
    )
    parser.add_argument(
        "--distress-weight",
        type=float,
        default=1.2,
        help="Loss multiplier for Distress class to prioritize high recall.",
    )
    return parser.parse_args()


def load_fold_data(npz_path: Path, feature_type: str = "concat") -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Loads embedding data from a fold .npz file."""
    if not npz_path.exists():
        raise FileNotFoundError(f"Cached fold file not found: {npz_path}")

    data = np.load(npz_path, allow_pickle=True)
    mean_embs = data["mean_embeddings"]
    max_embs = data["max_embeddings"]
    labels = data["labels"]

    if feature_type == "concat":
        features = np.concatenate([mean_embs, max_embs], axis=1)
    elif feature_type == "mean":
        features = mean_embs
    elif feature_type == "max":
        features = max_embs
    else:
        raise ValueError(f"Unknown feature_type: {feature_type}")

    meta = {
        "filepaths": data["filepaths"],
        "group_ids": data["group_ids"],
        "source_datasets": data["source_datasets"],
    }
    return features.astype(np.float32), labels.astype(np.int32), meta


def build_head_for_features(feature_dim: int, num_classes: int = NUM_CLASSES) -> tf.keras.Model:
    """Builds MLP classifier head matching the feature dimension."""
    inputs = tf.keras.Input(shape=(feature_dim,), name="feature_input")
    x = tf.keras.layers.Dense(256, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(1e-4))(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(64, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="safety_probabilities")(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs)


def train_single_fold(
    val_fold_idx: int,
    all_folds_data: Dict[int, Tuple[np.ndarray, np.ndarray]],
    args,
) -> Dict[str, Any]:
    """Trains on 4 folds, validates on val_fold_idx, and returns performance metrics."""
    # Assemble Train Set (all j != val_fold_idx)
    X_train_list, y_train_list = [], []
    for j in range(5):
        if j != val_fold_idx:
            X_train_list.append(all_folds_data[j][0])
            y_train_list.append(all_folds_data[j][1])

    X_train = np.concatenate(X_train_list, axis=0)
    y_train = np.concatenate(y_train_list, axis=0)

    # Validation Set
    X_val, y_val = all_folds_data[val_fold_idx]

    feature_dim = X_train.shape[1]
    model = build_head_for_features(feature_dim, NUM_CLASSES)

    # Class weights to prioritize Distress Recall
    class_weights = {
        0: args.distress_weight,  # Distress (penalize missed distress heavily)
        1: 1.0,                   # Aggression
        2: 1.0,                   # Normal
    }

    optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
    model.compile(
        optimizer=optimizer,
        loss=loss_fn,
        metrics=["accuracy"],
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=8,
            restore_best_weights=True,
            verbose=0,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=0,
        ),
    ]

    # Convert to one-hot for sample weight computation if needed or use class_weight dict
    model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=0,
    )

    # Evaluate on Validation Fold
    val_probs = model.predict(X_val, batch_size=args.batch_size, verbose=0)
    val_preds = np.argmax(val_probs, axis=1)

    macro_f1 = float(f1_score(y_val, val_preds, average="macro"))
    weighted_f1 = float(f1_score(y_val, val_preds, average="weighted"))
    distress_recall = float(recall_score(y_val, val_preds, labels=[0], average=None)[0])
    aggression_recall = float(recall_score(y_val, val_preds, labels=[1], average=None)[0])
    normal_precision = float(precision_score(y_val, val_preds, labels=[2], average=None)[0])
    accuracy = float(np.mean(val_preds == y_val))
    conf_mat = confusion_matrix(y_val, val_preds, labels=[0, 1, 2]).tolist()

    report = classification_report(
        y_val,
        val_preds,
        target_names=["Distress", "Aggression", "Normal"],
        output_dict=True,
    )

    return {
        "val_fold": val_fold_idx,
        "model": model,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "distress_recall": distress_recall,
        "aggression_recall": aggression_recall,
        "normal_precision": normal_precision,
        "accuracy": accuracy,
        "confusion_matrix": conf_mat,
        "report": report,
    }


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.models_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("SheAlert Stage 1: 5-Fold Cross-Validation YAMNet Safety Classifier")
    print(f"Feature Type   : {args.feature_type.upper()}")
    print(f"Distress Weight: {args.distress_weight}")
    print("=" * 65)

    # 1. Check cached folds
    all_folds_data = {}
    for i in range(5):
        fold_file = args.cached_dir / f"fold_{i}.npz"
        if not fold_file.exists():
            print(f"Error: Missing {fold_file}. Please run cache_embeddings.py first!")
            sys.exit(1)
        X, y, meta = load_fold_data(fold_file, feature_type=args.feature_type)
        all_folds_data[i] = (X, y)
        print(f"Loaded Fold {i}: {len(X)} clips | Features: {X.shape[1]}-d")

    # 2. Train 5 folds
    fold_results = []
    best_f1 = -1.0
    best_model = None
    best_fold_idx = -1

    for fold_idx in range(5):
        t0 = time.time()
        print(f"\n--- Training Fold {fold_idx} (Val Set: Fold {fold_idx}) ---")
        res = train_single_fold(fold_idx, all_folds_data, args)
        elapsed = time.time() - t0

        print(f"Fold {fold_idx} Finished in {elapsed:.1f}s:")
        print(f"  Macro F1          : {res['macro_f1']*100:.2f}%")
        print(f"  Distress Recall   : {res['distress_recall']*100:.2f}% (Safety Critical)")
        print(f"  Aggression Recall : {res['aggression_recall']*100:.2f}%")
        print(f"  Normal Precision  : {res['normal_precision']*100:.2f}%")
        print(f"  Overall Accuracy  : {res['accuracy']*100:.2f}%")

        if res["macro_f1"] > best_f1:
            best_f1 = res["macro_f1"]
            best_model = res["model"]
            best_fold_idx = fold_idx

        # Strip model instance before serialization
        clean_res = {k: v for k, v in res.items() if k != "model"}
        fold_results.append(clean_res)

    # 3. Aggregate 5-Fold Cross-Validation Metrics
    macro_f1s = [r["macro_f1"] for r in fold_results]
    distress_recalls = [r["distress_recall"] for r in fold_results]
    aggression_recalls = [r["aggression_recall"] for r in fold_results]
    normal_precisions = [r["normal_precision"] for r in fold_results]
    accuracies = [r["accuracy"] for r in fold_results]

    mean_f1, std_f1 = np.mean(macro_f1s), np.std(macro_f1s)
    mean_d_rec, std_d_rec = np.mean(distress_recalls), np.std(distress_recalls)
    mean_a_rec, std_a_rec = np.mean(aggression_recalls), np.std(aggression_recalls)
    mean_n_prec, std_n_prec = np.mean(normal_precisions), np.std(normal_precisions)
    mean_acc, std_acc = np.mean(accuracies), np.std(accuracies)

    summary_text = f"""SheAlert Stage 1 Audio Classifier — 5-Fold Cross-Validation Summary
========================================================================
Generated Date        : {time.strftime('%Y-%m-%d %H:%M:%S')}
Feature Representation: {args.feature_type.upper()} ({all_folds_data[0][0].shape[1]}-dimensional)
Distress Class Weight : {args.distress_weight}
Total Training Clips  : {sum(len(all_folds_data[i][0]) for i in range(5))} across 5 folds

5-Fold Cross-Validation Results:
  Macro F1-Score      : {mean_f1*100:.2f}% (+/- {std_f1*100:.2f}%)
  Distress Recall     : {mean_d_rec*100:.2f}% (+/- {std_d_rec*100:.2f}%)  <-- Critical Safety Metric
  Aggression Recall   : {mean_a_rec*100:.2f}% (+/- {std_a_rec*100:.2f}%)
  Normal Precision    : {mean_n_prec*100:.2f}% (+/- {std_n_prec*100:.2f}%)
  Overall Accuracy    : {mean_acc*100:.2f}% (+/- {std_acc*100:.2f}%)

Per-Fold Macro F1 Breakdown:
  Fold 0 : {macro_f1s[0]*100:.2f}%
  Fold 1 : {macro_f1s[1]*100:.2f}%
  Fold 2 : {macro_f1s[2]*100:.2f}%
  Fold 3 : {macro_f1s[3]*100:.2f}%
  Fold 4 : {macro_f1s[4]*100:.2f}%

Best Performing Fold  : Fold {best_fold_idx} (Macro F1: {best_f1*100:.2f}%)
"""
    print("\n" + summary_text)

    # Save summary report and JSON
    summary_path = args.output_dir / "summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary_text)

    json_path = args.output_dir / "5fold_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "mean_macro_f1": mean_f1,
                "std_macro_f1": std_f1,
                "mean_distress_recall": mean_d_rec,
                "std_distress_recall": std_d_rec,
                "mean_aggression_recall": mean_a_rec,
                "mean_normal_precision": mean_n_prec,
                "mean_accuracy": mean_acc,
                "fold_results": fold_results,
            },
            f,
            indent=2,
        )

    # Save Best Model Checkpoint
    best_model_path = args.models_dir / "stage1_head_best.keras"
    best_model.save(str(best_model_path))
    print(f"Saved Best Model checkpoint to {best_model_path}")
    print(f"Saved metrics summary to {summary_path}")


if __name__ == "__main__":
    main()
