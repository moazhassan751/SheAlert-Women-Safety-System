"""SheAlert Stage 1 Out-Of-Distribution (OOD) Evaluation (Module 4).

Evaluates the trained Stage 1 safety classifier on the held-out RealWorld-Urdu
benchmark (77 clips) to quantify zero-shot cross-corpus domain resilience.
"""

import sys
import argparse
from pathlib import Path
import json

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

import config
from train_stage1 import load_fold_data
from models.yamnet_classifier import INDEX_TO_LABEL


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Stage 1 model on OOD RealWorld-Urdu dataset.")
    parser.add_argument(
        "--model-path",
        type=Path,
        default=config.OUTPUT_DIR / "models" / "stage1_head_best.keras",
        help="Path to trained Keras safety head model.",
    )
    parser.add_argument(
        "--ood-file",
        type=Path,
        default=config.OUTPUT_DIR / "cached_embeddings" / "fold_ood.npz",
        help="Path to cached fold_ood.npz file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=config.OUTPUT_DIR / "stage1_metrics",
        help="Directory to save OOD evaluation report.",
    )
    parser.add_argument(
        "--feature-type",
        choices=["concat", "mean", "max"],
        default="concat",
        help="Feature pooling type (must match training feature type).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if not args.model_path.exists():
        print(f"Error: Model not found at {args.model_path}. Train a model first via train_stage1.py!")
        sys.exit(1)

    if not args.ood_file.exists():
        print(f"Error: OOD cached file not found at {args.ood_file}. Run cache_embeddings.py first!")
        sys.exit(1)

    print("=" * 65)
    print("SheAlert Stage 1: Out-of-Distribution (OOD) Benchmark Evaluation")
    print(f"Model Path  : {args.model_path.name}")
    print(f"OOD Dataset : RealWorld-Urdu ({args.ood_file.name})")
    print("=" * 65)

    # 1. Load Model
    model = tf.keras.models.load_model(str(args.model_path))

    # 2. Load OOD Data
    X_ood, y_ood, meta = load_fold_data(args.ood_file, feature_type=args.feature_type)
    print(f"Loaded OOD Set: {len(X_ood)} clips | Features: {X_ood.shape[1]}-d")

    # 3. Predict
    probs = model.predict(X_ood, verbose=0)
    preds = np.argmax(probs, axis=1)

    # 4. Metrics
    present_classes = sorted(list(set(y_ood)))
    present_names = [INDEX_TO_LABEL[c] for c in present_classes]

    report = classification_report(
        y_ood,
        preds,
        labels=present_classes,
        target_names=present_names,
        zero_division=0,
    )
    acc = accuracy_score(y_ood, preds)
    conf_mat = confusion_matrix(y_ood, preds, labels=[0, 1, 2]).tolist()

    report_text = f"""SheAlert Stage 1 — Out-Of-Distribution (OOD) Benchmark Report
========================================================================
Evaluation Dataset    : RealWorld-Urdu (Held-out Field Recordings)
Evaluated Model       : {args.model_path.name}
Total OOD Samples     : {len(y_ood)} clips
Accuracy              : {acc*100:.2f}%

Classification Report:
{report}

Full 3x3 Confusion Matrix (Rows=True, Cols=Pred: [0=Distress, 1=Aggression, 2=Normal]):
  Distress (0)   : {conf_mat[0]}  (Note: RealWorld-Urdu contains 0 distress clips)
  Aggression (1) : {conf_mat[1]}
  Normal (2)     : {conf_mat[2]}

Analysis & Domain Shift Observations:
- RealWorld-Urdu represents unconstrained smartphone field recordings from prior student work.
- It tests zero-shot robustness against acoustic room reverberation, differing phone mic frequency responses, and ambient hiss without prior exposure.
"""
    print("\n" + report_text)

    out_file = args.output_dir / "ood_evaluation.txt"
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"Saved OOD report to {out_file}")


if __name__ == "__main__":
    main()
