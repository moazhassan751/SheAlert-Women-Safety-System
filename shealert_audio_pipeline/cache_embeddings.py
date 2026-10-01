"""SheAlert Offline YAMNet Feature Extraction & Embedding Caching Engine.

Iterates over splits.csv (22,368 training/validation clips + 77 OOD clips),
extracts 1024-dimensional YAMNet embedding vectors, and saves them to
compressed .npz binary archives partitioned by fold.

Caching embeddings accelerates subsequent 5-fold cross-validation and hyperparameter
tuning by 10x-20x without repetitive spectrogram computation.
"""

import sys
import os
import argparse
import time
from pathlib import Path
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd
import soundfile as sf
import librosa
import tensorflow as tf
from tqdm import tqdm

import config
from utils import resolve_relative_path
from models.yamnet_classifier import load_yamnet_backbone, LABEL_TO_INDEX, NUM_CLASSES, EMBEDDING_DIM


def parse_args():
    parser = argparse.ArgumentParser(description="Cache YAMNet embeddings for SheAlert audio splits.")
    parser.add_argument(
        "--splits-file",
        type=Path,
        default=config.OUTPUT_DIR / "splits.csv",
        help="Path to splits.csv containing fold assignments.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=config.OUTPUT_DIR / "cached_embeddings",
        help="Directory where per-fold .npz files will be saved.",
    )
    parser.add_argument(
        "--folds",
        nargs="+",
        default=["0", "1", "2", "3", "4", "ood"],
        help="List of folds to process (default: 0 1 2 3 4 ood).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing .npz files.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on number of clips per fold (for smoke testing).",
    )
    return parser.parse_args()


def load_and_preprocess_audio(filepath: Path, target_sr: int = 16000) -> np.ndarray:
    """Loads audio, ensures mono, resamples to target_sr, pads if short, and normalizes."""
    try:
        y, sr = sf.read(str(filepath), dtype="float32")
        if y.ndim > 1:
            y = np.mean(y, axis=1)
        if sr != target_sr:
            y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
    except Exception:
        # Fallback for MP3 or non-standard headers
        y, _ = librosa.load(str(filepath), sr=target_sr, mono=True)

    # If shorter than 1 second (16000 samples), pad with trailing zeros
    min_samples = target_sr
    if len(y) < min_samples:
        y = np.pad(y, (0, min_samples - len(y)), mode="constant")

    # Peak normalization
    max_val = np.max(np.abs(y))
    if max_val > 1e-6:
        y = y / max_val

    return y.astype(np.float32)


def process_fold(
    fold_name: str,
    df_fold: pd.DataFrame,
    yamnet_model: Any,
    output_path: Path,
    limit: Optional[int] = None,
):
    """Processes all clips in a single fold and writes output_path.npz."""
    if limit is not None:
        df_fold = df_fold.head(limit)

    total_clips = len(df_fold)
    print(f"\n[Fold {fold_name}] Extracting embeddings for {total_clips} clips...")

    mean_embs: List[np.ndarray] = []
    max_embs: List[np.ndarray] = []
    labels: List[int] = []
    filepaths: List[str] = []
    group_ids: List[str] = []
    source_datasets: List[str] = []
    durations: List[float] = []

    start_time = time.time()
    skipped_count = 0

    for idx, row in tqdm(df_fold.iterrows(), total=total_clips, desc=f"Fold {fold_name}", ncols=90):
        rel_path = row["filepath"]
        abs_path = resolve_relative_path(rel_path)

        if not abs_path.exists():
            print(f"Warning: File not found: {abs_path}")
            skipped_count += 1
            continue

        try:
            waveform = load_and_preprocess_audio(abs_path)
            # Pass to YAMNet (expects float32 1-D tensor)
            wav_tensor = tf.constant(waveform, dtype=tf.float32)
            _, embeddings, _ = yamnet_model(wav_tensor)
            emb_np = embeddings.numpy()  # Shape: (N_patches, 1024)

            # Extract both mean-pooled and max-pooled representations across patches
            mean_emb = np.mean(emb_np, axis=0).astype(np.float32)
            max_emb = np.max(emb_np, axis=0).astype(np.float32)

            label_str = row["label"]
            label_idx = LABEL_TO_INDEX[label_str]

            mean_embs.append(mean_emb)
            max_embs.append(max_emb)
            labels.append(label_idx)
            filepaths.append(rel_path)
            group_ids.append(str(row["group_id"]))
            source_datasets.append(str(row["source_dataset"]))
            durations.append(float(row.get("duration", len(waveform) / 16000.0)))

        except Exception as e:
            print(f"Error processing {abs_path}: {e}")
            skipped_count += 1
            continue

    elapsed = time.time() - start_time
    print(f"[Fold {fold_name}] Done in {elapsed:.1f}s ({len(mean_embs)} clips saved, {skipped_count} skipped).")

    # Save to compressed .npz archive
    np.savez_compressed(
        output_path,
        mean_embeddings=np.array(mean_embs, dtype=np.float32),
        max_embeddings=np.array(max_embs, dtype=np.float32),
        labels=np.array(labels, dtype=np.int32),
        filepaths=np.array(filepaths, dtype=object),
        group_ids=np.array(group_ids, dtype=object),
        source_datasets=np.array(source_datasets, dtype=object),
        durations=np.array(durations, dtype=np.float32),
        fold=np.array([fold_name] * len(labels), dtype=object),
    )
    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"[Fold {fold_name}] Saved to {output_path} ({file_size_mb:.2f} MB)")


def main():
    args = parse_args()
    splits_file = args.splits_file
    output_dir = args.output_dir

    if not splits_file.exists():
        print(f"Error: Splits file not found at {splits_file}")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)
    df_splits = pd.read_csv(splits_file)
    print(f"Loaded splits from {splits_file}: Total rows = {len(df_splits)}")

    # Load YAMNet backbone once for the entire session
    print("Loading YAMNet model backbone...")
    yamnet_model = load_yamnet_backbone()
    print("YAMNet loaded successfully.")

    for fold_name in args.folds:
        out_npz = output_dir / f"fold_{fold_name}.npz"
        if out_npz.exists() and not args.force:
            print(f"[Fold {fold_name}] Already exists at {out_npz}. Skipping (use --force to overwrite).")
            continue

        df_fold = df_splits[df_splits["fold"].astype(str) == str(fold_name)]
        if len(df_fold) == 0:
            print(f"[Fold {fold_name}] No rows found for this fold in splits.csv!")
            continue

        process_fold(fold_name, df_fold, yamnet_model, out_npz, limit=args.limit)

    print("\nAll requested folds cached successfully.")


if __name__ == "__main__":
    main()
