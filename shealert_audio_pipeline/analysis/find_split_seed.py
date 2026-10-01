"""
Search for an optimal random seed for StratifiedGroupKFold on SheAlert training manifest.
Evaluates seeds 0 to 499 on non-OOD training clips:
- size_dev: max over folds of |fold_size - mean_size| / mean_size
- class_dev: max over folds and classes of |class_share_in_fold - class_share_overall| (in percentage points)
- missing: number of (fold, dataset) pairs where a non-OOD dataset has 0 clips
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

def evaluate_seeds(
    training_manifest_path: Path = config.OUTPUT_TRAINING_MANIFEST,
    max_seeds: int = 500,
    ood_dataset: str = config.OOD_DATASET,
    n_splits: int = 5,
):
    df = pd.read_csv(training_manifest_path)
    # Filter non-OOD
    dev_df = df[df["source_dataset"] != ood_dataset].copy()
    dev_df = dev_df.sort_values("filepath", ascending=True).reset_index(drop=True)

    n_samples = len(dev_df)
    mean_fold_size = n_samples / n_splits
    non_ood_datasets = sorted(dev_df["source_dataset"].unique())
    classes = sorted(dev_df["label"].unique())

    overall_class_counts = dev_df["label"].value_counts()
    overall_class_shares = {c: overall_class_counts[c] / n_samples for c in classes}

    print(f"Total non-OOD training samples: {n_samples}")
    print(f"Mean fold size: {mean_fold_size:.1f}")
    print(f"Non-OOD datasets ({len(non_ood_datasets)}): {non_ood_datasets}")
    print("Overall class shares:")
    for c in classes:
        print(f"  {c}: {overall_class_shares[c]:.2%}")

    results = []

    for seed in range(max_seeds):
        sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        fold_sizes = []
        max_c_dev = 0.0
        missing_count = 0

        for fold_idx, (trn_idx, val_idx) in enumerate(
            sgkf.split(dev_df, y=dev_df["label"], groups=dev_df["group_id"])
        ):
            val_df = dev_df.iloc[val_idx]
            fold_len = len(val_df)
            fold_sizes.append(fold_len)

            # Class deviations (percentage points)
            val_class_counts = val_df["label"].value_counts()
            for c in classes:
                val_share = val_class_counts.get(c, 0) / fold_len if fold_len else 0
                c_dev = abs(val_share - overall_class_shares[c]) * 100.0
                if c_dev > max_c_dev:
                    max_c_dev = c_dev

            # Dataset representation
            val_datasets = set(val_df["source_dataset"].unique())
            for d in non_ood_datasets:
                if d not in val_datasets:
                    missing_count += 1

        size_devs = [abs(s - mean_fold_size) / mean_fold_size for s in fold_sizes]
        max_size_dev = max(size_devs)

        results.append({
            "seed": seed,
            "size_dev": max_size_dev,
            "class_dev": max_c_dev,
            "missing": missing_count,
            "fold_sizes": fold_sizes,
        })

    results_df = pd.DataFrame(results)

    # Filter missing == 0
    zero_missing = results_df[results_df["missing"] == 0].copy()
    zero_missing = zero_missing.sort_values(by=["size_dev", "class_dev"], ascending=[True, True]).reset_index(drop=True)

    print(f"\nTotal seeds evaluated: {len(results_df)}")
    print(f"Seeds with missing == 0: {len(zero_missing)}")

    # Target constraints: size_dev <= 0.15 and class_dev <= 5.0
    qualified = zero_missing[(zero_missing["size_dev"] <= 0.15) & (zero_missing["class_dev"] <= 5.0)]
    print(f"Seeds meeting hard targets (size_dev <= 0.15, class_dev <= 5.0 points): {len(qualified)}")

    print("\n=== TOP 5 CANDIDATE SEEDS (missing == 0, min size_dev, min class_dev) ===")
    top5 = zero_missing.head(5)
    print(f"{'Rank':4s} | {'Seed':5s} | {'size_dev':9s} | {'class_dev':14s} | {'missing':7s} | {'fold_sizes'}")
    print("-" * 75)
    for i, row in top5.iterrows():
        sizes_str = ", ".join(str(s) for s in row["fold_sizes"])
        print(f"{i+1:4d} | {row['seed']:5d} | {row['size_dev']:9.4f} | {row['class_dev']:14.2f} | {row['missing']:7d} | [{sizes_str}]")

    print("\n=== ALL ZERO-MISSING SEEDS ===")
    for i, row in zero_missing.iterrows():
        sizes_str = ", ".join(str(s) for s in row["fold_sizes"])
        print(f"{i+1:4d} | Seed {row['seed']:5d} | size_dev: {row['size_dev']:.4f} | class_dev: {row['class_dev']:.2f} pp | missing: {row['missing']} | [{sizes_str}]")

    print("\n=== GLOBAL METRICS ACROSS ALL 500 SEEDS ===")
    print(f"Min size_dev overall (any missing): {results_df['size_dev'].min():.4f}")
    print(f"Min class_dev overall (any missing): {results_df['class_dev'].min():.2f} pp")
    print(f"Seeds with size_dev <= 0.15 (any missing): {(results_df['size_dev'] <= 0.15).sum()}")
    print(f"Seeds with class_dev <= 5.0 pp (any missing): {(results_df['class_dev'] <= 5.0).sum()}")

    return results_df, zero_missing, qualified

if __name__ == "__main__":
    evaluate_seeds()
