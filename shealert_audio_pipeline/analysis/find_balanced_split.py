"""
find_balanced_split.py
-----------------------
Assign groups to 5 folds using greedy smallest-bin-first placement + local
search (single moves and pairwise swaps).

Evaluates seeds 0 to 199. For each seed:
  1. Shuffle groups, sort largest-first (stable sort preserves random tie order).
  2. Place each group into the fold with the smallest current total clips.
     Break ties by which fold has class proportions closest to the overall.
  3. Local search: try moving one group to another fold, or swapping two
     groups between folds. Accept any change that lowers
     cost = size_dev + class_dev/100. Repeat until no improvement
     (max 2000 iterations).
  4. Compute final metrics: size_dev, class_dev (pp), missing (fold, dataset).
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import config

CLASSES = ["Distress", "Aggression", "Normal"]


def build_group_table(training_manifest_path=None, ood_dataset=None):
    """Build a group table from non-OOD rows of training_manifest.csv."""
    if training_manifest_path is None:
        training_manifest_path = config.OUTPUT_TRAINING_MANIFEST
    if ood_dataset is None:
        ood_dataset = config.OOD_DATASET

    df = pd.read_csv(training_manifest_path)
    non_ood = df[df["source_dataset"] != ood_dataset].copy()

    group_rows = []
    for gid, sub in non_ood.groupby("group_id"):
        src = sub["source_dataset"].iloc[0]
        total = len(sub)
        d = int((sub["label"] == "Distress").sum())
        a = int((sub["label"] == "Aggression").sum())
        n = int((sub["label"] == "Normal").sum())
        group_rows.append({
            "group_id": gid, "source_dataset": src, "total": total,
            "Distress": d, "Aggression": a, "Normal": n,
        })

    gt = pd.DataFrame(group_rows)
    return gt, non_ood


def _class_imbalance(fold_class_counts, fold_size, overall_class_shares):
    """Sum of squared class-share deviations for one fold."""
    if fold_size == 0:
        return 999.0
    total = 0.0
    for c in CLASSES:
        fold_share = fold_class_counts.get(c, 0) / fold_size
        total += (fold_share - overall_class_shares[c]) ** 2
    return total


def compute_metrics(fold_sizes, fold_class_counts, fold_dataset_sets,
                    overall_class_shares, non_ood_datasets, n_splits=5):
    """Compute size_dev, class_dev (pp), and missing count."""
    total = sum(fold_sizes)
    mean_sz = total / n_splits

    size_dev = max(abs(s - mean_sz) / mean_sz for s in fold_sizes) if mean_sz > 0 else 999.0

    class_dev = 0.0
    for i in range(n_splits):
        fs = fold_sizes[i]
        if fs == 0:
            return 999.0, 999.0, 999
        for c in CLASSES:
            fold_share = fold_class_counts[i].get(c, 0) / fs
            dev = abs(fold_share - overall_class_shares[c]) * 100.0
            if dev > class_dev:
                class_dev = dev

    missing = 0
    for i in range(n_splits):
        for d in non_ood_datasets:
            if d not in fold_dataset_sets[i]:
                missing += 1

    return size_dev, class_dev, missing


def cost_fn(size_dev, class_dev):
    """Cost for placement and local search."""
    return size_dev + class_dev / 100.0


def assign_balanced_split(gt, seed, n_splits=5, max_local_iters=2000):
    """
    Assign groups to folds using greedy smallest-bin-first + local search.
    Returns (assignment dict {group_id: fold}, size_dev, class_dev, missing, fold_sizes).
    """
    rng = np.random.RandomState(seed)

    total_clips = gt["total"].sum()
    overall_class_shares = {c: gt[c].sum() / total_clips for c in CLASSES}
    non_ood_datasets = sorted(gt["source_dataset"].unique())
    mean_sz = total_clips / n_splits

    # Shuffle then stable-sort descending by total (random tie-breaking)
    idx = rng.permutation(len(gt))
    gt_shuffled = gt.iloc[idx].sort_values("total", ascending=False, kind="mergesort").reset_index(drop=True)

    groups = gt_shuffled.to_dict("records")
    n_groups = len(groups)

    # Initialise fold state
    fold_sizes = [0] * n_splits
    fold_class_counts = [{c: 0 for c in CLASSES} for _ in range(n_splits)]
    fold_dataset_sets = [set() for _ in range(n_splits)]
    assignment = [0] * n_groups

    # --- Greedy placement: smallest-bin-first ---
    for gi, g in enumerate(groups):
        # Find the fold(s) with the smallest current size
        min_sz = min(fold_sizes)
        candidates = [fi for fi in range(n_splits) if fold_sizes[fi] == min_sz]

        if len(candidates) == 1:
            best_fold = candidates[0]
        else:
            # Break ties by class imbalance after tentative placement
            best_fold = candidates[0]
            best_imb = float("inf")
            for fi in candidates:
                trial_cc = {**fold_class_counts[fi]}
                trial_sz = fold_sizes[fi] + g["total"]
                for c in CLASSES:
                    trial_cc[c] += g[c]
                imb = _class_imbalance(trial_cc, trial_sz, overall_class_shares)
                if imb < best_imb:
                    best_imb = imb
                    best_fold = fi

        assignment[gi] = best_fold
        fold_sizes[best_fold] += g["total"]
        for c in CLASSES:
            fold_class_counts[best_fold][c] += g[c]
        fold_dataset_sets[best_fold].add(g["source_dataset"])

    # Current cost
    cur_sd, cur_cd, _ = compute_metrics(
        fold_sizes, fold_class_counts, fold_dataset_sets,
        overall_class_shares, non_ood_datasets, n_splits)
    cur_cost = cost_fn(cur_sd, cur_cd)

    def _rebuild_dataset_sets():
        ds = [set() for _ in range(n_splits)]
        for gi2 in range(n_groups):
            ds[assignment[gi2]].add(groups[gi2]["source_dataset"])
        return ds

    # --- Local search ---
    improved = True
    iters = 0
    while improved and iters < max_local_iters:
        improved = False
        iters += 1

        # Try single moves
        for gi in range(n_groups):
            g = groups[gi]
            old_fold = assignment[gi]
            best_new_fold = -1
            best_new_cost = cur_cost

            for new_fold in range(n_splits):
                if new_fold == old_fold:
                    continue
                new_old_sz = fold_sizes[old_fold] - g["total"]
                new_new_sz = fold_sizes[new_fold] + g["total"]
                if new_old_sz <= 0:
                    continue

                # Quick size_dev check
                trial_sizes = fold_sizes[:]
                trial_sizes[old_fold] = new_old_sz
                trial_sizes[new_fold] = new_new_sz
                sd = max(abs(s - mean_sz) / mean_sz for s in trial_sizes)

                trial_cc = [{**cc} for cc in fold_class_counts]
                for c in CLASSES:
                    trial_cc[old_fold][c] -= g[c]
                    trial_cc[new_fold][c] += g[c]

                cd = 0.0
                for fj in range(n_splits):
                    fs = trial_sizes[fj]
                    if fs <= 0:
                        cd = 999.0
                        break
                    for c2 in CLASSES:
                        dev = abs(trial_cc[fj][c2] / fs - overall_class_shares[c2]) * 100.0
                        if dev > cd:
                            cd = dev

                trial_cost = cost_fn(sd, cd)
                if trial_cost < best_new_cost - 1e-9:
                    best_new_cost = trial_cost
                    best_new_fold = new_fold

            if best_new_fold >= 0:
                # Apply best move
                old_f = assignment[gi]
                fold_sizes[old_f] -= g["total"]
                fold_sizes[best_new_fold] += g["total"]
                for c in CLASSES:
                    fold_class_counts[old_f][c] -= g[c]
                    fold_class_counts[best_new_fold][c] += g[c]
                assignment[gi] = best_new_fold
                fold_dataset_sets = _rebuild_dataset_sets()
                cur_cost = best_new_cost
                improved = True
                break  # restart

        if improved:
            continue

        # Try pairwise swaps
        for gi in range(n_groups):
            if improved:
                break
            ga = groups[gi]
            fi = assignment[gi]
            for gj in range(gi + 1, n_groups):
                gb = groups[gj]
                fj_idx = assignment[gj]
                if fi == fj_idx:
                    continue

                trial_sizes = fold_sizes[:]
                trial_sizes[fi] = trial_sizes[fi] - ga["total"] + gb["total"]
                trial_sizes[fj_idx] = trial_sizes[fj_idx] - gb["total"] + ga["total"]

                if any(s <= 0 for s in trial_sizes):
                    continue

                trial_cc = [{**cc} for cc in fold_class_counts]
                for c in CLASSES:
                    trial_cc[fi][c] = trial_cc[fi][c] - ga[c] + gb[c]
                    trial_cc[fj_idx][c] = trial_cc[fj_idx][c] - gb[c] + ga[c]

                sd = max(abs(s - mean_sz) / mean_sz for s in trial_sizes)
                cd = 0.0
                for fk in range(n_splits):
                    fs = trial_sizes[fk]
                    for c2 in CLASSES:
                        dev = abs(trial_cc[fk][c2] / fs - overall_class_shares[c2]) * 100.0
                        if dev > cd:
                            cd = dev

                trial_cost = cost_fn(sd, cd)
                if trial_cost < cur_cost - 1e-9:
                    fold_sizes = trial_sizes
                    fold_class_counts = trial_cc
                    assignment[gi] = fj_idx
                    assignment[gj] = fi
                    fold_dataset_sets = _rebuild_dataset_sets()
                    cur_cost = trial_cost
                    improved = True
                    break

    # Final metrics
    fold_dataset_sets = _rebuild_dataset_sets()
    final_sd, final_cd, final_missing = compute_metrics(
        fold_sizes, fold_class_counts, fold_dataset_sets,
        overall_class_shares, non_ood_datasets, n_splits)

    result = {}
    for gi, g in enumerate(groups):
        result[g["group_id"]] = assignment[gi]

    return result, final_sd, final_cd, final_missing, fold_sizes


def search_seeds(max_seeds=200, n_splits=5):
    gt, non_ood = build_group_table()
    total_clips = gt["total"].sum()
    overall_class_shares = {c: gt[c].sum() / total_clips for c in CLASSES}
    non_ood_datasets = sorted(gt["source_dataset"].unique())

    print(f"Total non-OOD clips: {total_clips}")
    print(f"Total groups: {len(gt)}")
    print(f"Mean fold size: {total_clips / n_splits:.1f}")
    print(f"15% of mean: {total_clips / n_splits * 0.15:.1f}")
    print(f"Non-OOD datasets ({len(non_ood_datasets)}): {non_ood_datasets}")
    print("Overall class shares:")
    for c in CLASSES:
        print(f"  {c}: {overall_class_shares[c]:.2%}")

    results = []
    for seed in range(max_seeds):
        asgn, sd, cd, missing, fold_sizes = assign_balanced_split(gt, seed, n_splits)
        results.append({
            "seed": seed, "size_dev": sd, "class_dev": cd,
            "missing": missing, "fold_sizes": fold_sizes,
        })
        if seed % 20 == 0:
            print(f"  Seed {seed:4d}: size_dev={sd:.4f}, class_dev={cd:.2f} pp, missing={missing}, folds={fold_sizes}")

    results_df = pd.DataFrame(results)

    zero_missing = results_df[results_df["missing"] == 0].copy()
    zero_missing = zero_missing.sort_values(by=["size_dev", "class_dev"]).reset_index(drop=True)

    qualified = zero_missing[(zero_missing["size_dev"] <= 0.15) & (zero_missing["class_dev"] <= 5.0)]

    print(f"\nTotal seeds evaluated: {len(results_df)}")
    print(f"Seeds with missing == 0: {len(zero_missing)}")
    print(f"Seeds meeting hard targets (size_dev <= 0.15, class_dev <= 5.0 pp): {len(qualified)}")

    print("\n=== TOP 5 CANDIDATE SEEDS (missing == 0) ===")
    top5 = zero_missing.head(5)
    print(f"{'Rank':4s} | {'Seed':5s} | {'size_dev':9s} | {'class_dev':10s} | {'missing':7s} | fold_sizes")
    print("-" * 80)
    for rank, (_, row) in enumerate(top5.iterrows(), 1):
        sizes_str = ", ".join(str(s) for s in row["fold_sizes"])
        print(f"{rank:4d} | {row['seed']:5.0f} | {row['size_dev']:9.4f} | {row['class_dev']:10.2f} | {row['missing']:7.0f} | [{sizes_str}]")

    if len(qualified) > 0:
        best = qualified.iloc[0]
        print(f"\nBest qualified seed: {int(best['seed'])} "
              f"(size_dev={best['size_dev']:.4f}, class_dev={best['class_dev']:.2f} pp, "
              f"missing={int(best['missing'])})")
    else:
        print("\nNo seed met the hard targets. Review the top 5 above and decide.")

    return results_df, zero_missing, qualified


if __name__ == "__main__":
    search_seeds()
