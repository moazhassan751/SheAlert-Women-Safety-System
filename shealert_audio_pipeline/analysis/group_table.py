"""
Step 1: Build group table from non-OOD rows of training_manifest.csv.
Print the 15 largest groups and group count per dataset.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import config

df = pd.read_csv(config.OUTPUT_TRAINING_MANIFEST)
non_ood = df[df["source_dataset"] != config.OOD_DATASET].copy()

# Build one row per group_id
group_rows = []
for gid, sub in non_ood.groupby("group_id"):
    src = sub["source_dataset"].iloc[0]
    total = len(sub)
    d = int((sub["label"] == "Distress").sum())
    a = int((sub["label"] == "Aggression").sum())
    n = int((sub["label"] == "Normal").sum())
    group_rows.append({"group_id": gid, "source_dataset": src, "total": total,
                       "Distress": d, "Aggression": a, "Normal": n})

gt = pd.DataFrame(group_rows).sort_values("total", ascending=False).reset_index(drop=True)

print(f"Total non-OOD clips: {len(non_ood)}")
print(f"Total groups: {len(gt)}")
print(f"\n=== 15 LARGEST GROUPS ===")
print(f"{'group_id':30s} | {'dataset':20s} | {'total':6s} | {'Distress':8s} | {'Aggression':10s} | {'Normal':6s}")
print("-" * 90)
for _, row in gt.head(15).iterrows():
    print(f"{str(row['group_id']):30s} | {row['source_dataset']:20s} | {row['total']:6d} | {row['Distress']:8d} | {row['Aggression']:10d} | {row['Normal']:6d}")

print(f"\n=== GROUP COUNT PER DATASET ===")
gc = gt.groupby("source_dataset").agg(
    n_groups=("group_id", "count"),
    total_clips=("total", "sum"),
    min_size=("total", "min"),
    max_size=("total", "max"),
    mean_size=("total", "mean"),
).sort_values("total_clips", ascending=False)
print(gc.to_string())
