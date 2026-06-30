"""
compute_class_weights.py
------------------------
Computes inverse-frequency class weights for the Cityscapes segmentation head.

Cityscapes GT labels are in the range [-1, 18]:
  -1 = void / unlabelled
   0 = road
   1 = sidewalk
   ...
  18 = bicycle

The segmentation loss shifts labels by +1 (so -1 -> 0, 0 -> 1, ..., 18 -> 19)
and uses ignore_index=0 to mask the void class.  The weights produced here
correspond to that shifted indexing, i.e., weights[i] is for shifted class i.

Usage:
    python scripts/compute_class_weights.py \
        --label_dir datasets/cityscapes_data/data/train/label

The script prints the per-class pixel frequencies and the final normalised
inverse-frequency weight tensor ready to paste into MultiScaleSegmLoss.
"""

import argparse
import os
import numpy as np


def compute_weights(label_dir: str) -> np.ndarray:
    files = sorted([f for f in os.listdir(label_dir) if f.endswith(".npy")])
    if not files:
        raise FileNotFoundError(f"No .npy files found in: {label_dir}")

    print(f"Found {len(files)} label files. Processing...")

    # 20 bins: shifted index 0 (void/GT -1) through 19 (GT 18)
    class_counts = np.zeros(20, dtype=np.int64)

    for i, fname in enumerate(files):
        data = np.load(os.path.join(label_dir, fname)).flatten().astype(np.int32)
        data_shifted = data + 1  # GT -1..18  ->  0..19
        for c in range(20):
            class_counts[c] += int(np.sum(data_shifted == c))
        if (i + 1) % 500 == 0:
            print(f"  Processed {i+1}/{len(files)}...")

    total = class_counts.sum()
    gt_labels = list(range(-1, 19))

    print("\nFull training set class frequencies (shifted index = GT label + 1):")
    print(f"{'GT label':>10} {'Shifted idx':>12} {'Frequency %':>12} {'Pixel count':>14}")
    print("-" * 52)
    for c in range(20):
        print(
            f"{gt_labels[c]:>10d} {c:>12d} "
            f"{100*class_counts[c]/total:>11.4f}% "
            f"{class_counts[c]:>14,}"
        )

    # Inverse-frequency weights for non-void classes (shifted indices 1..19)
    freqs = class_counts[1:].astype(np.float64)
    weights = 1.0 / freqs
    weights = weights / weights.mean()  # normalise so mean weight = 1

    print("\nNormalised inverse-frequency weights (GT classes 0..18 / shifted 1..19):")
    for i, w in enumerate(weights):
        print(f"  GT {i:2d} (shifted {i+1:2d}): {w:.6f}")

    print("\n# Paste this directly into MultiScaleSegmLoss (void weight prepended as 0.0):")
    inner = ", ".join(f"{w:.4f}" for w in weights)
    print(f"class_weights = torch.tensor([0.0, {inner}])")

    return weights


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute Cityscapes class weights.")
    parser.add_argument(
        "--label_dir",
        default="datasets/cityscapes_data/data/train/label",
        help="Path to directory containing training label .npy files.",
    )
    args = parser.parse_args()
    compute_weights(args.label_dir)
