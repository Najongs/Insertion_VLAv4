#!/usr/bin/env python
"""
Extract action statistics from LeRobot dataset for normalization.

This script analyzes your training dataset and extracts min/max statistics
for action values, which can be used for proper normalization during inference.

Usage:
    python extract_dataset_stats.py --dataset-path /path/to/dataset

Output:
    Prints min/max statistics for each action dimension that you can copy
    into lerobot_to_MECA.py configuration.
"""
import argparse
import numpy as np
from pathlib import Path
from datasets import load_from_disk
from tqdm import tqdm


def extract_action_stats(dataset_path: str):
    """Extract min/max statistics from dataset actions.

    Args:
        dataset_path: Path to the dataset directory
    """
    print(f"Loading dataset from: {dataset_path}")
    dataset = load_from_disk(dataset_path)

    print(f"Dataset size: {len(dataset)} frames")
    print(f"Dataset features: {dataset.features}")

    # Check if 'action' column exists
    if 'action' not in dataset.features:
        print("ERROR: Dataset does not contain 'action' column")
        return

    # Collect all actions
    print("\nExtracting actions from dataset...")
    actions = []
    for sample in tqdm(dataset):
        action = sample['action']
        # Handle different action formats
        if isinstance(action, (list, tuple)):
            action = np.array(action)
        actions.append(action)

    actions = np.array(actions)
    print(f"Action shape: {actions.shape}")

    # Compute statistics
    action_min = actions.min(axis=0)
    action_max = actions.max(axis=0)
    action_mean = actions.mean(axis=0)
    action_std = actions.std(axis=0)

    # Print results
    print("\n" + "="*70)
    print("ACTION STATISTICS")
    print("="*70)
    print("\nPer-dimension statistics:")
    print(f"{'Dimension':<12} {'Min':<12} {'Max':<12} {'Mean':<12} {'Std':<12}")
    print("-"*70)
    dim_names = ['X', 'Y', 'Z', 'Rx', 'Ry', 'Rz']
    for i in range(len(action_min)):
        dim_name = dim_names[i] if i < len(dim_names) else f"Dim{i}"
        print(f"{dim_name:<12} {action_min[i]:<12.4f} {action_max[i]:<12.4f} "
              f"{action_mean[i]:<12.4f} {action_std[i]:<12.4f}")

    # Print configuration to copy
    print("\n" + "="*70)
    print("CONFIGURATION FOR lerobot_to_MECA.py")
    print("="*70)
    print("\nCopy these lines to your configuration section:")
    print("\n# Data-driven action normalization (from dataset statistics)")
    print(f"DATA_ACTION_MIN = np.array({list(action_min)})")
    print(f"DATA_ACTION_MAX = np.array({list(action_max)})")

    # Additional insights
    print("\n" + "="*70)
    print("INSIGHTS")
    print("="*70)
    action_range = action_max - action_min
    print("\nAction ranges (max - min):")
    for i, dim_name in enumerate(dim_names):
        print(f"  {dim_name}: {action_range[i]:.4f}")

    print("\nPercentage of actions near zero (within ±0.1):")
    for i, dim_name in enumerate(dim_names):
        near_zero = np.abs(actions[:, i]) < 0.1
        percentage = near_zero.sum() / len(actions) * 100
        print(f"  {dim_name}: {percentage:.1f}%")

    # Check for outliers
    print("\nOutlier detection (values beyond ±3 std from mean):")
    for i, dim_name in enumerate(dim_names):
        threshold_low = action_mean[i] - 3 * action_std[i]
        threshold_high = action_mean[i] + 3 * action_std[i]
        outliers = (actions[:, i] < threshold_low) | (actions[:, i] > threshold_high)
        num_outliers = outliers.sum()
        if num_outliers > 0:
            print(f"  {dim_name}: {num_outliers} outliers ({num_outliers/len(actions)*100:.2f}%)")

    print("\n" + "="*70)


def main():
    parser = argparse.ArgumentParser(description="Extract action statistics from LeRobot dataset")
    parser.add_argument(
        "--dataset-path",
        type=str,
        required=True,
        help="Path to the dataset directory"
    )

    args = parser.parse_args()

    dataset_path = Path(args.dataset_path)
    if not dataset_path.exists():
        print(f"ERROR: Dataset path does not exist: {dataset_path}")
        return

    extract_action_stats(str(dataset_path))


if __name__ == "__main__":
    main()
