#!/usr/bin/env python
"""
Generate action_comparison plot from existing CSV file.

Usage:
python plot_action_comparison_from_csv.py \
    --csv outputs/episode_eval_ep4/frame_by_frame_comparison.csv \
    --output outputs/episode_eval_ep4/action_comparison_new.png \
    --max_frames 300
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def plot_action_comparison(csv_path, output_path, max_frames=300):
    """Generate action comparison plot from CSV file."""

    # Read CSV
    print(f"Reading CSV from: {csv_path}")
    df = pd.read_csv(csv_path)

    print(f"Total frames in CSV: {len(df)}")

    # Limit to max_frames
    df = df.head(max_frames)
    print(f"Plotting first {len(df)} frames")

    # Extract data
    frames = df['Frame'].values

    dim_names = ['dx', 'dy', 'dz', 'drx', 'dry', 'drz']

    # Create figure
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    fig.suptitle(f'Ground Truth vs Predicted Actions (First {len(df)} Frames)', fontsize=16)

    for i, dim_name in enumerate(dim_names):
        row = i // 2
        col = i % 2

        # Get GT and Pred columns
        gt_col = f'GT_{dim_name}'
        pred_col = f'Pred_{dim_name}'

        if gt_col not in df.columns or pred_col not in df.columns:
            print(f"Warning: Missing columns for {dim_name}")
            continue

        gt_values = df[gt_col].values
        pred_values = df[pred_col].values

        # Plot
        axes[row, col].plot(frames, gt_values, label='Ground Truth', linewidth=2, alpha=0.7, color='blue')
        axes[row, col].plot(frames, pred_values, label='Predicted', linewidth=2, alpha=0.7, linestyle='--', color='orange')
        axes[row, col].set_xlabel('Frame', fontsize=11)
        axes[row, col].set_ylabel(f'{dim_name}', fontsize=11)
        axes[row, col].set_title(f'{dim_name} Comparison', fontsize=12)
        axes[row, col].set_ylim(-1, 1)  # Set y-axis range to [-1, 1]
        axes[row, col].legend()
        axes[row, col].grid(True, alpha=0.3)

    plt.tight_layout()

    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Generate action comparison plot from CSV")

    parser.add_argument(
        "--csv",
        type=str,
        required=True,
        help="Path to frame_by_frame_comparison.csv"
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for plot (default: same dir as CSV with _new suffix)"
    )

    parser.add_argument(
        "--max_frames",
        type=int,
        default=300,
        help="Maximum number of frames to plot (default: 300)"
    )

    args = parser.parse_args()

    # Set default output path
    if args.output is None:
        csv_path = Path(args.csv)
        args.output = csv_path.parent / "action_comparison_new.png"

    # Generate plot
    plot_action_comparison(args.csv, args.output, args.max_frames)


if __name__ == "__main__":
    main()
