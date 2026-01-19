#!/usr/bin/env python
"""
Plot action values from a single HDF5 episode file.

Usage:
# Full path
python plot_episode_actions.py \
    --episode /home/najo/NAS/VLA/dataset/New_dataset/collected_data/Eye_trocar/episode_20251223_142857.h5 \
    --output_dir ./outputs/action_plots

# Episode name only (will search in dataset directory)
python plot_episode_actions.py \
    --episode episode_20260107_140151.h5 \
    --output_dir ./outputs/action_plots
"""

import argparse
import h5py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import glob


def find_episode_file(episode_input: str, base_dir: str = "/home/najo/NAS/VLA/dataset/New_dataset/collected_data"):
    """
    Find episode file in dataset directory.

    Args:
        episode_input: Either full path or just episode filename
        base_dir: Base directory to search in

    Returns:
        Full path to episode file
    """
    episode_path = Path(episode_input)

    # If it's already a full path and exists, return it
    if episode_path.exists():
        return str(episode_path)

    # If it's just a filename, search for it recursively
    episode_name = episode_path.name
    print(f"Searching for '{episode_name}' in {base_dir}...")

    # Search recursively
    search_pattern = f"{base_dir}/**/{episode_name}"
    matches = glob.glob(search_pattern, recursive=True)

    if not matches:
        raise FileNotFoundError(f"Episode file '{episode_name}' not found in {base_dir}")

    if len(matches) > 1:
        print(f"Warning: Found {len(matches)} matching files:")
        for i, match in enumerate(matches, 1):
            print(f"  {i}. {match}")
        print(f"Using first match: {matches[0]}")

    return matches[0]


def load_episode_actions(h5_path: str):
    """Load action data from HDF5 episode file."""
    print(f"Loading episode from: {h5_path}")

    with h5py.File(h5_path, 'r') as f:
        # Load actions
        actions = f['action'][:]

        # Try to load timestamps if available
        try:
            timestamps = f['timestamp'][:]
        except KeyError:
            timestamps = None

        num_frames = len(actions)
        print(f"Loaded {num_frames} frames")
        print(f"Action shape: {actions.shape}")

    return actions, timestamps


def plot_actions(actions, timestamps=None, output_dir=None, episode_name=""):
    """Plot 6D action values across all frames."""
    num_frames = len(actions)
    action_dim = actions.shape[1]

    # Check if timestamps are valid (not all the same)
    use_timestamps = False
    if timestamps is not None:
        # Check if timestamps vary meaningfully
        unique_timestamps = np.unique(timestamps)
        if len(unique_timestamps) > 1:
            # Calculate time differences
            time_diffs = np.diff(timestamps)
            non_zero_diffs = time_diffs[time_diffs != 0]

            # Only use timestamps if:
            # 1. There are enough unique values (more than 10% of frames)
            # 2. The timestamps change reasonably often
            if len(unique_timestamps) > num_frames * 0.1 and len(non_zero_diffs) > 0:
                # Normalize to start from 0
                x_values = timestamps - timestamps[0]
                x_label = 'Time (s)'
                use_timestamps = True
                print(f"Using timestamps: {len(unique_timestamps)} unique values out of {num_frames} frames")

    # Use frame indices if timestamps are invalid or not available
    if not use_timestamps:
        x_values = np.arange(num_frames)
        x_label = 'Frame Index'

    # Dimension names
    dim_names = ['dx', 'dy', 'dz', 'drx', 'dry', 'drz']

    # Ensure we have 6 dimensions to plot
    if action_dim < 6:
        print(f"Warning: Action has only {action_dim} dimensions, expected 6")
        num_plots = action_dim
    else:
        num_plots = 6

    # Create figure with 6 subplots (3 rows x 2 columns)
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    fig.suptitle(f'Action Values Over Episode ({num_frames} frames)', fontsize=16)

    for i in range(num_plots):
        row = i // 2
        col = i % 2

        axes[row, col].plot(x_values, actions[:, i], linewidth=1.5, color='blue')
        axes[row, col].set_xlabel(x_label)
        axes[row, col].set_ylabel(dim_names[i])
        axes[row, col].set_title(f'{dim_names[i]} over time')
        axes[row, col].set_ylim(-1, 1)  # Fixed y-axis range for consistency
        axes[row, col].grid(True, alpha=0.3)

    plt.tight_layout()

    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        # Add underscore before episode_name if it's not empty
        name_suffix = f"_{episode_name}" if episode_name else ""
        save_path = output_path / f'action_plots{name_suffix}.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    else:
        plt.show()

    plt.close()

    # Print statistics for each dimension
    print("\n" + "="*80)
    print("ACTION STATISTICS")
    print("="*80)
    for i in range(num_plots):
        print(f"\n{dim_names[i]}:")
        print(f"  Mean: {actions[:, i].mean():.6f}")
        print(f"  Std:  {actions[:, i].std():.6f}")
        print(f"  Min:  {actions[:, i].min():.6f}")
        print(f"  Max:  {actions[:, i].max():.6f}")
    print("="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Plot actions from HDF5 episode")

    parser.add_argument(
        "--episode",
        type=str,
        required=True,
        help="Path to HDF5 episode file or just the episode filename (will search in dataset directory)"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for plot (if not specified, plot will be shown)"
    )

    parser.add_argument(
        "--base_dir",
        type=str,
        default="/home/najo/NAS/VLA/dataset/New_dataset/collected_data",
        help="Base directory to search for episode files (default: /home/najo/NAS/VLA/dataset/New_dataset/collected_data)"
    )

    args = parser.parse_args()

    try:
        # Find episode file (supports both full path and filename only)
        episode_path = find_episode_file(args.episode, args.base_dir)

        # Extract episode name from path (without .h5 extension)
        episode_name = Path(episode_path).stem

        # Load actions
        actions, timestamps = load_episode_actions(episode_path)

        # Plot actions
        plot_actions(actions, timestamps, args.output_dir, episode_name)

        print("Done!")

    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return


if __name__ == "__main__":
    main()
