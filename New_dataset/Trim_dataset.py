#!/usr/bin/env python3
"""
Interactive Dataset Trimming Tool

This script allows you to trim HDF5 dataset files by selecting a frame range.
It preserves the original file and creates a new trimmed version.

Usage:
    python Trim_dataset.py episode.h5
    python Trim_dataset.py episode.h5 --start 0 --end 100
    python Trim_dataset.py episode.h5 --interactive
"""

import h5py
import numpy as np
import argparse
import sys
from pathlib import Path


def print_dataset_info(file_path):
    """Print information about the dataset."""
    print(f"\n{'='*60}")
    print(f"Dataset: {Path(file_path).name}")
    print(f"{'='*60}")

    with h5py.File(file_path, 'r') as f:
        action_len = len(f['action'])
        print(f"Total frames: {action_len}")
        print(f"Duration: ~{action_len/15:.1f} seconds (at 15 FPS)")

        # Show data structure
        print(f"\n📊 Data Structure:")
        print(f"  - action: {f['action'].shape}")
        print(f"  - qpos: {f['observations/qpos'].shape}")

        if 'observations/ee_pose' in f:
            print(f"  - ee_pose: {f['observations/ee_pose'].shape}")

        if 'observations/images' in f:
            images_grp = f['observations/images']
            cam_keys = sorted(list(images_grp.keys()))
            print(f"  - cameras: {cam_keys}")
            for cam in cam_keys:
                print(f"      {cam}: {len(images_grp[cam])} frames")

        if 'observations/sensor' in f:
            sensor_grp = f['observations/sensor']
            if 'force' in sensor_grp:
                print(f"  - force: {sensor_grp['force'].shape}")
            if 'aline' in sensor_grp:
                print(f"  - aline: {sensor_grp['aline'].shape}")

        # Show some statistics
        actions = f['action'][:]
        print(f"\n📈 Action Statistics:")
        print(f"  Movement (XYZ): min={actions[:,:3].min():.3f}, max={actions[:,:3].max():.3f}")
        print(f"  Rotation (RPY): min={actions[:,3:].min():.3f}, max={actions[:,3:].max():.3f}")

    print(f"{'='*60}\n")


def show_frame_samples(file_path, num_samples=5):
    """Show sample frames to help user decide where to trim."""
    with h5py.File(file_path, 'r') as f:
        total_frames = len(f['action'])
        actions = f['action'][:]
        qpos = f['observations/qpos'][:]

        # Sample frames evenly distributed
        indices = np.linspace(0, total_frames-1, num_samples, dtype=int)

        print(f"\n📸 Sample Frames (to help you decide trim range):")
        print(f"{'Frame':<10} {'Action (Movement)':<30} {'Action (Rotation)':<30} {'Joint Pos (avg)':<15}")
        print("-" * 90)

        for idx in indices:
            act = actions[idx]
            qp = qpos[idx]
            move_str = f"[{act[0]:.2f}, {act[1]:.2f}, {act[2]:.2f}]"
            rot_str = f"[{act[3]:.2f}, {act[4]:.2f}, {act[5]:.2f}]"
            qpos_avg = f"{qp.mean():.1f}"
            print(f"{idx:<10} {move_str:<30} {rot_str:<30} {qpos_avg:<15}")


def trim_dataset(input_path, output_path, start_frame, end_frame):
    """
    Trim dataset to specified frame range and save to new file.

    Args:
        input_path: Path to input HDF5 file
        output_path: Path to output HDF5 file
        start_frame: Starting frame (inclusive)
        end_frame: Ending frame (exclusive)
    """
    print(f"\n🔧 Trimming dataset...")
    print(f"  Input: {input_path}")
    print(f"  Output: {output_path}")
    print(f"  Frame range: {start_frame} to {end_frame} (total: {end_frame - start_frame} frames)")

    with h5py.File(input_path, 'r') as f_in:
        total_frames = len(f_in['action'])

        # Validate frame range
        if start_frame < 0 or end_frame > total_frames or start_frame >= end_frame:
            raise ValueError(f"Invalid frame range: [{start_frame}, {end_frame}). Total frames: {total_frames}")

        # Create output file
        with h5py.File(output_path, 'w') as f_out:
            print(f"\n📦 Copying data...")

            # Copy action
            action_data = f_in['action'][start_frame:end_frame]
            f_out.create_dataset('action', data=action_data)
            print(f"  ✓ action: {action_data.shape}")

            # Copy timestamp if exists
            if 'timestamp' in f_in:
                timestamp_data = f_in['timestamp'][start_frame:end_frame]
                f_out.create_dataset('timestamp', data=timestamp_data)
                print(f"  ✓ timestamp: {timestamp_data.shape}")

            # Create observations group
            obs_grp = f_out.create_group('observations')

            # Copy qpos
            qpos_data = f_in['observations/qpos'][start_frame:end_frame]
            obs_grp.create_dataset('qpos', data=qpos_data)
            print(f"  ✓ qpos: {qpos_data.shape}")

            # Copy ee_pose if exists
            if 'observations/ee_pose' in f_in:
                ee_pose_data = f_in['observations/ee_pose'][start_frame:end_frame]
                obs_grp.create_dataset('ee_pose', data=ee_pose_data)
                print(f"  ✓ ee_pose: {ee_pose_data.shape}")

            # Copy images
            if 'observations/images' in f_in:
                images_grp_in = f_in['observations/images']
                images_grp_out = obs_grp.create_group('images')

                cam_keys = sorted(list(images_grp_in.keys()))
                for cam in cam_keys:
                    # Read compressed frames
                    frames = images_grp_in[cam][start_frame:end_frame]

                    # Create dataset with object dtype for variable-length data
                    images_grp_out.create_dataset(
                        cam,
                        data=frames,
                        dtype=h5py.special_dtype(vlen=np.uint8)
                    )
                    print(f"  ✓ {cam}: {len(frames)} frames")

            # Copy sensor data if exists
            if 'observations/sensor' in f_in:
                sensor_grp_in = f_in['observations/sensor']
                sensor_grp_out = obs_grp.create_group('sensor')

                if 'force' in sensor_grp_in:
                    force_data = sensor_grp_in['force'][start_frame:end_frame]
                    sensor_grp_out.create_dataset('force', data=force_data)
                    print(f"  ✓ force: {force_data.shape}")

                if 'aline' in sensor_grp_in:
                    aline_data = sensor_grp_in['aline'][start_frame:end_frame]
                    sensor_grp_out.create_dataset('aline', data=aline_data)
                    print(f"  ✓ aline: {aline_data.shape}")

    print(f"\n✅ Trimmed dataset saved to: {output_path}")

    # Show file sizes
    input_size_mb = Path(input_path).stat().st_size / (1024 * 1024)
    output_size_mb = Path(output_path).stat().st_size / (1024 * 1024)
    print(f"\n📊 File sizes:")
    print(f"  Original: {input_size_mb:.2f} MB")
    print(f"  Trimmed:  {output_size_mb:.2f} MB ({output_size_mb/input_size_mb*100:.1f}%)")


def interactive_trim(input_path):
    """Interactive mode to select trim range."""
    print_dataset_info(input_path)
    show_frame_samples(input_path, num_samples=10)

    with h5py.File(input_path, 'r') as f:
        total_frames = len(f['action'])

    print(f"\n📝 Enter trim range:")
    print(f"  Total frames: 0 to {total_frames-1}")

    while True:
        try:
            start_str = input(f"  Start frame [0]: ").strip()
            start_frame = int(start_str) if start_str else 0

            end_str = input(f"  End frame [{total_frames}]: ").strip()
            end_frame = int(end_str) if end_str else total_frames

            if start_frame < 0 or end_frame > total_frames or start_frame >= end_frame:
                print(f"  ❌ Invalid range. Must be 0 <= start < end <= {total_frames}")
                continue

            break
        except ValueError:
            print("  ❌ Please enter valid integers")

    # Generate output filename
    input_file = Path(input_path)
    output_name = f"{input_file.stem}_trimmed_{start_frame}_{end_frame}.h5"
    output_path = input_file.parent / output_name

    # Confirm
    print(f"\n📋 Summary:")
    print(f"  Input:  {input_file.name} ({total_frames} frames)")
    print(f"  Output: {output_name} ({end_frame - start_frame} frames)")
    print(f"  Frames: {start_frame} to {end_frame}")

    confirm = input(f"\nProceed? [Y/n]: ").strip().lower()
    if confirm and confirm not in ['y', 'yes']:
        print("❌ Cancelled")
        return

    # Perform trim
    trim_dataset(input_path, output_path, start_frame, end_frame)


def main():
    parser = argparse.ArgumentParser(
        description="Trim HDF5 dataset files to specified frame range",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode (shows samples and prompts for range)
  python Trim_dataset.py episode.h5 --interactive

  # Direct trim with specified range
  python Trim_dataset.py episode.h5 --start 0 --end 100

  # Show info only
  python Trim_dataset.py episode.h5 --info
        """
    )

    parser.add_argument(
        'input_file',
        type=str,
        help='Path to input HDF5 dataset file'
    )

    parser.add_argument(
        '--start',
        type=int,
        default=None,
        help='Start frame (inclusive)'
    )

    parser.add_argument(
        '--end',
        type=int,
        default=None,
        help='End frame (exclusive)'
    )

    parser.add_argument(
        '-o', '--output',
        type=str,
        default=None,
        help='Output file path (default: auto-generated)'
    )

    parser.add_argument(
        '--interactive', '-i',
        action='store_true',
        help='Interactive mode (prompts for frame range)'
    )

    parser.add_argument(
        '--info',
        action='store_true',
        help='Show dataset info and exit'
    )

    args = parser.parse_args()

    # Check input file exists
    input_path = Path(args.input_file)
    if not input_path.exists():
        print(f"❌ Error: File not found: {input_path}")
        sys.exit(1)

    if not input_path.suffix == '.h5':
        print(f"❌ Error: File must be an HDF5 file (.h5): {input_path}")
        sys.exit(1)

    # Info mode
    if args.info:
        print_dataset_info(input_path)
        show_frame_samples(input_path)
        return

    # Interactive mode
    if args.interactive:
        interactive_trim(input_path)
        return

    # Direct mode (requires --start and --end)
    if args.start is None or args.end is None:
        print("❌ Error: --start and --end are required (or use --interactive)")
        print("Use --help for more information")
        sys.exit(1)

    # Generate output path if not specified
    if args.output:
        output_path = Path(args.output)
    else:
        output_name = f"{input_path.stem}_trimmed_{args.start}_{args.end}.h5"
        output_path = input_path.parent / output_name

    # Check if output file exists
    if output_path.exists():
        confirm = input(f"⚠️  Output file exists: {output_path}\nOverwrite? [y/N]: ").strip().lower()
        if confirm not in ['y', 'yes']:
            print("❌ Cancelled")
            sys.exit(1)

    # Perform trim
    trim_dataset(input_path, output_path, args.start, args.end)


if __name__ == "__main__":
    main()
