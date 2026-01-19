#!/usr/bin/env python3
"""
Debug script to check normalization in training pipeline.

This script loads a batch from the dataloader and prints:
A. Action values BEFORE normalization (should have mean ~0.175, std ~0.36)
B. Action values AFTER normalization (should have mean ~0, std ~1)
C. Action chunk structure (to verify we're using the right index)
"""

import torch
import yaml
from pathlib import Path
from torch.utils.data import DataLoader

# Import training components
from hdf5_lerobot_adapter import create_hdf5_lerobot_dataset, hdf5_lerobot_collate_fn
from normalization_utils import Normalizer, load_stats

def main():
    # Load config
    config_path = Path(__file__).parent / "train_config_smolvla.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    print("=" * 80)
    print("🔍 NORMALIZATION DEBUG SCRIPT")
    print("=" * 80)

    # Load dataset
    dataset_config = config["dataset"]
    dataset_root = Path(dataset_config["root_dir"])
    hdf5_files = sorted(list(dataset_root.rglob("*.h5")))[:10]  # Just load 10 episodes for quick test

    print(f"\n📁 Loading {len(hdf5_files)} episodes...")

    dataset = create_hdf5_lerobot_dataset(
        hdf5_paths=[str(f) for f in hdf5_files],
        horizon=dataset_config.get("chunk_size", 1),
        n_obs_steps=dataset_config.get("n_obs_steps", 1),
        use_ee_pose_delta_as_action=dataset_config.get("use_ee_pose_delta_as_action", False),
    )

    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        collate_fn=hdf5_lerobot_collate_fn,
        num_workers=0,
    )

    # Load normalizer
    stats_path = Path(__file__).parent / "dataset_stats.yaml"
    stats = load_stats(str(stats_path))
    normalizer = Normalizer(stats)

    # Get one batch
    batch = next(iter(dataloader))

    # Print BEFORE normalization
    print("\n" + "=" * 80)
    print("A. ACTION VALUES BEFORE NORMALIZATION (Raw GT)")
    print("=" * 80)

    raw_actions = batch["action"]
    print(f"Shape: {raw_actions.shape}")
    print(f"  Expected: (batch_size, chunk_size, action_dim) or (batch_size, action_dim)")

    # If chunk dimension exists, look at first step only
    if raw_actions.ndim == 3:
        print(f"\n⚠️  Chunk dimension detected! Shape = {raw_actions.shape}")
        print(f"  We should analyze ONLY the first step [: 0, :]")
        first_step_actions = raw_actions[:, 0, :]
    else:
        first_step_actions = raw_actions

    print(f"\nFirst step actions shape: {first_step_actions.shape}")
    print(f"First step statistics:")
    print(f"  Mean: {first_step_actions.mean(dim=0).numpy()}")
    print(f"  Std:  {first_step_actions.std(dim=0).numpy()}")
    print(f"  Min:  {first_step_actions.min(dim=0).values.numpy()}")
    print(f"  Max:  {first_step_actions.max(dim=0).values.numpy()}")

    print(f"\nExpected from dataset_stats.yaml:")
    print(f"  Mean: {stats['action']['mean']}")
    print(f"  Std:  {stats['action']['std']}")

    # Apply normalization
    print("\n" + "=" * 80)
    print("B. ACTION VALUES AFTER NORMALIZATION")
    print("=" * 80)

    normalized_actions = normalizer.normalize(raw_actions, "action")

    print(f"Shape: {normalized_actions.shape}")

    if normalized_actions.ndim == 3:
        norm_first_step = normalized_actions[:, 0, :]
    else:
        norm_first_step = normalized_actions

    print(f"\nFirst step normalized statistics:")
    print(f"  Mean: {norm_first_step.mean(dim=0).numpy()}")
    print(f"  Std:  {norm_first_step.std(dim=0).numpy()}")
    print(f"  Min:  {norm_first_step.min(dim=0).values.numpy()}")
    print(f"  Max:  {norm_first_step.max(dim=0).values.numpy()}")

    print(f"\nExpected after normalization:")
    print(f"  Mean: ~0.0 (for all dimensions)")
    print(f"  Std:  ~1.0 (for all dimensions)")

    # Check if normalization is working correctly
    print("\n" + "=" * 80)
    print("C. NORMALIZATION VERIFICATION")
    print("=" * 80)

    mean_check = torch.abs(norm_first_step.mean(dim=0)) < 2.0  # Should be close to 0
    std_check = torch.abs(norm_first_step.std(dim=0) - 1.0) < 2.0  # Should be close to 1

    print(f"\nMean check (should be close to 0): {mean_check.numpy()}")
    print(f"Std check (should be close to 1):  {std_check.numpy()}")

    if mean_check.all():
        print("\n✅ Normalization appears to be working correctly!")
    else:
        print("\n❌ WARNING: Normalization may not be working correctly!")
        print("   Some dimensions have mean far from 0")

    # Test unnormalization
    print("\n" + "=" * 80)
    print("D. UNNORMALIZATION TEST")
    print("=" * 80)

    unnormalized = normalizer.unnormalize(normalized_actions, "action")

    if unnormalized.ndim == 3:
        unnorm_first_step = unnormalized[:, 0, :]
    else:
        unnorm_first_step = unnormalized

    reconstruction_error = torch.abs(unnorm_first_step - first_step_actions).mean(dim=0)
    print(f"Reconstruction error (should be ~0): {reconstruction_error.numpy()}")

    if (reconstruction_error < 1e-5).all():
        print("✅ Unnormalization working correctly!")
    else:
        print("❌ WARNING: Unnormalization error detected!")

    print("\n" + "=" * 80)
    print("🎯 SUMMARY")
    print("=" * 80)
    print("""
If you see:
- Raw actions with mean ~0.175, std ~0.36 → ✅ Correct GT values
- Normalized actions with mean ~0, std ~1 → ✅ Normalization working
- Reconstruction error ~0 → ✅ Unnormalization working

If normalization looks wrong, check:
1. Is normalize_batch() being called in training?
2. Is the model receiving normalized actions?
3. Is unnormalization applied during inference?
    """)

if __name__ == "__main__":
    main()
