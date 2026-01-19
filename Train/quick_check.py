#!/usr/bin/env python3
"""Quick check: Training batch vs Inference batch comparison"""

import sys
import torch
import yaml
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from torch.utils.data import DataLoader
from hdf5_lerobot_adapter import create_hdf5_lerobot_dataset, hdf5_lerobot_collate_fn
from normalization_utils import Normalizer, load_stats

print("=" * 80)
print("🔍 QUICK CHECK: Training vs Inference Batch Comparison")
print("=" * 80)

# Load config
config_path = Path(__file__).parent / "train_config_smolvla.yaml"
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

dataset_config = config["dataset"]
dataset_root = Path(dataset_config["root_dir"])
hdf5_files = sorted(list(dataset_root.rglob("*.h5")))[:1]  # Just 1 episode

print(f"\n📁 Loading 1 episode for quick test...")

# Create dataset
dataset = create_hdf5_lerobot_dataset(
    hdf5_paths=[str(f) for f in hdf5_files],
    horizon=dataset_config.get("chunk_size", 1),
    n_obs_steps=dataset_config.get("n_obs_steps", 1),
    use_ee_pose_delta_as_action=dataset_config.get("use_ee_pose_delta_as_action", False),
)

dataloader = DataLoader(
    dataset,
    batch_size=1,
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

print("\n" + "=" * 80)
print("TRAINING BATCH (what goes into model.forward())")
print("=" * 80)

# Show keys
print(f"\nBatch keys: {list(batch.keys())}")

# Show observation.state
if 'observation.state' in batch:
    state = batch['observation.state']
    print(f"\nobservation.state:")
    print(f"  Shape: {state.shape}")
    print(f"  Mean: {state.mean(dim=0 if state.ndim == 2 else (0, 1)).numpy()}")
    print(f"  Std:  {state.std(dim=0 if state.ndim == 2 else (0, 1)).numpy()}")

    # Normalize it
    state_norm = normalizer.normalize(state, 'observation.state')
    print(f"\nobservation.state (AFTER normalization):")
    print(f"  Mean: {state_norm.mean(dim=0 if state_norm.ndim == 2 else (0, 1)).numpy()}")
    print(f"  Std:  {state_norm.std(dim=0 if state_norm.ndim == 2 else (0, 1)).numpy()}")
else:
    print("\n❌ observation.state NOT FOUND in batch!")

# Show images
for key in batch.keys():
    if 'images' in key and 'camera' in key:
        img = batch[key]
        print(f"\n{key}:")
        print(f"  Shape: {img.shape}")
        print(f"  Min/Max: {img.min():.3f} / {img.max():.3f}")
        print(f"  Mean: {img.mean():.3f}")

# Show action
if 'action' in batch:
    action = batch['action']
    print(f"\naction:")
    print(f"  Shape: {action.shape}")
    print(f"  Mean: {action.mean(dim=0 if action.ndim == 2 else (0, 1)).numpy()}")
    print(f"  Std:  {action.std(dim=0 if action.ndim == 2 else (0, 1)).numpy()}")

    # Normalize it
    action_norm = normalizer.normalize(action, 'action')
    print(f"\naction (AFTER normalization):")
    print(f"  Mean: {action_norm.mean(dim=0 if action_norm.ndim == 2 else (0, 1)).numpy()}")
    print(f"  Std:  {action_norm.std(dim=0 if action_norm.ndim == 2 else (0, 1)).numpy()}")

print("\n" + "=" * 80)
print("✅ TO CHECK IN INFERENCE:")
print("=" * 80)
print("""
1. Does observation.state exist and have the same shape?
2. Are images preprocessed the same way (resize, normalize)?
3. Is observation.state normalized before feeding to model?
4. Is predicted action unnormalized before returning?

Compare these values with what you see in evaluate_episode_normalized.py!
""")
