#!/usr/bin/env python3
"""
Proper overfit test with:
1. Per-dimension MAE (unnormalized units)
2. Per-dimension correlation
3. k-shift sweep to find temporal alignment
"""

import sys
import torch
import numpy as np
import h5py
from pathlib import Path
from scipy.stats import pearsonr

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "lerobot" / "src"))

from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.utils.utils import get_safe_torch_device
from normalization_utils import Normalizer, load_stats
from PIL import Image
import cv2

def load_episode(episode_path):
    """Load episode data from HDF5."""
    with h5py.File(episode_path, 'r') as f:
        data = {
            'images': {},
            'ee_pose': f['observations']['ee_pose'][:],
            'action': f['action'][:],
            'num_frames': f['action'].shape[0]
        }

        # Load images
        for cam in ['camera1', 'camera2', 'camera3']:
            data['images'][cam] = f['observations']['images'][cam][:]

    return data

def decode_and_preprocess_image(jpeg_data):
    """Decode JPEG and preprocess for model."""
    if isinstance(jpeg_data, np.ndarray):
        jpeg_array = jpeg_data.flatten().astype(np.uint8)
    else:
        jpeg_array = np.frombuffer(jpeg_data, dtype=np.uint8)

    img_bgr = cv2.imdecode(jpeg_array, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (512, 512))
    img_float = img_resized.astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_float).permute(2, 0, 1)
    return img_tensor

def prepare_observation(episode_data, frame_idx, device, normalizer):
    """Prepare observation exactly as in training."""
    # Decode images
    images = {}
    for cam_idx, cam_name in enumerate(['camera1', 'camera2', 'camera3'], 1):
        img_tensor = decode_and_preprocess_image(episode_data['images'][cam_name][frame_idx])
        images[f'observation.images.camera{cam_idx}'] = img_tensor.unsqueeze(0).to(device)

    # Get state
    state = torch.tensor(episode_data['ee_pose'][frame_idx], dtype=torch.float32).to(device)
    state_normalized = normalizer.normalize(state, 'observation.state')

    observation = {**images, 'observation.state': state_normalized.unsqueeze(0)}
    return observation

def predict_all_actions(policy, episode_data, normalizer, device):
    """Predict actions for all frames."""
    num_frames = episode_data['num_frames']
    pred_actions = []

    policy.eval()
    with torch.no_grad():
        for frame_idx in range(num_frames):
            obs = prepare_observation(episode_data, frame_idx, device, normalizer)

            # Predict (normalized)
            action_norm = policy.select_action(obs)

            # Handle shape
            if action_norm.ndim == 3:
                action_norm = action_norm[0, 0, :]  # (1, 1, 6) -> (6,)
            elif action_norm.ndim == 2:
                action_norm = action_norm[0, :]  # (1, 6) -> (6,)

            # Unnormalize
            action_unnorm = normalizer.unnormalize(action_norm, 'action')
            pred_actions.append(action_unnorm.cpu().numpy())

    return np.array(pred_actions)

def compute_k_shift_mae(gt_actions, pred_actions, k_range=range(-3, 4)):
    """
    Compute MAE for different k shifts.
    k=0: pred[i] vs gt[i]
    k>0: pred[i] vs gt[i+k] (pred is ahead)
    k<0: pred[i] vs gt[i+k] (pred is behind)
    """
    results = {}

    for k in k_range:
        if k == 0:
            mae = np.abs(pred_actions - gt_actions).mean(axis=0)
        elif k > 0:
            # pred is ahead, so compare pred[:-k] with gt[k:]
            mae = np.abs(pred_actions[:-k] - gt_actions[k:]).mean(axis=0)
        else:  # k < 0
            # pred is behind, so compare pred[-k:] with gt[:k]
            mae = np.abs(pred_actions[-k:] - gt_actions[:k]).mean(axis=0)

        results[k] = mae

    return results

def main():
    print("=" * 80)
    print("🔍 PROPER OVERFIT TEST")
    print("=" * 80)

    # Paths
    checkpoint_path = Path("/home/najo/NAS/VLA/Insertion_VLAv4/Train/outputs/train/smolvla_needle_insertion/checkpoints/checkpoint_latest.pt")
    stats_path = Path("/home/najo/NAS/VLA/Insertion_VLAv4/Train/dataset_stats.yaml")

    # Use ONE episode from training set
    episode_path = Path("/home/najo/NAS/VLA/dataset/New_dataset/collected_data/Eye_trocar/Eye_trocar/260106/2_JYT/episode_20260106_134625_trimmed_0_556.h5")

    if not checkpoint_path.exists():
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return

    if not episode_path.exists():
        print(f"❌ Episode not found: {episode_path}")
        return

    print(f"\n📁 Loading episode: {episode_path.name}")
    print(f"📦 Loading checkpoint: {checkpoint_path.name}")

    # Setup
    device = get_safe_torch_device("cuda:0", log=True)

    # Load normalizer
    stats = load_stats(str(stats_path))
    normalizer = Normalizer(stats).to(device)

    # Load policy
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Load model config from checkpoint
    if 'config' in checkpoint:
        from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
        policy_config = checkpoint['config']['policy']
        config = SmolVLAConfig(
            pretrained_model_name_or_path=policy_config['pretrained_model_name_or_path'],
            chunk_size=policy_config.get('chunk_size', 1),
            n_obs_steps=policy_config.get('n_obs_steps', 1),
        )
        policy = SmolVLAPolicy(config)
    else:
        policy = SmolVLAPolicy.from_pretrained("HuggingFaceTB/smolvlm-500m-instruct")

    policy.load_state_dict(checkpoint['model_state_dict'], strict=False)
    policy.to(device)
    policy.eval()

    print("✅ Model loaded\n")

    # Load episode
    episode_data = load_episode(episode_path)
    num_frames = episode_data['num_frames']

    print(f"Episode has {num_frames} frames\n")

    # Get GT actions
    gt_actions = episode_data['action']

    # Predict all actions
    print("🔮 Predicting actions for all frames...")
    pred_actions = predict_all_actions(policy, episode_data, normalizer, device)

    print(f"✅ Predictions complete: {pred_actions.shape}\n")

    # 1. Per-dimension MAE (unnormalized)
    print("=" * 80)
    print("1️⃣  PER-DIMENSION MAE (Unnormalized Units)")
    print("=" * 80)

    per_dim_mae = np.abs(pred_actions - gt_actions).mean(axis=0)

    dim_names = ['dx (mm)', 'dy (mm)', 'dz (mm)', 'drx (°)', 'dry (°)', 'drz (°)']

    for i, (name, mae) in enumerate(zip(dim_names, per_dim_mae)):
        print(f"  {name}: {mae:.6f}")

    # 2. Per-dimension correlation
    print("\n" + "=" * 80)
    print("2️⃣  PER-DIMENSION CORRELATION")
    print("=" * 80)

    for i, name in enumerate(dim_names):
        if gt_actions[:, i].std() > 1e-6 and pred_actions[:, i].std() > 1e-6:
            corr, _ = pearsonr(gt_actions[:, i], pred_actions[:, i])
            print(f"  {name}: {corr:.4f}")
        else:
            print(f"  {name}: N/A (no variance)")

    # 3. k-shift sweep
    print("\n" + "=" * 80)
    print("3️⃣  K-SHIFT SWEEP (Finding Temporal Alignment)")
    print("=" * 80)
    print("\nk=0 means pred[i] vs gt[i] (correct alignment)")
    print("k>0 means pred is AHEAD (comparing pred[i] with gt[i+k])")
    print("k<0 means pred is BEHIND (comparing pred[i] with gt[i+k])\n")

    k_results = compute_k_shift_mae(gt_actions, pred_actions, k_range=range(-3, 4))

    # Print overall MAE for each k
    print("Overall MAE (average across all dims):")
    for k in sorted(k_results.keys()):
        overall_mae = k_results[k].mean()
        marker = " ← BEST" if k == min(k_results.keys(), key=lambda x: k_results[x].mean()) else ""
        print(f"  k={k:+2d}: {overall_mae:.6f}{marker}")

    # Find best k per dimension
    print("\nBest k per dimension:")
    for i, name in enumerate(dim_names):
        best_k = min(k_results.keys(), key=lambda x: k_results[x][i])
        best_mae = k_results[best_k][i]
        print(f"  {name}: k={best_k:+2d} (MAE={best_mae:.6f})")

    # 4. Diagnosis
    print("\n" + "=" * 80)
    print("🎯 DIAGNOSIS")
    print("=" * 80)

    best_overall_k = min(k_results.keys(), key=lambda x: k_results[x].mean())

    if best_overall_k == 0:
        print("\n✅ CORRECT ALIGNMENT: k=0 has lowest MAE")
        print("   → Model is predicting the right temporal step")

        if per_dim_mae.mean() < 0.5:
            print("\n✅ GOOD OVERFIT: Overall MAE < 0.5mm/degree")
        else:
            print("\n⚠️  WEAK OVERFIT: MAE is high even with correct alignment")
            print("   → Possible causes:")
            print("      - Model capacity too small")
            print("      - Data is multimodal (same obs → different actions)")
            print("      - Training not converged")
    else:
        print(f"\n❌ TEMPORAL MISALIGNMENT: k={best_overall_k} has lowest MAE (not k=0)")
        print("   → Model is predicting the WRONG temporal step!")

        if best_overall_k > 0:
            print(f"   → Prediction is {best_overall_k} steps AHEAD of GT")
            print("   → Possible cause: action label is for frame t+k, not frame t")
        else:
            print(f"   → Prediction is {-best_overall_k} steps BEHIND GT")
            print("   → Possible cause: observation is from frame t+k, not frame t")

    # Check for high variance in GT
    print("\n" + "=" * 80)
    print("4️⃣  GT ACTION VARIANCE (Multimodality Check)")
    print("=" * 80)

    gt_std = gt_actions.std(axis=0)
    for i, (name, std) in enumerate(zip(dim_names, gt_std)):
        print(f"  {name}: std={std:.6f}")

    print("\nIf std is very high, data might be multimodal")
    print("(same observation → multiple different actions)")

if __name__ == "__main__":
    main()
