#!/usr/bin/env python3
"""
Simple MAE measurement script with k-shift sweep
"""

import sys
import torch
import numpy as np
import h5py
import cv2
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "lerobot" / "src"))

from normalization_utils import Normalizer, load_stats

def load_checkpoint_simple(checkpoint_path, device):
    """Load checkpoint without complex config parsing."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Import here to avoid issues
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig

    # Create config from checkpoint
    policy_config = checkpoint['config']['policy']

    # Load pretrained model
    pretrained_id = policy_config.get('pretrained_model_id', 'lerobot/smolvla_base')
    policy = SmolVLAPolicy.from_pretrained(pretrained_id)

    # Update config if needed
    policy.config.chunk_size = policy_config.get('chunk_size', 50)
    policy.config.n_obs_steps = policy_config.get('n_obs_steps', 1)
    policy.load_state_dict(checkpoint['policy_state_dict'], strict=False)
    policy.to(device)
    policy.eval()

    return policy, policy.config, policy_config

def load_episode(episode_path):
    """Load episode from HDF5."""
    with h5py.File(episode_path, 'r') as f:
        data = {
            'images': {},
            'ee_pose': f['observations']['ee_pose'][:],
            'action': f['action'][:],
        }
        for cam in ['camera1', 'camera2', 'camera3']:
            data['images'][cam] = f['observations']['images'][cam][:]
    return data

def decode_image(jpeg_data):
    """Decode JPEG to tensor."""
    jpeg_array = np.frombuffer(jpeg_data, dtype=np.uint8) if isinstance(jpeg_data, bytes) else jpeg_data.flatten().astype(np.uint8)
    img_bgr = cv2.imdecode(jpeg_array, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (512, 512))
    img_float = img_resized.astype(np.float32) / 255.0
    return torch.from_numpy(img_float).permute(2, 0, 1)

def predict_single_frame(policy, episode_data, frame_idx, normalizer, device, tokenizer):
    """Predict action for single frame."""
    # Prepare images
    images = {}
    for i, cam in enumerate(['camera1', 'camera2', 'camera3'], 1):
        img_tensor = decode_image(episode_data['images'][cam][frame_idx])
        images[f'observation.images.camera{i}'] = img_tensor.unsqueeze(0).to(device)

    # Prepare state
    state = torch.tensor(episode_data['ee_pose'][frame_idx], dtype=torch.float32).to(device)
    state_norm = normalizer.normalize(state, 'observation.state')

    # Tokenize language instruction
    task_instruction = "Insert needle into red point"
    tokenized = tokenizer(
        task_instruction,
        max_length=48,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )

    # Create batch
    batch = {
        **images,
        'observation.state': state_norm.unsqueeze(0),
        'observation.language.tokens': tokenized['input_ids'].to(device),
        'observation.language.attention_mask': tokenized['attention_mask'].to(device),
    }

    # Predict
    with torch.no_grad():
        action_norm = policy.select_action(batch)

        # Handle shape
        if action_norm.ndim == 3:
            action_norm = action_norm[0, 0, :]
        elif action_norm.ndim == 2:
            action_norm = action_norm[0, :]

        # Unnormalize
        action = normalizer.unnormalize(action_norm, 'action')
        return action.cpu().numpy()

def compute_k_shift_mae(gt_actions, pred_actions, k_range=range(-3, 4)):
    """Compute MAE for different k shifts."""
    results = {}

    for k in k_range:
        if k == 0:
            valid_gt = gt_actions
            valid_pred = pred_actions
        elif k > 0:
            valid_gt = gt_actions[k:]
            valid_pred = pred_actions[:-k]
        else:  # k < 0
            valid_gt = gt_actions[:k]
            valid_pred = pred_actions[-k:]

        mae = np.abs(valid_pred - valid_gt).mean(axis=0)
        results[k] = mae

    return results

def main():
    print("=" * 80)
    print("🔍 QUICK MAE MEASUREMENT")
    print("=" * 80)

    # Paths
    checkpoint_path = Path("/home/najo/NAS/VLA/Insertion_VLAv4/Train/outputs/train/smolvla_needle_insertion/checkpoints/checkpoint_step_5000.pt")
    stats_path = Path("/home/najo/NAS/VLA/Insertion_VLAv4/Train/dataset_stats.yaml")
    episode_path = Path("/home/najo/NAS/VLA/dataset/New_dataset/collected_data/Eye_trocar/Eye_trocar/260106/2_JYT/episode_20260106_134625_trimmed_0_556.h5")

    # Setup
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")

    # Load
    print("Loading model...")
    policy, config, policy_config = load_checkpoint_simple(checkpoint_path, device)
    print(f"  chunk_size: {config.chunk_size}")
    print(f"  n_obs_steps: {config.n_obs_steps}")

    print("\nLoading normalizer...")
    stats = load_stats(str(stats_path))
    normalizer = Normalizer(stats).to(device)

    print("\nLoading tokenizer...")
    from transformers import AutoTokenizer
    vlm_model = policy_config.get('vlm_model_name', 'HuggingFaceTB/SmolVLM2-500M-Video-Instruct')
    tokenizer = AutoTokenizer.from_pretrained(vlm_model)

    print("\nLoading episode...")
    episode_data = load_episode(episode_path)
    num_frames = len(episode_data['action'])
    print(f"  Total frames: {num_frames}")

    # Predict all frames
    print("\nPredicting actions...")
    pred_actions = []
    for frame_idx in tqdm(range(num_frames)):
        pred_action = predict_single_frame(policy, episode_data, frame_idx, normalizer, device, tokenizer)
        pred_actions.append(pred_action)

    pred_actions = np.array(pred_actions)
    gt_actions = episode_data['action']

    print(f"\nPredictions shape: {pred_actions.shape}")
    print(f"GT actions shape: {gt_actions.shape}")

    # 1. Per-dimension MAE (k=0)
    print("\n" + "=" * 80)
    print("1️⃣  PER-DIMENSION MAE (k=0, unnormalized units)")
    print("=" * 80)

    mae = np.abs(pred_actions - gt_actions).mean(axis=0)
    dim_names = ['dx (mm)', 'dy (mm)', 'dz (mm)', 'drx (°)', 'dry (°)', 'drz (°)']

    for name, val in zip(dim_names, mae):
        print(f"  {name}: {val:.6f}")
    print(f"\nOverall MAE: {mae.mean():.6f}")

    # 2. k-shift sweep
    print("\n" + "=" * 80)
    print("2️⃣  K-SHIFT SWEEP")
    print("=" * 80)

    k_results = compute_k_shift_mae(gt_actions, pred_actions)

    print("\nOverall MAE for each k:")
    for k in sorted(k_results.keys()):
        overall_mae = k_results[k].mean()
        marker = " ← BEST" if k == min(k_results.keys(), key=lambda x: k_results[x].mean()) else ""
        print(f"  k={k:+2d}: {overall_mae:.6f}{marker}")

    # 3. Per-dimension best k
    print("\nBest k per dimension:")
    for i, name in enumerate(dim_names):
        best_k = min(k_results.keys(), key=lambda x: k_results[x][i])
        best_mae = k_results[best_k][i]
        print(f"  {name}: k={best_k:+2d} (MAE={best_mae:.6f})")

    # 4. Diagnosis
    print("\n" + "=" * 80)
    print("🎯 DIAGNOSIS")
    print("=" * 80)

    best_k = min(k_results.keys(), key=lambda x: k_results[x].mean())
    best_mae = k_results[best_k].mean()

    if best_k == 0:
        print("\n✅ TEMPORAL ALIGNMENT IS CORRECT (k=0 is best)")
        if best_mae < 0.5:
            print(f"✅ GOOD OVERFIT: MAE={best_mae:.4f} < 0.5")
        elif best_mae < 1.0:
            print(f"⚠️  WEAK OVERFIT: MAE={best_mae:.4f} (acceptable but not great)")
        else:
            print(f"❌ NO OVERFIT: MAE={best_mae:.4f} > 1.0")
            print("\nPossible causes:")
            print("  - Model hasn't converged")
            print("  - Data is multimodal (same obs → different actions)")
            print("  - Model capacity too small")
    else:
        print(f"\n❌ TEMPORAL MISALIGNMENT DETECTED!")
        print(f"   Best k={best_k} (should be k=0)")
        print(f"   MAE at k={best_k}: {best_mae:.4f}")
        print(f"   MAE at k=0: {k_results[0].mean():.4f}")

        if best_k > 0:
            print(f"\n   → Prediction is {best_k} steps AHEAD")
            print("   → Action label might be for wrong timestep")
        else:
            print(f"\n   → Prediction is {-best_k} steps BEHIND")
            print("   → Observation might be from wrong timestep")

if __name__ == "__main__":
    main()
