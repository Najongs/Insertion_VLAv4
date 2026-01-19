#!/usr/bin/env python
"""
Evaluate a trained SmolVLA model on a single episode with proper normalization.

This script applies MEAN_STD normalization to states and actions, matching the
training normalization approach from LeRobot.

Usage:
PYTHONPATH=/home/najo/NAS/VLA/Insertion_VLAv4/lerobot/src:$PYTHONPATH python3 \
    /home/najo/NAS/VLA/Insertion_VLAv4/Eval/evaluate_episode_normalized.py \
    --checkpoint /home/najo/NAS/VLA/Insertion_VLAv4/Train/outputs/train/smolvla_needle_insertion/checkpoints/checkpoint_step_10000.pt \
    --episode /home/najo/NAS/VLA/dataset/New_dataset/collected_data/Eye_trocar/Eye_trocar/260107/1_MIN/episode_20260107_134411.h5 \
    --output_dir /home/najo/NAS/VLA/Insertion_VLAv4/Eval/outputs/episode_eval_ep10_pre \
    --stats /home/najo/NAS/VLA/Insertion_VLAv4/Train/dataset_stats.yaml \
    --task_instruction "Insert needle into eye trocar"
"""

import argparse
import logging
import sys
from pathlib import Path
import yaml
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Optional
import torch
from PIL import Image
import io

# Add Train directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "Train"))

from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.utils.utils import get_safe_torch_device, init_logging
from normalization_utils import Normalizer, load_stats

# Initialize logging
init_logging()
logger = logging.getLogger(__name__)


def load_checkpoint(checkpoint_path: str, device):
    """Load model checkpoint."""
    logger.info(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Extract policy state dict
    policy_state_dict = checkpoint.get("policy_state_dict")
    if policy_state_dict is None:
        raise ValueError("Checkpoint does not contain 'policy_state_dict'")

    # Get config from checkpoint
    train_config = checkpoint.get("config")
    if train_config is None:
        logger.warning("Checkpoint does not contain training config")
        train_config = {}

    logger.info(f"Checkpoint info:")
    logger.info(f"  Step: {checkpoint.get('step', 'unknown')}")

    # Load policy config
    policy_cfg = train_config.get("policy", {})
    pretrained_model_id = policy_cfg.get("pretrained_model_id", "lerobot/smolvla_base")

    logger.info(f"Loading base policy from: {pretrained_model_id}")

    # Load pretrained policy architecture
    policy = SmolVLAPolicy.from_pretrained(pretrained_model_id)

    # Update policy config to match training config
    logger.info("Updating policy config from training config...")
    policy.config.n_obs_steps = policy_cfg.get("n_obs_steps", 1)
    policy.config.chunk_size = policy_cfg.get("chunk_size", 1)
    policy.config.n_action_steps = policy_cfg.get("n_action_steps", 1)

    logger.info(f"  n_obs_steps: {policy.config.n_obs_steps}")
    logger.info(f"  chunk_size: {policy.config.chunk_size}")
    logger.info(f"  n_action_steps: {policy.config.n_action_steps}")

    # Load trained weights
    logger.info("Loading trained weights...")
    policy.load_state_dict(policy_state_dict, strict=False)

    policy.to(device)
    policy.eval()
    logger.info("Model loaded successfully")

    return policy, checkpoint


def load_episode_data(h5_path: str):
    """Load all data from HDF5 episode file."""
    logger.info(f"Loading episode from: {h5_path}")

    with h5py.File(h5_path, 'r') as f:
        # Load images (all frames, all cameras)
        images = {}
        for cam_key in ['camera1', 'camera2', 'camera3']:
            cam_data = f['observations']['images'][cam_key][:]
            images[cam_key] = cam_data

        # Load actions
        actions = f['action'][:]

        # Load state (ee_pose)
        ee_pose = f['observations']['ee_pose'][:]

        # Load timestamps
        timestamps = f['timestamp'][:]

        num_frames = len(actions)
        logger.info(f"Loaded {num_frames} frames")
        logger.info(f"Action shape: {actions.shape}")
        logger.info(f"EE pose shape: {ee_pose.shape}")

    return {
        'images': images,
        'actions': actions,
        'ee_pose': ee_pose,
        'timestamps': timestamps,
        'num_frames': num_frames,
    }


def decode_image(img_bytes):
    """Decode JPEG bytes to PIL Image."""
    return Image.open(io.BytesIO(img_bytes))


def prepare_observation(
    episode_data,
    frame_idx,
    device,
    policy,
    normalizer,
    task_instruction="Insert needle into target point"
):
    """Prepare observation for model input with normalization."""
    # Decode images from JPEG bytes
    images_list = []
    for cam_key in ['camera1', 'camera2', 'camera3']:
        img_bytes = episode_data['images'][cam_key][frame_idx]
        img = decode_image(img_bytes)
        images_list.append(img)

    # Get state (ee_pose)
    state = episode_data['ee_pose'][frame_idx]
    state_tensor = torch.tensor(state, dtype=torch.float32).to(device)

    # *** NORMALIZE STATE ***
    state_normalized = normalizer.normalize(state_tensor, 'observation.state')

    # Prepare observation dict
    observation = {
        'observation.images.camera1': images_list[0],
        'observation.images.camera2': images_list[1],
        'observation.images.camera3': images_list[2],
        'observation.state': state_normalized,  # Use normalized state
    }

    # Add language tokens
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(policy.config.vlm_model_name)
    tokens = tokenizer(
        task_instruction,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=policy.config.tokenizer_max_length,
    )
    observation['observation.language.tokens'] = tokens['input_ids'].squeeze(0)
    observation['observation.language.attention_mask'] = tokens['attention_mask'].squeeze(0)

    # Move to device and add batch dimension
    batch = {}
    for key, value in observation.items():
        if isinstance(value, torch.Tensor):
            batch[key] = value.unsqueeze(0).to(device)
        elif isinstance(value, Image.Image):
            # Convert PIL Image to tensor
            import torchvision.transforms as T
            to_tensor = T.ToTensor()
            batch[key] = to_tensor(value).unsqueeze(0).to(device)

    return batch


def predict_action(policy, observation, normalizer, device):
    """Predict action for a single observation with unnormalization."""
    with torch.no_grad():
        # Forward pass
        action = policy.select_action(observation)

        # Get first action from chunk (if chunk_size > 1)
        if len(action.shape) == 2:  # [batch, action_dim]
            action_normalized = action[0]  # Remove batch dimension
        elif len(action.shape) == 3:  # [batch, chunk_size, action_dim]
            action_normalized = action[0, 0, :]  # Take first action
        else:
            raise ValueError(f"Unexpected action shape: {action.shape}")

        # *** UNNORMALIZE ACTION ***
        action_unnormalized = normalizer.unnormalize(action_normalized, 'action')

        return action_unnormalized.cpu().numpy()


def evaluate_episode(
    policy,
    episode_data,
    normalizer,
    device,
    task_instruction="Insert needle into target point",
):
    """Evaluate policy on entire episode, frame by frame."""
    num_frames = episode_data['num_frames']
    action_dim = episode_data['actions'].shape[1]

    # Storage for results
    results = {
        'frame_idx': [],
        'gt_actions': [],
        'pred_actions': [],
        'action_errors': [],
        'position_errors': [],
        'rotation_errors': [],
    }

    logger.info(f"Evaluating {num_frames} frames with normalization...")

    for frame_idx in range(num_frames):
        if frame_idx % 50 == 0:
            logger.info(f"Processing frame {frame_idx}/{num_frames}")

        try:
            # Get ground truth action
            gt_action = episode_data['actions'][frame_idx]

            # Prepare observation (with normalized state)
            observation = prepare_observation(
                episode_data,
                frame_idx,
                device,
                policy,
                normalizer,
                task_instruction
            )

            # Predict action (with unnormalization)
            pred_action = predict_action(policy, observation, normalizer, device)

            # Handle dimension mismatch
            min_dim = min(len(pred_action), len(gt_action))
            pred_action_cmp = pred_action[:min_dim]
            gt_action_cmp = gt_action[:min_dim]

            # Compute errors
            action_error = np.mean((pred_action_cmp - gt_action_cmp) ** 2)
            position_error = np.mean((pred_action_cmp[:3] - gt_action_cmp[:3]) ** 2)
            rotation_error = np.mean((pred_action_cmp[3:6] - gt_action_cmp[3:6]) ** 2)

            # Store results
            results['frame_idx'].append(frame_idx)
            results['gt_actions'].append(gt_action)
            results['pred_actions'].append(pred_action)
            results['action_errors'].append(action_error)
            results['position_errors'].append(position_error)
            results['rotation_errors'].append(rotation_error)

        except Exception as e:
            logger.warning(f"Failed to process frame {frame_idx}: {e}")
            continue

    logger.info("Evaluation complete!")

    return results


def create_comparison_dataframe(results):
    """Create pandas DataFrame with detailed comparison."""
    data = []

    for i in range(len(results['frame_idx'])):
        frame_idx = results['frame_idx'][i]
        gt_action = results['gt_actions'][i]
        pred_action = results['pred_actions'][i]

        row = {
            'Frame': frame_idx,
            'Action_Error': results['action_errors'][i],
            'Position_Error': results['position_errors'][i],
            'Rotation_Error': results['rotation_errors'][i],
        }

        # Add per-dimension values
        min_dim = min(len(gt_action), len(pred_action))
        dim_names = ['dx', 'dy', 'dz', 'drx', 'dry', 'drz', 'gripper']

        for d in range(min_dim):
            dim_name = dim_names[d] if d < len(dim_names) else f'dim{d}'
            row[f'GT_{dim_name}'] = gt_action[d]
            row[f'Pred_{dim_name}'] = pred_action[d] if d < len(pred_action) else 0.0
            row[f'Error_{dim_name}'] = (pred_action[d] - gt_action[d]) ** 2 if d < len(pred_action) else 0.0

        data.append(row)

    df = pd.DataFrame(data)
    return df


def plot_results(results, output_dir: Path):
    """Create visualization plots."""
    output_dir.mkdir(parents=True, exist_ok=True)

    frames = results['frame_idx']

    # Plot 1: Error over time
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    fig.suptitle('Prediction Errors Over Episode (With Normalization)', fontsize=16)

    axes[0].plot(frames, results['action_errors'], linewidth=1.5, color='blue')
    axes[0].set_ylabel('Action MSE')
    axes[0].set_title('Overall Action Error')
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(frames, results['position_errors'], linewidth=1.5, color='green')
    axes[1].set_ylabel('Position MSE')
    axes[1].set_title('Position Error (dx, dy, dz)')
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(frames, results['rotation_errors'], linewidth=1.5, color='red')
    axes[2].set_xlabel('Frame')
    axes[2].set_ylabel('Rotation MSE')
    axes[2].set_title('Rotation Error (drx, dry, drz)')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'error_over_time.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Plot 2: Per-dimension comparison
    max_frames_to_plot = min(300, len(frames))
    gt_actions = np.array(results['gt_actions'][:max_frames_to_plot])
    pred_actions = np.array(results['pred_actions'][:max_frames_to_plot])
    frames_subset = frames[:max_frames_to_plot]

    dim_names = ['dx', 'dy', 'dz', 'drx', 'dry', 'drz']
    min_dim = min(gt_actions.shape[1], pred_actions.shape[1], 6)

    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    fig.suptitle(f'Ground Truth vs Predicted Actions (First {max_frames_to_plot} Frames, Normalized)', fontsize=16)

    for i in range(min_dim):
        row = i // 2
        col = i % 2

        axes[row, col].plot(frames_subset, gt_actions[:, i], label='Ground Truth', linewidth=2, alpha=0.7)
        axes[row, col].plot(frames_subset, pred_actions[:, i], label='Predicted', linewidth=2, alpha=0.7, linestyle='--')
        axes[row, col].set_xlabel('Frame')
        axes[row, col].set_ylabel(dim_names[i])
        axes[row, col].set_title(f'{dim_names[i]} Comparison')
        axes[row, col].set_ylim(-1, 1)
        axes[row, col].legend()
        axes[row, col].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'action_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()

    logger.info(f"Plots saved to {output_dir}")


def print_summary(results):
    """Print summary statistics."""
    action_errors = np.array(results['action_errors'])
    position_errors = np.array(results['position_errors'])
    rotation_errors = np.array(results['rotation_errors'])

    print("\n" + "="*80)
    print("EPISODE EVALUATION SUMMARY (WITH NORMALIZATION)")
    print("="*80)
    print(f"Total frames evaluated: {len(results['frame_idx'])}")
    print()
    print("Action MSE:")
    print(f"  Mean: {action_errors.mean():.6f}")
    print(f"  Std:  {action_errors.std():.6f}")
    print(f"  Min:  {action_errors.min():.6f}")
    print(f"  Max:  {action_errors.max():.6f}")
    print()
    print("Position MSE (dx, dy, dz):")
    print(f"  Mean: {position_errors.mean():.6f}")
    print(f"  Std:  {position_errors.std():.6f}")
    print(f"  Min:  {position_errors.min():.6f}")
    print(f"  Max:  {position_errors.max():.6f}")
    print()
    print("Rotation MSE (drx, dry, drz):")
    print(f"  Mean: {rotation_errors.mean():.6f}")
    print(f"  Std:  {rotation_errors.std():.6f}")
    print(f"  Min:  {rotation_errors.min():.6f}")
    print(f"  Max:  {rotation_errors.max():.6f}")
    print("="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Evaluate episode with normalization")

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint (.pt)"
    )

    parser.add_argument(
        "--episode",
        type=str,
        required=True,
        help="Path to HDF5 episode file"
    )

    parser.add_argument(
        "--stats",
        type=str,
        required=True,
        help="Path to dataset statistics YAML file"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/episode_eval_normalized",
        help="Output directory for results"
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (cpu/cuda)"
    )

    parser.add_argument(
        "--task_instruction",
        type=str,
        default="Insert needle into target point",
        help="Task instruction text"
    )

    args = parser.parse_args()

    # Setup
    checkpoint_path = Path(args.checkpoint)
    episode_path = Path(args.episode)
    stats_path = Path(args.stats)
    output_dir = Path(args.output_dir)

    if not checkpoint_path.exists():
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        sys.exit(1)

    if not episode_path.exists():
        logger.error(f"Episode not found: {episode_path}")
        sys.exit(1)

    if not stats_path.exists():
        logger.error(f"Statistics file not found: {stats_path}")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Device
    device = get_safe_torch_device(args.device)
    logger.info(f"Using device: {device}")

    # Load normalization statistics
    logger.info(f"Loading normalization statistics from: {stats_path}")
    stats = load_stats(str(stats_path))
    normalizer = Normalizer(stats).to(device)
    logger.info("Normalizer created successfully")

    # Load checkpoint
    policy, checkpoint = load_checkpoint(str(checkpoint_path), device)

    # Load episode data
    episode_data = load_episode_data(str(episode_path))

    # Evaluate episode
    results = evaluate_episode(
        policy=policy,
        episode_data=episode_data,
        normalizer=normalizer,
        device=device,
        task_instruction=args.task_instruction,
    )

    # Print summary
    print_summary(results)

    # Create DataFrame
    logger.info("Creating detailed comparison DataFrame...")
    df = create_comparison_dataframe(results)

    # Save DataFrame
    csv_path = output_dir / 'frame_by_frame_comparison.csv'
    df.to_csv(csv_path, index=False)
    logger.info(f"Detailed comparison saved to: {csv_path}")

    # Save summary statistics
    summary = {
        'num_frames': len(results['frame_idx']),
        'action_mse_mean': float(np.mean(results['action_errors'])),
        'action_mse_std': float(np.std(results['action_errors'])),
        'position_mse_mean': float(np.mean(results['position_errors'])),
        'position_mse_std': float(np.std(results['position_errors'])),
        'rotation_mse_mean': float(np.mean(results['rotation_errors'])),
        'rotation_mse_std': float(np.std(results['rotation_errors'])),
    }

    summary_path = output_dir / 'summary.yaml'
    with open(summary_path, 'w') as f:
        yaml.dump(summary, f, default_flow_style=False)
    logger.info(f"Summary saved to: {summary_path}")

    # Create plots
    logger.info("Creating visualization plots...")
    try:
        plot_results(results, output_dir)
    except Exception as e:
        logger.warning(f"Failed to create plots: {e}")

    logger.info(f"\nAll results saved to: {output_dir}")
    logger.info("Evaluation complete!")


if __name__ == "__main__":
    main()
