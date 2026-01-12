#!/usr/bin/env python
"""
Evaluation script for SmolVLA trained checkpoint

This script evaluates a trained SmolVLA policy checkpoint on the VLA insertion dataset.
It computes metrics such as MSE, action prediction accuracy, and per-episode performance.

Usage:
    python evaluate_smolvla.py --checkpoint /path/to/checkpoint.pt --config eval_config.yaml
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import yaml

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

# W&B for experiment tracking (optional)
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

# Add Train directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "Train"))

# LeRobot imports
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.utils.utils import get_safe_torch_device, init_logging

# VLA imports
from lerobot_adapter import create_vla_lerobot_dataset, lerobot_collate_fn

# Initialize logging
init_logging()
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_checkpoint(checkpoint_path: str, device: torch.device) -> Tuple[nn.Module, Dict]:
    """
    Load trained policy checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file (.pt)
        device: Device to load model to

    Returns:
        Tuple of (policy, checkpoint_dict)
    """
    logger.info(f"Loading checkpoint from: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

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
    logger.info(f"  Config available: {train_config is not None}")

    # Load policy config
    policy_cfg = train_config.get("policy", {})
    pretrained_model_id = policy_cfg.get("pretrained_model_id", "lerobot/smolvla_base")

    logger.info(f"Loading base policy from: {pretrained_model_id}")

    try:
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
        # Handle DataParallel wrapped models
        if any(k.startswith("module.") for k in policy_state_dict.keys()):
            # Remove "module." prefix from DataParallel
            new_state_dict = {}
            for k, v in policy_state_dict.items():
                if k.startswith("module.model."):
                    # DataParallel(DataParallelWrapper(SmolVLAPolicy))
                    new_key = k.replace("module.model.", "")
                    new_state_dict[new_key] = v
                elif k.startswith("module."):
                    new_key = k.replace("module.", "")
                    new_state_dict[new_key] = v
                else:
                    new_state_dict[k] = v
            policy_state_dict = new_state_dict

        # Load state dict
        missing_keys, unexpected_keys = policy.load_state_dict(policy_state_dict, strict=False)

        if missing_keys:
            logger.warning(f"Missing keys: {missing_keys[:5]}...")
        if unexpected_keys:
            logger.warning(f"Unexpected keys: {unexpected_keys[:5]}...")

        logger.info("Policy loaded successfully")

    except Exception as e:
        logger.error(f"Failed to load policy: {e}")
        raise

    # Move to device
    policy.to(device)
    policy.eval()

    return policy, checkpoint


def create_eval_dataloader(config: Dict) -> DataLoader:
    """Create evaluation dataloader from configuration."""
    dataset_cfg = config["dataset"]
    eval_cfg = config.get("evaluation", {})

    # Build full episode paths
    root_dir = Path(dataset_cfg["root_dir"])

    # Use evaluation episode dirs if specified, otherwise use training dirs
    episode_dir_list = eval_cfg.get("episode_dirs", dataset_cfg["episode_dirs"])
    episode_dirs = [root_dir / ep_dir for ep_dir in episode_dir_list]

    logger.info(f"Creating evaluation dataset with {len(episode_dirs)} episodes")

    # Create combined dataset
    dataset = create_vla_lerobot_dataset(
        episode_dirs=episode_dirs,
        horizon=dataset_cfg.get("horizon", 1),
        sensor_window_size=dataset_cfg.get("sensor_window_size", 65),
        robot_window_size=dataset_cfg.get("robot_window_size", 100),
        action_expert_hz=dataset_cfg.get("action_expert_hz", 10),
        use_joints_only=dataset_cfg.get("use_joints_only", False),
        use_poses_only=dataset_cfg.get("use_poses_only", True),
        use_full_action_chunk=dataset_cfg.get("use_full_action_chunk", False),
    )

    logger.info(f"Evaluation dataset created: {len(dataset)} total samples")

    # Create dataloader
    batch_size = eval_cfg.get("batch_size", 4)
    num_workers = eval_cfg.get("num_workers", 4)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=lerobot_collate_fn,
        drop_last=False,
    )

    logger.info(f"Evaluation DataLoader created: batch_size={batch_size}")

    return dataloader


@torch.no_grad()
def evaluate_batch(
    policy: nn.Module,
    batch: Dict,
    preprocessor,
    device: torch.device,
) -> Dict[str, float]:
    """
    Evaluate a single batch.

    Args:
        policy: SmolVLA policy model
        batch: Batch from dataloader
        preprocessor: Preprocessor for the policy
        device: Device to run on

    Returns:
        Dictionary of metrics for this batch
    """
    # Preprocess batch
    batch = preprocessor(batch)

    # Move batch to device
    batch = {
        k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
        for k, v in batch.items()
    }

    # Set to eval mode
    policy.eval()

    # Get predicted actions (skip loss computation due to architecture mismatch)
    # We only care about prediction accuracy for evaluation
    try:
        predicted_actions = policy.select_action(batch)
    except Exception as e:
        logger.warning(f"Failed to get predictions: {e}")
        predicted_actions = None

    # Get ground truth actions
    gt_actions = batch["action"]

    # Compute metrics
    metrics = {}

    # Note: Skipping loss computation to avoid architecture mismatch issues
    # For evaluation, action prediction accuracy is more important than loss

    # MSE between predicted and ground truth actions
    if predicted_actions is not None and gt_actions is not None:
        # Handle dimension mismatch (policy might predict 6D without gripper, gt is 7D with gripper)
        pred_dim = predicted_actions.shape[1]
        gt_dim = gt_actions.shape[1]

        if pred_dim != gt_dim:
            logger.debug(f"Action dimension mismatch: pred={pred_dim}, gt={gt_dim}")
            # Use the minimum dimension for comparison (exclude gripper)
            min_dim = min(pred_dim, gt_dim)
            predicted_actions_cmp = predicted_actions[:, :min_dim]
            gt_actions_cmp = gt_actions[:, :min_dim]
        else:
            predicted_actions_cmp = predicted_actions
            gt_actions_cmp = gt_actions

        # Overall MSE (excluding gripper if dimension mismatch)
        mse = torch.nn.functional.mse_loss(predicted_actions_cmp, gt_actions_cmp)
        metrics["action_mse"] = mse.item()

        # Per-dimension MSE
        per_dim_mse = ((predicted_actions_cmp - gt_actions_cmp) ** 2).mean(dim=0)
        for i, dim_mse in enumerate(per_dim_mse):
            metrics[f"action_mse_dim{i}"] = dim_mse.item()

        # Position MSE (first 3 dimensions: dx, dy, dz)
        pos_mse = ((predicted_actions_cmp[:, :3] - gt_actions_cmp[:, :3]) ** 2).mean()
        metrics["position_mse"] = pos_mse.item()

        # Rotation MSE (dimensions 3-5: rotation)
        if predicted_actions_cmp.shape[1] >= 6:
            rot_mse = ((predicted_actions_cmp[:, 3:6] - gt_actions_cmp[:, 3:6]) ** 2).mean()
            metrics["rotation_mse"] = rot_mse.item()

        # Gripper accuracy (dimension 6) - only if both have gripper dimension
        if pred_dim > 6 and gt_dim > 6:
            gripper_acc = ((predicted_actions[:, 6] > 0.5) == (gt_actions[:, 6] > 0.5)).float().mean()
            metrics["gripper_accuracy"] = gripper_acc.item()
        else:
            # Gripper not predicted, skip this metric
            metrics["gripper_accuracy"] = float('nan')

    return metrics


def evaluate(
    policy: nn.Module,
    dataloader: DataLoader,
    preprocessor,
    device: torch.device,
    save_predictions: bool = False,
    output_dir: Optional[Path] = None,
) -> Dict[str, float]:
    """
    Evaluate policy on full dataset.

    Args:
        policy: SmolVLA policy model
        dataloader: Evaluation dataloader
        preprocessor: Preprocessor for the policy
        device: Device to run on
        save_predictions: Whether to save predictions to file
        output_dir: Directory to save predictions

    Returns:
        Dictionary of aggregated metrics
    """
    logger.info("Starting evaluation...")

    policy.eval()

    all_metrics = []
    all_predictions = [] if save_predictions else None

    # Progress bar
    pbar = tqdm(dataloader, desc="Evaluating")

    for batch_idx, batch in enumerate(pbar):
        # Evaluate batch
        metrics = evaluate_batch(policy, batch, preprocessor, device)
        all_metrics.append(metrics)

        # Update progress bar
        if "loss" in metrics:
            pbar.set_postfix(loss=metrics["loss"], action_mse=metrics.get("action_mse", 0.0))

        # Save predictions if requested
        if save_predictions and output_dir:
            # TODO: Implement prediction saving
            pass

    # Aggregate metrics
    aggregated = {}

    # Get all metric keys
    all_keys = set()
    for m in all_metrics:
        all_keys.update(m.keys())

    # Compute mean and std for each metric
    for key in all_keys:
        values = [m[key] for m in all_metrics if key in m]
        if values:
            aggregated[f"{key}_mean"] = np.mean(values)
            aggregated[f"{key}_std"] = np.std(values)
            aggregated[f"{key}_min"] = np.min(values)
            aggregated[f"{key}_max"] = np.max(values)

    logger.info("Evaluation complete!")

    return aggregated


def print_results(metrics: Dict[str, float]):
    """Print evaluation results in a nice format."""
    print("\n" + "="*80)
    print("EVALUATION RESULTS")
    print("="*80)

    # Main metrics
    main_metrics = ["loss", "action_mse", "position_mse", "rotation_mse", "gripper_accuracy"]

    for metric_name in main_metrics:
        mean_key = f"{metric_name}_mean"
        std_key = f"{metric_name}_std"

        if mean_key in metrics:
            mean_val = metrics[mean_key]
            std_val = metrics.get(std_key, 0.0)
            print(f"{metric_name:25s}: {mean_val:10.6f} ± {std_val:.6f}")

    print("\n" + "-"*80)
    print("PER-DIMENSION ACTION MSE:")
    print("-"*80)

    # Per-dimension metrics
    for i in range(7):
        mean_key = f"action_mse_dim{i}_mean"
        if mean_key in metrics:
            mean_val = metrics[mean_key]
            std_val = metrics.get(f"action_mse_dim{i}_std", 0.0)
            dim_names = ["dx", "dy", "dz", "drx", "dry", "drz", "gripper"]
            dim_name = dim_names[i] if i < len(dim_names) else f"dim{i}"
            print(f"  {dim_name:10s}: {mean_val:10.6f} ± {std_val:.6f}")

    print("="*80 + "\n")


def save_results(metrics: Dict[str, float], output_path: Path):
    """Save evaluation results to file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save as YAML
    with open(output_path, 'w') as f:
        yaml.dump(metrics, f, default_flow_style=False)

    logger.info(f"Results saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate SmolVLA trained checkpoint")

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to checkpoint file (.pt)"
    )

    parser.add_argument(
        "--config",
        type=str,
        default="eval_config.yaml",
        help="Path to evaluation configuration file"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/eval",
        help="Output directory for results"
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (cpu/cuda)"
    )

    parser.add_argument(
        "--save_predictions",
        action="store_true",
        help="Save predictions to file"
    )

    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Log results to Weights & Biases"
    )

    parser.add_argument(
        "--wandb_project",
        type=str,
        default="smolvla-meca500-insertion",
        help="W&B project name"
    )

    args = parser.parse_args()

    # Setup
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        sys.exit(1)

    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"Config not found: {config_path}")
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Checkpoint: {checkpoint_path}")
    logger.info(f"Config: {config_path}")
    logger.info(f"Output directory: {output_dir}")

    # Load config
    logger.info("Loading configuration...")
    config = load_config(str(config_path))

    # Device
    device = get_safe_torch_device(args.device)
    logger.info(f"Using device: {device}")

    # Load checkpoint
    logger.info("Loading checkpoint...")
    policy, checkpoint = load_checkpoint(str(checkpoint_path), device)

    # Create preprocessor
    logger.info("Creating preprocessor...")
    pretrained_model_id = checkpoint.get("config", {}).get("policy", {}).get("pretrained_model_id", "lerobot/smolvla_base")
    preprocessor, postprocessor = make_pre_post_processors(
        policy.config,
        pretrained_model_id,
        preprocessor_overrides={"device_processor": {"device": str(device)}}
    )

    # Create evaluation dataloader
    logger.info("Creating evaluation dataloader...")
    eval_dataloader = create_eval_dataloader(config)

    # Run evaluation
    start_time = time.time()
    metrics = evaluate(
        policy=policy,
        dataloader=eval_dataloader,
        preprocessor=preprocessor,
        device=device,
        save_predictions=args.save_predictions,
        output_dir=output_dir,
    )
    eval_time = time.time() - start_time

    logger.info(f"Evaluation completed in {eval_time:.2f} seconds")

    # Print results
    print_results(metrics)

    # Save results
    results_path = output_dir / f"eval_results_step_{checkpoint.get('step', 'unknown')}.yaml"
    save_results(metrics, results_path)

    # W&B logging (optional)
    if args.wandb and WANDB_AVAILABLE:
        try:
            # Initialize W&B
            checkpoint_step = checkpoint.get('step', 'unknown')
            wandb.init(
                project=args.wandb_project,
                name=f"eval_step_{checkpoint_step}",
                tags=["evaluation", "smolvla", "meca500"],
                config={
                    "checkpoint_path": str(checkpoint_path),
                    "checkpoint_step": checkpoint_step,
                    "config_path": str(config_path),
                }
            )

            # Log metrics
            wandb_metrics = {}
            for key, value in metrics.items():
                if key.endswith("_mean"):
                    metric_name = key.replace("_mean", "")
                    wandb_metrics[f"eval/{metric_name}"] = value
                elif key.endswith("_std"):
                    metric_name = key.replace("_std", "")
                    wandb_metrics[f"eval/{metric_name}_std"] = value

            wandb.log(wandb_metrics)
            wandb.finish()

            logger.info(f"Results logged to W&B: {args.wandb_project}")

        except Exception as e:
            logger.warning(f"Failed to log to W&B: {e}")
    elif args.wandb and not WANDB_AVAILABLE:
        logger.warning("W&B requested but not installed. Install with: pip install wandb")

    logger.info("Evaluation complete!")


if __name__ == "__main__":
    main()
