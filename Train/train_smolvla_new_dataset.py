#!/usr/bin/env python
"""
Training script for SmolVLA on New HDF5 VLA Dataset

This script trains SmolVLA policy using the new HDF5 dataset format with downloaded model.

Usage:
    python train_smolvla_new_dataset.py --config train_config_new_dataset.yaml
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional
import yaml

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

# LeRobot imports
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.factory import make_pre_post_processors
from lerobot.utils.utils import get_safe_torch_device, init_logging

# HDF5 VLA imports
from hdf5_lerobot_adapter import create_hdf5_lerobot_dataset, hdf5_lerobot_collate_fn

# Initialize logging
init_logging()
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_dataset_from_config(config: Dict) -> torch.utils.data.Dataset:
    """Create HDF5 LeRobot dataset from configuration."""
    dataset_cfg = config["dataset"]

    # Build full HDF5 file paths
    root_dir = Path(dataset_cfg["root_dir"])
    hdf5_files = []

    # Support both specific file list and pattern matching
    if "hdf5_files" in dataset_cfg:
        # Specific file list
        hdf5_files = [root_dir / f for f in dataset_cfg["hdf5_files"]]
    else:
        # Use all HDF5 files in directory
        hdf5_files = sorted(root_dir.glob("*.h5"))

    logger.info(f"Creating dataset with {len(hdf5_files)} HDF5 episodes")

    # Create combined dataset
    dataset = create_hdf5_lerobot_dataset(
        hdf5_paths=hdf5_files,
        horizon=dataset_cfg.get("horizon", 1),
        use_qpos=dataset_cfg.get("use_qpos", False),
        use_ee_pose=dataset_cfg.get("use_ee_pose", True),
        task_instruction=dataset_cfg.get("task_instruction", "Insert the needle into the target point"),
        camera_dropout_prob=dataset_cfg.get("camera_dropout_prob", 0.0),
        min_cameras=dataset_cfg.get("min_cameras", 1),
        augment=dataset_cfg.get("augment", True),
        augment_brightness=dataset_cfg.get("augment_brightness", 0.2),
        augment_contrast=dataset_cfg.get("augment_contrast", 0.2),
        augment_saturation=dataset_cfg.get("augment_saturation", 0.2),
        augment_hue=dataset_cfg.get("augment_hue", 0.05),
        augment_noise=dataset_cfg.get("augment_noise", 0.02),
    )

    logger.info(f"Dataset created: {len(dataset)} total samples")

    return dataset


def create_dataloader_from_config(
    dataset: torch.utils.data.Dataset,
    config: Dict,
    is_train: bool = True
) -> DataLoader:
    """Create DataLoader from configuration."""
    if is_train:
        training_cfg = config["training"]
        batch_size = training_cfg["batch_size"]
        shuffle = training_cfg.get("shuffle", True)
        num_workers = training_cfg.get("num_workers", 4)
        pin_memory = training_cfg.get("pin_memory", True)
    else:
        eval_cfg = config.get("evaluation", {})
        batch_size = eval_cfg.get("batch_size", 4)
        shuffle = False
        num_workers = config["training"].get("num_workers", 4)
        pin_memory = config["training"].get("pin_memory", True)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=hdf5_lerobot_collate_fn,
        drop_last=True if is_train else False,
    )

    logger.info(f"DataLoader created: batch_size={batch_size}, num_workers={num_workers}")

    return dataloader


def create_policy_from_config(config: Dict, sample_batch: Dict) -> SmolVLAPolicy:
    """Create SmolVLA policy from configuration."""
    policy_cfg = config["policy"]

    # Get state and action dimensions from sample
    state_dim = sample_batch["observation.state"].shape[-1]
    action_dim = sample_batch["action"].shape[-1]

    # Count number of cameras
    num_cameras = sum(1 for key in sample_batch.keys() if key.startswith("observation.images.camera"))

    logger.info(f"Policy input: {num_cameras} cameras, state_dim={state_dim}, action_dim={action_dim}")

    # Load pretrained model (downloaded from Hugging Face or local)
    pretrained_model_path = policy_cfg.get("pretrained_model_path")

    if not pretrained_model_path:
        raise ValueError("pretrained_model_path must be specified in config")

    # =========================================================================
    # [FIX] 로컬 경로 체크 로직 수정: 경로가 없으면 Hugging Face ID로 간주
    # =========================================================================
    model_source = str(pretrained_model_path)
    if Path(model_source).exists():
        logger.info(f"Loading pretrained SmolVLA from local path: {model_source}")
    else:
        logger.info(f"Local path '{model_source}' not found. Assuming Hugging Face Hub ID and downloading...")

    try:
        from lerobot.configs.types import FeatureType, PolicyFeature

        # Load pretrained policy (from_pretrained handles both path and ID)
        policy = SmolVLAPolicy.from_pretrained(model_source)

        # Update config for our dataset
        policy.config.n_obs_steps = policy_cfg.get("n_obs_steps", 1)
        policy.config.chunk_size = policy_cfg.get("chunk_size", 1)
        policy.config.n_action_steps = policy_cfg.get("n_action_steps", 1)

        # Configure input/output features
        logger.info("Configuring input/output features...")
        policy.config.input_features = {}

        # Add camera features (512x512 to match model's expected size)
        for cam_idx in range(1, num_cameras + 1):
            policy.config.input_features[f"observation.images.camera{cam_idx}"] = PolicyFeature(
                type=FeatureType.VISUAL,
                shape=(3, 512, 512),
            )

        # Add state feature
        policy.config.input_features["observation.state"] = PolicyFeature(
            type=FeatureType.STATE,
            shape=(state_dim,),
        )

        # Add output features
        policy.config.output_features = {
            "action": PolicyFeature(
                type=FeatureType.ACTION,
                shape=(action_dim,),
            )
        }

        # =========================================================================
        # IMPORTANT: First, unfreeze ALL parameters (reset to trainable state)
        # This ensures we start from a clean slate before applying freeze settings
        # =========================================================================
        logger.info("Unfreezing all parameters first...")
        for param in policy.parameters():
            param.requires_grad = True

        # Count initial trainable params
        initial_trainable = sum(p.numel() for p in policy.parameters() if p.requires_grad)
        initial_total = sum(p.numel() for p in policy.parameters())
        logger.info(f"After unfreezing: {initial_trainable:,} / {initial_total:,} params trainable")

        # Now apply freeze settings if configured
        # Note: Default is False (don't freeze) - only freeze if explicitly set to True
        if policy_cfg.get("freeze_vision_encoder", False):
            logger.info("Freezing vision encoder")
            try:
                if hasattr(policy.model.vlm_with_expert.vlm, 'vision_encoder'):
                    for param in policy.model.vlm_with_expert.vlm.vision_encoder.parameters():
                        param.requires_grad = False
                elif hasattr(policy.model.vlm_with_expert.vlm, 'model') and hasattr(policy.model.vlm_with_expert.vlm.model, 'vision_tower'):
                    for param in policy.model.vlm_with_expert.vlm.model.vision_tower.parameters():
                        param.requires_grad = False
                logger.info("Vision encoder frozen")
            except Exception as e:
                logger.warning(f"Failed to freeze vision encoder: {e}")

        if policy_cfg.get("train_expert_only", False):
            logger.info("Training expert only (freezing VLM)")
            try:
                for param in policy.model.vlm_with_expert.vlm.parameters():
                    param.requires_grad = False
                for param in policy.model.vlm_with_expert.lm_expert.parameters():
                    param.requires_grad = True
                logger.info("VLM frozen, expert unfrozen")
            except Exception as e:
                logger.warning(f"Failed to configure train_expert_only: {e}")

        if policy_cfg.get("train_state_proj", False):
            logger.info("Training state projection layer")
            try:
                if hasattr(policy.model, 'state_proj'):
                    for param in policy.model.state_proj.parameters():
                        param.requires_grad = True
                logger.info("State projection layer unfrozen")
            except Exception as e:
                logger.warning(f"Failed to unfreeze state_proj: {e}")

        # Count final trainable params after freeze settings
        final_trainable = sum(p.numel() for p in policy.parameters() if p.requires_grad)
        logger.info(f"After freeze settings: {final_trainable:,} / {initial_total:,} params trainable")

        logger.info("Pretrained policy loaded successfully")

    except Exception as e:
        logger.error(f"Failed to load pretrained model: {e}")
        raise

    # Move to device
    device = get_safe_torch_device(policy_cfg.get("device", "cuda"))
    policy.to(device)

    # Multi-GPU support
    use_multi_gpu = policy_cfg.get("use_multi_gpu", False)
    if use_multi_gpu and torch.cuda.device_count() > 1:
        logger.info(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
        
        class DataParallelWrapper(nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model
                self.config = model.config
            def forward(self, batch):
                loss, loss_dict = self.model(batch)
                return loss
            def __getattr__(self, name):
                try:
                    return super().__getattr__(name)
                except AttributeError:
                    return getattr(self.model, name)

        wrapped_policy = DataParallelWrapper(policy)
        policy = nn.DataParallel(wrapped_policy)

    # Count trainable parameters
    model_to_count = policy.module.model if isinstance(policy, nn.DataParallel) else policy
    trainable_params = sum(p.numel() for p in model_to_count.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model_to_count.parameters())
    logger.info(f"Policy parameters: {trainable_params:,} trainable / {total_params:,} total")

    return policy


def create_optimizer(policy: nn.Module, config: Dict) -> torch.optim.Optimizer:
    """Create optimizer from configuration."""
    opt_cfg = config["optimizer"]

    # Handle DataParallel wrapped model
    if isinstance(policy, nn.DataParallel):
        model_to_optimize = policy.module.model
    else:
        model_to_optimize = policy

    # Get trainable parameters
    trainable_params = [p for p in model_to_optimize.parameters() if p.requires_grad]

    optimizer = AdamW(
        trainable_params,
        lr=opt_cfg.get("lr", 1e-4),
        betas=tuple(opt_cfg.get("betas", [0.9, 0.95])),
        weight_decay=opt_cfg.get("weight_decay", 1e-10),
        eps=opt_cfg.get("eps", 1e-8),
    )

    logger.info(f"Optimizer created: AdamW with lr={opt_cfg.get('lr', 1e-4)}")

    return optimizer


def create_scheduler(optimizer: torch.optim.Optimizer, config: Dict) -> torch.optim.lr_scheduler.LRScheduler:
    """Create learning rate scheduler from configuration."""
    sched_cfg = config["scheduler"]

    num_warmup_steps = sched_cfg.get("num_warmup_steps", 1000)
    num_decay_steps = sched_cfg.get("num_decay_steps", 20000)
    peak_lr = sched_cfg.get("peak_lr", 1e-4)
    decay_lr = sched_cfg.get("decay_lr", 1e-6)

    def lr_lambda(current_step: int) -> float:
        """Learning rate schedule with warmup and cosine decay."""
        if current_step < num_warmup_steps:
            # Warmup: linear increase from 0 to peak_lr
            return current_step / num_warmup_steps
        else:
            # Cosine decay from peak_lr to decay_lr
            progress = (current_step - num_warmup_steps) / (num_decay_steps - num_warmup_steps)
            progress = min(progress, 1.0)
            cosine_decay = 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.14159)))
            return (decay_lr / peak_lr) + (1.0 - decay_lr / peak_lr) * cosine_decay

    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

    logger.info(f"Scheduler created: warmup={num_warmup_steps}, decay={num_decay_steps}")

    return scheduler


def train_step(
    policy: nn.Module,
    batch: Dict,
    preprocessor,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip_norm: Optional[float] = None,
) -> Dict[str, float]:
    """Execute one training step."""

    # Preprocess batch
    batch = preprocessor(batch)

    # Move batch to device
    batch = {
        k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
        for k, v in batch.items()
    }

    # Forward pass
    policy.train()

    # Handle DataParallel
    is_data_parallel = isinstance(policy, nn.DataParallel)

    if is_data_parallel:
        # DataParallel mode: policy returns only loss
        loss = policy(batch)
        loss = loss.mean()
        loss_dict = {}
    else:
        # Normal mode: policy returns (loss, loss_dict)
        loss, loss_dict = policy(batch)

    # Backward pass
    optimizer.zero_grad()
    loss.backward()

    # Gradient clipping
    if grad_clip_norm is not None:
        torch.nn.utils.clip_grad_norm_(policy.parameters(), grad_clip_norm)

    # Optimizer step
    optimizer.step()

    # Return metrics
    metrics = {}
    for key, value in loss_dict.items():
        if isinstance(value, torch.Tensor):
            metrics[key] = value.item() if value.numel() == 1 else value.mean().item()
        else:
            metrics[key] = value

    # Ensure 'loss' is in metrics
    if "loss" not in metrics:
        if isinstance(loss, torch.Tensor):
            metrics["loss"] = loss.mean().item() if loss.numel() > 1 else loss.item()
        else:
            metrics["loss"] = loss

    return metrics


def save_checkpoint(
    policy: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    step: int,
    config: Dict,
    output_dir: Path,
) -> None:
    """Save training checkpoint (latest only, overwrites previous)."""
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "step": step,
        "policy_state_dict": policy.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "config": config,
    }

    # Save as latest (overwrites previous checkpoint)
    latest_path = checkpoint_dir / "checkpoint_latest.pt"
    torch.save(checkpoint, latest_path)
    logger.info(f"Checkpoint saved: {latest_path} (step {step})")


def train(config: Dict, args: argparse.Namespace) -> None:
    """Main training function."""

    # Setup output directory
    output_dir = Path(config.get("output_dir", "outputs/train/smolvla_new_dataset"))
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")

    # Save config
    config_save_path = output_dir / "config.yaml"
    with open(config_save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    logger.info(f"Config saved: {config_save_path}")

    # Set random seed
    seed = config.get("seed", 1000)
    torch.manual_seed(seed)
    logger.info(f"Random seed: {seed}")

    # Create dataset
    logger.info("Creating dataset...")
    dataset = create_dataset_from_config(config)

    # Create dataloader
    logger.info("Creating dataloader...")
    dataloader = create_dataloader_from_config(dataset, config, is_train=True)

    # Get sample batch for policy initialization
    logger.info("Getting sample batch...")
    sample_batch = next(iter(dataloader))

    # Create policy
    logger.info("Creating policy...")
    policy = create_policy_from_config(config, sample_batch)
    device = next(policy.parameters()).device

    # Create preprocessor and postprocessor
    logger.info("Creating preprocessor and postprocessor...")
    pretrained_model_id = "lerobot/smolvla_base"
    # Get the actual policy (unwrap DataParallel if needed)
    if isinstance(policy, nn.DataParallel):
        policy_for_config = policy.module.model
    else:
        policy_for_config = policy
    preprocessor, postprocessor = make_pre_post_processors(
        policy_for_config.config,
        pretrained_model_id,
        preprocessor_overrides={"device_processor": {"device": str(device)}}
    )
    logger.info("Preprocessor and postprocessor created")

    # Create optimizer
    logger.info("Creating optimizer...")
    optimizer = create_optimizer(policy, config)

    # Create scheduler
    logger.info("Creating scheduler...")
    scheduler = create_scheduler(optimizer, config)

    # Training parameters
    total_steps = config["training"]["steps"]
    log_freq = config["training"].get("log_freq", 100)
    save_freq = config["training"].get("save_freq", 2000)
    grad_clip_norm = config["training"].get("grad_clip_norm", 10.0)

    logger.info(f"Training for {total_steps} steps")
    logger.info(f"Log frequency: {log_freq}")
    logger.info(f"Save frequency: {save_freq}")

    # Training loop
    step = 0
    epoch = 0
    running_metrics = {}

    logger.info("Starting training...")

    while step < total_steps:
        epoch += 1
        logger.info(f"Epoch {epoch}")

        # Create progress bar
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}", leave=False)

        for batch in pbar:
            step += 1

            # Training step
            step_start = time.time()
            metrics = train_step(policy, batch, preprocessor, optimizer, device, grad_clip_norm)
            step_time = time.time() - step_start

            # Update scheduler
            scheduler.step()

            # Accumulate metrics
            for key, value in metrics.items():
                if key not in running_metrics:
                    running_metrics[key] = []
                running_metrics[key].append(value)

            # Logging
            if step % log_freq == 0:
                avg_metrics = {k: sum(v) / len(v) for k, v in running_metrics.items()}
                lr = optimizer.param_groups[0]['lr']

                log_msg = f"Step {step}/{total_steps} | "
                log_msg += f"Loss: {avg_metrics['loss']:.4f} | "
                log_msg += f"LR: {lr:.2e} | "
                log_msg += f"Time: {step_time:.3f}s"

                logger.info(log_msg)
                pbar.set_postfix(loss=avg_metrics['loss'], lr=lr)

                # Reset running metrics
                running_metrics = {}

            # Save checkpoint
            if step % save_freq == 0:
                save_checkpoint(policy, optimizer, scheduler, step, config, output_dir)

            # Check if training is complete
            if step >= total_steps:
                break

    # Final checkpoint
    logger.info("Training complete!")
    save_checkpoint(policy, optimizer, scheduler, step, config, output_dir)

    # Save final model (unwrap DataParallel if needed)
    final_model_path = output_dir / "final_model"
    if isinstance(policy, nn.DataParallel):
        model_to_save = policy.module.model
    else:
        model_to_save = policy
    model_to_save.save_pretrained(final_model_path)
    logger.info(f"Final model saved: {final_model_path}")


def main():
    parser = argparse.ArgumentParser(description="Train SmolVLA on New HDF5 VLA Dataset")

    parser.add_argument(
        "--config",
        type=str,
        default="train_config_new_dataset.yaml",
        help="Path to training configuration file"
    )

    # Override config options
    parser.add_argument("--batch_size", type=int, help="Override batch size")
    parser.add_argument("--steps", type=int, help="Override total training steps")
    parser.add_argument("--lr", type=float, help="Override learning rate")
    parser.add_argument("--device", type=str, help="Override device (cpu/cuda)")
    parser.add_argument("--output_dir", type=str, help="Override output directory")

    args = parser.parse_args()

    # Load config
    logger.info(f"Loading config: {args.config}")
    config = load_config(args.config)

    # Apply overrides
    if args.batch_size is not None:
        config["training"]["batch_size"] = args.batch_size
        logger.info(f"Override batch_size: {args.batch_size}")

    if args.steps is not None:
        config["training"]["steps"] = args.steps
        logger.info(f"Override steps: {args.steps}")

    if args.lr is not None:
        config["optimizer"]["lr"] = args.lr
        logger.info(f"Override lr: {args.lr}")

    if args.device is not None:
        config["policy"]["device"] = args.device
        logger.info(f"Override device: {args.device}")

    if args.output_dir is not None:
        config["output_dir"] = args.output_dir
        logger.info(f"Override output_dir: {args.output_dir}")

    # Start training
    try:
        train(config, args)
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
