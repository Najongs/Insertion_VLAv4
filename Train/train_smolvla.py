#!/usr/bin/env python
"""
SmolVLA Training script for Meca500 + Vision-Language-Action

Based on lerobot/scripts/lerobot_train.py with improvements:
- Data-driven action normalization (min/max from dataset statistics)
- Temporal consistency loss for smoother trajectories
- Expert-only training for memory efficiency
- Multi-camera support (3 OAK cameras)

Memory Requirements:
- Full Fine-Tuning: > 40 GB
- Expert-Only Training: ~12-16 GB (fits on RTX 3090/4090)

Usage:
    python train_smolvla.py --config train_config_smolvla.yaml

DDP Usage:
    torchrun --nproc_per_node=5 train_smolvla.py --config train_config_smolvla.yaml
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional
from contextlib import nullcontext
import yaml

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.amp import GradScaler
from tqdm import tqdm
import numpy as np

# W&B for experiment tracking
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    logger.warning("wandb not installed. Install with: pip install wandb")

# LeRobot imports
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.factory import make_pre_post_processors
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.utils.utils import init_logging
from lerobot.policies.utils import get_device_from_parameters

# HDF5 VLA imports
from hdf5_lerobot_adapter import create_hdf5_lerobot_dataset, hdf5_lerobot_collate_fn

# Initialize logging
init_logging()
logger = logging.getLogger(__name__)


def setup_distributed():
    """Initialize distributed training environment (following lerobot_train.py)."""
    if torch.cuda.is_available() and int(os.environ.get("WORLD_SIZE", 1)) > 1:
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        logger.info(f"[DDP INIT] rank={rank}/{world_size}, local_rank={local_rank}")
        return rank, world_size, local_rank, True
    else:
        return 0, 1, 0, False


def cleanup_distributed():
    """Cleanup distributed training."""
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def is_main_process():
    """Check if current process is main process (rank 0)."""
    return not dist.is_initialized() or dist.get_rank() == 0


def setup_wandb(config: Dict, resume: bool = False):
    """
    Initialize Weights & Biases for experiment tracking.

    Args:
        config: Full training configuration
        resume: Whether to resume from previous run

    Returns:
        wandb run object or None if wandb not available
    """
    if not WANDB_AVAILABLE:
        logger.warning("W&B not available, skipping W&B initialization")
        return None

    if not is_main_process():
        return None

    wandb_cfg = config.get("wandb", {})

    # Check if W&B is enabled
    if not wandb_cfg.get("enable", False):
        logger.info("W&B disabled in config")
        return None

    # W&B initialization
    project = wandb_cfg.get("project", "smolvla-training")
    entity = wandb_cfg.get("entity", None)
    name = wandb_cfg.get("name", None)
    tags = wandb_cfg.get("tags", [])
    notes = wandb_cfg.get("notes", "")

    # Auto-generate name if not provided
    if name is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        name = f"smolvla_{timestamp}"

    logger.info(f"Initializing W&B: project={project}, name={name}")

    try:
        run = wandb.init(
            project=project,
            entity=entity,
            name=name,
            tags=tags,
            notes=notes,
            config=config,
            resume="allow" if resume else False,
        )

        logger.info(f"W&B initialized successfully: {run.url}")
        return run

    except Exception as e:
        logger.error(f"Failed to initialize W&B: {e}")
        return None


def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_dataset_from_config(config: Dict) -> torch.utils.data.Dataset:
    """Create HDF5 LeRobot dataset from configuration."""
    dataset_cfg = config["dataset"]
    policy_cfg = config["policy"]

    # Build full HDF5 file paths
    root_dir = Path(dataset_cfg["root_dir"])
    hdf5_files = []

    # Support both specific file list and pattern matching
    if "hdf5_files" in dataset_cfg:
        hdf5_files = [root_dir / f for f in dataset_cfg["hdf5_files"]]
    else:
        hdf5_files = sorted(list(root_dir.glob("*.h5")) + list(root_dir.glob("*.hdf5")))

    if len(hdf5_files) == 0:
        raise FileNotFoundError(f"No HDF5 files found in {root_dir}")

    if is_main_process():
        logger.info(f"Found {len(hdf5_files)} HDF5 files")

    # Create tokenizer for language conditioning
    from transformers import AutoTokenizer
    tokenizer_max_length = policy_cfg.get("tokenizer_max_length", 48)
    vlm_model_name = policy_cfg.get("vlm_model_name", "HuggingFaceTB/SmolVLM2-500M-Video-Instruct")
    tokenizer = AutoTokenizer.from_pretrained(vlm_model_name)
    if is_main_process():
        logger.info(f"Created tokenizer from {vlm_model_name} (max_length={tokenizer_max_length})")

    # Create dataset
    dataset = create_hdf5_lerobot_dataset(
        hdf5_paths=hdf5_files,
        horizon=dataset_cfg.get("horizon", 50),
        n_obs_steps=policy_cfg.get("n_obs_steps", 1),
        squeeze_n_obs_steps=(policy_cfg.get("n_obs_steps", 1) == 1),
        use_qpos=dataset_cfg.get("use_qpos", False),
        use_ee_pose=dataset_cfg.get("use_ee_pose", True),
        task_instruction=dataset_cfg.get("task_instruction", "Insert needle into red point"),
        tokenizer=tokenizer,
        tokenizer_max_length=tokenizer_max_length,
        augment=dataset_cfg.get("augment", True),
        augment_brightness=dataset_cfg.get("augment_brightness", 0.15),
        augment_contrast=dataset_cfg.get("augment_contrast", 0.15),
        augment_saturation=dataset_cfg.get("augment_saturation", 0.10),
        augment_hue=dataset_cfg.get("augment_hue", 0.05),
        augment_noise=dataset_cfg.get("augment_noise", 0.01),
    )

    if is_main_process():
        logger.info(f"Dataset created: {len(dataset)} samples")

    return dataset


def create_policy_from_config(config: Dict, device: torch.device) -> SmolVLAPolicy:
    """Create SmolVLA policy from configuration."""
    policy_cfg = config["policy"]

    if is_main_process():
        logger.info("=" * 80)
        logger.info("Creating SmolVLA Policy")
        logger.info("=" * 80)
        logger.info(f"vlm_model_name: {policy_cfg.get('vlm_model_name', 'HuggingFaceTB/SmolVLM2-500M-Video-Instruct')}")
        logger.info(f"n_obs_steps: {policy_cfg.get('n_obs_steps', 1)}")
        logger.info(f"chunk_size: {policy_cfg.get('chunk_size', 50)}")
        logger.info(f"freeze_vision_encoder: {policy_cfg.get('freeze_vision_encoder', True)}")
        logger.info(f"train_expert_only: {policy_cfg.get('train_expert_only', True)}")

    # Create SmolVLAConfig
    smolvla_config = SmolVLAConfig(
        vlm_model_name=policy_cfg.get("vlm_model_name", "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"),
        n_obs_steps=policy_cfg.get("n_obs_steps", 1),
        chunk_size=policy_cfg.get("chunk_size", 50),
        n_action_steps=policy_cfg.get("n_action_steps", 50),
        max_state_dim=policy_cfg.get("max_state_dim", 32),
        max_action_dim=policy_cfg.get("max_action_dim", 32),
        num_steps=policy_cfg.get("num_steps", 10),
        freeze_vision_encoder=policy_cfg.get("freeze_vision_encoder", True),
        train_expert_only=policy_cfg.get("train_expert_only", True),
        train_state_proj=policy_cfg.get("train_state_proj", True),
        tokenizer_max_length=policy_cfg.get("tokenizer_max_length", 48),
        resize_imgs_with_padding=tuple(policy_cfg.get("resize_imgs_with_padding", [512, 512])),
        input_features={
            "observation.images.camera1": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 512, 512)),
            "observation.images.camera2": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 512, 512)),
            "observation.images.camera3": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 512, 512)),
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(6,)),
        },
        output_features={
            "action": PolicyFeature(type=FeatureType.ACTION, shape=(6,)),
        },
    )

    # Create policy
    policy = SmolVLAPolicy(smolvla_config)
    policy.to(device)

    if is_main_process():
        logger.info(f"Policy created and moved to {device}")
        logger.info("=" * 80)

    return policy


def compute_temporal_consistency_loss(actions: torch.Tensor, lambda_temporal: float = 0.1) -> torch.Tensor:
    """
    Compute temporal consistency loss to encourage smooth action trajectories.
    """
    if actions.shape[1] <= 1:
        return torch.tensor(0.0, device=actions.device)

    action_diffs = actions[:, 1:, :] - actions[:, :-1, :]
    temporal_loss = torch.mean(torch.square(action_diffs))

    return lambda_temporal * temporal_loss


def create_optimizer_from_config(policy: nn.Module, config: Dict) -> AdamW:
    """Create AdamW optimizer."""
    optimizer_cfg = config["optimizer"]

    trainable_params = [p for p in policy.parameters() if p.requires_grad]

    if is_main_process():
        total_params = sum(p.numel() for p in policy.parameters())
        trainable_params_count = sum(p.numel() for p in trainable_params)
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params_count:,}")
        logger.info(f"Trainable %: {100 * trainable_params_count / total_params:.2f}%")

    optimizer = AdamW(
        trainable_params,
        lr=optimizer_cfg.get("lr", 1e-4),
        betas=tuple(optimizer_cfg.get("betas", [0.9, 0.95])),
        eps=optimizer_cfg.get("eps", 1e-8),
        weight_decay=optimizer_cfg.get("weight_decay", 1e-10),
    )

    return optimizer


def create_scheduler_from_config(optimizer: AdamW, config: Dict) -> LambdaLR:
    """Create learning rate scheduler with warmup and cosine decay."""
    scheduler_cfg = config["scheduler"]

    num_warmup_steps = scheduler_cfg.get("num_warmup_steps", 1000)
    num_decay_steps = scheduler_cfg.get("num_decay_steps", 30000)
    peak_lr = scheduler_cfg.get("peak_lr", 1e-4)
    decay_lr = scheduler_cfg.get("decay_lr", 2.5e-6)

    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        else:
            progress = float(current_step - num_warmup_steps) / float(max(1, num_decay_steps - num_warmup_steps))
            progress = min(progress, 1.0)
            cosine_decay = 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.14159)))
            return (decay_lr / peak_lr) + (1.0 - decay_lr / peak_lr) * cosine_decay

    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
    return scheduler


def update_policy(
    policy: nn.Module,
    batch: dict,
    optimizer: AdamW,
    grad_scaler: GradScaler,
    grad_clip_norm: float,
    lr_scheduler: LambdaLR,
    lambda_temporal: float,
    use_amp: bool = False,
):
    """
    Performs a single training step (following lerobot_train.py structure).
    """
    device = get_device_from_parameters(policy)
    policy.train()

    with torch.autocast(device_type=device.type) if use_amp else nullcontext():
        # Forward pass - SmolVLA returns (loss, loss_dict) tuple
        loss, loss_dict = policy.forward(batch)
        action_loss = loss

        # Compute temporal consistency loss on ground truth actions
        if "action" in batch and lambda_temporal > 0:
            gt_actions = batch["action"]
            temporal_loss = compute_temporal_consistency_loss(gt_actions, lambda_temporal)
            loss = action_loss + temporal_loss
        else:
            temporal_loss = torch.tensor(0.0, device=device)

    grad_scaler.scale(loss).backward()
    grad_scaler.unscale_(optimizer)

    grad_norm = torch.nn.utils.clip_grad_norm_(
        policy.parameters(),
        grad_clip_norm,
        error_if_nonfinite=False,
    )

    grad_scaler.step(optimizer)
    grad_scaler.update()
    optimizer.zero_grad()

    if lr_scheduler is not None:
        lr_scheduler.step()

    return loss.item(), action_loss.item(), temporal_loss.item(), grad_norm.item()


def train(
    policy: nn.Module,
    dataloader: DataLoader,
    optimizer: AdamW,
    scheduler: LambdaLR,
    grad_scaler: GradScaler,
    config: Dict,
    rank: int,
    world_size: int,
    wandb_run=None,
):
    """Main training loop (following lerobot_train.py structure)."""
    training_cfg = config["training"]
    optimizer_cfg = config["optimizer"]

    total_steps = training_cfg.get("steps", 30000)
    log_freq = training_cfg.get("log_freq", 100)
    save_freq = training_cfg.get("save_freq", 5000)
    grad_clip_norm = optimizer_cfg.get("grad_clip_norm", 10.0)
    lambda_temporal = training_cfg.get("lambda_temporal", 0.1)

    output_dir = Path(config["output_dir"])
    checkpoints_dir = output_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    policy.train()
    step = 0
    running_loss = 0.0
    running_action_loss = 0.0
    running_temporal_loss = 0.0
    running_grad_norm = 0.0

    # W&B: Log model architecture (main process only)
    if is_main_process() and wandb_run is not None:
        try:
            wandb.watch(policy, log="all", log_freq=log_freq)
        except Exception as e:
            logger.warning(f"Failed to setup wandb.watch: {e}")

    if is_main_process():
        logger.info("=" * 80)
        logger.info("Starting SmolVLA Expert-Only Training")
        logger.info("=" * 80)
        logger.info(f"Total steps: {total_steps}")
        logger.info(f"Batch size per GPU: {training_cfg.get('batch_size', 8)}")
        logger.info(f"Effective batch size: {training_cfg.get('batch_size', 8) * world_size}")
        logger.info(f"Number of GPUs: {world_size}")
        logger.info(f"Gradient clip norm: {grad_clip_norm}")
        logger.info(f"Temporal consistency weight (λ): {lambda_temporal}")
        if wandb_run is not None:
            logger.info(f"W&B tracking: {wandb_run.url}")
        logger.info("=" * 80)

    start_time = time.time()

    # Infinite dataloader (cycle)
    from itertools import cycle
    dl_iter = cycle(dataloader)

    for step in range(total_steps):
        # Get batch
        batch = next(dl_iter)

        # Move batch to device
        batch = {k: v.to(get_device_from_parameters(policy)) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}

        # Update policy
        loss, action_loss, temporal_loss, grad_norm = update_policy(
            policy,
            batch,
            optimizer,
            grad_scaler,
            grad_clip_norm,
            scheduler,
            lambda_temporal,
            use_amp=False,
        )

        # Accumulate losses
        running_loss += loss
        running_action_loss += action_loss
        running_temporal_loss += temporal_loss
        running_grad_norm += grad_norm

        # Logging
        if (step + 1) % log_freq == 0 and is_main_process():
            avg_loss = running_loss / log_freq
            avg_action_loss = running_action_loss / log_freq
            avg_temporal_loss = running_temporal_loss / log_freq
            avg_grad_norm = running_grad_norm / log_freq
            elapsed_time = time.time() - start_time
            steps_per_sec = log_freq / elapsed_time
            lr = scheduler.get_last_lr()[0]

            log_msg = (
                f"Step {step + 1}/{total_steps} | "
                f"Loss: {avg_loss:.4f} | "
                f"Action: {avg_action_loss:.4f} | "
                f"Temporal: {avg_temporal_loss:.4f} | "
                f"GradNorm: {avg_grad_norm:.3f} | "
                f"LR: {lr:.2e} | "
                f"Steps/sec: {steps_per_sec:.2f}"
            )

            logger.info(log_msg)

            # W&B logging
            if wandb_run is not None:
                wandb.log({
                    "train/loss": avg_loss,
                    "train/action_loss": avg_action_loss,
                    "train/temporal_loss": avg_temporal_loss,
                    "train/grad_norm": avg_grad_norm,
                    "train/learning_rate": lr,
                    "train/steps_per_sec": steps_per_sec,
                    "step": step + 1,
                })

            # Reset
            running_loss = 0.0
            running_action_loss = 0.0
            running_temporal_loss = 0.0
            running_grad_norm = 0.0
            start_time = time.time()

        # Save checkpoint
        if (step + 1) % save_freq == 0 and is_main_process():
            checkpoint_dir = checkpoints_dir / f"checkpoint_step_{step + 1}"

            # Get the actual policy (unwrap DDP if necessary)
            policy_to_save = policy.module if isinstance(policy, DDP) else policy
            policy_to_save.save_pretrained(checkpoint_dir)
            logger.info(f"Saved checkpoint to {checkpoint_dir}")

            # W&B: Upload checkpoint as artifact
            if wandb_run is not None:
                try:
                    artifact = wandb.Artifact(
                        name=f"checkpoint_step_{step + 1}",
                        type="model",
                        description=f"SmolVLA checkpoint at step {step + 1}",
                        metadata={
                            "step": step + 1,
                            "loss": avg_loss,
                            "action_loss": avg_action_loss,
                            "temporal_loss": avg_temporal_loss,
                        }
                    )
                    artifact.add_dir(str(checkpoint_dir))
                    wandb.log_artifact(artifact)
                    logger.info(f"Uploaded checkpoint to W&B")
                except Exception as e:
                    logger.warning(f"Failed to upload checkpoint to W&B: {e}")

    # Save final checkpoint
    if is_main_process():
        final_checkpoint_path = checkpoints_dir / "final"
        policy_to_save = policy.module if isinstance(policy, DDP) else policy
        policy_to_save.save_pretrained(final_checkpoint_path)
        logger.info(f"Saved final checkpoint to {final_checkpoint_path}")
        logger.info("=" * 80)
        logger.info("Training completed!")
        logger.info("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Train SmolVLA Policy on HDF5 VLA Dataset")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML file")
    parser.add_argument("--batch_size", type=int, default=None, help="Override batch size")
    parser.add_argument("--steps", type=int, default=None, help="Override training steps")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--output_dir", type=str, default=None, help="Override output directory")
    args = parser.parse_args()

    # Setup distributed training (following lerobot_train.py)
    rank, world_size, local_rank, is_distributed = setup_distributed()

    # Load config
    config = load_config(args.config)

    # Override config with command line arguments
    if args.batch_size is not None:
        config["training"]["batch_size"] = args.batch_size
    if args.steps is not None:
        config["training"]["steps"] = args.steps
    if args.lr is not None:
        config["optimizer"]["lr"] = args.lr
    if args.output_dir is not None:
        config["output_dir"] = args.output_dir

    # Set random seed
    seed = config.get("seed", 1000)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Get device
    device = torch.device(f"cuda:{local_rank}") if torch.cuda.is_available() else torch.device("cpu")

    if is_main_process():
        logger.info(f"Using device: {device}")

    # Create dataset
    dataset = create_dataset_from_config(config)

    # Create dataloader with distributed sampler (following lerobot_train.py)
    sampler = DistributedSampler(dataset, shuffle=True) if is_distributed else None
    dataloader = DataLoader(
        dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=(sampler is None),
        num_workers=config["training"].get("num_workers", 4),
        pin_memory=(device.type == "cuda"),
        sampler=sampler,
        collate_fn=hdf5_lerobot_collate_fn,
        drop_last=False,
        prefetch_factor=2,
    )

    # Create policy
    policy = create_policy_from_config(config, device)

    # Wrap policy with DDP if using distributed training (following lerobot_train.py)
    if is_distributed:
        policy = DDP(
            policy,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False,
        )
        if is_main_process():
            logger.info("Policy wrapped with DistributedDataParallel")

    # Create optimizer and scheduler
    optimizer = create_optimizer_from_config(policy, config)
    scheduler = create_scheduler_from_config(optimizer, config)
    grad_scaler = GradScaler(device.type, enabled=False)

    # Setup W&B (main process only)
    wandb_run = None
    if is_main_process():
        wandb_run = setup_wandb(config)

    # Train
    try:
        train(policy, dataloader, optimizer, scheduler, grad_scaler, config, rank, world_size, wandb_run)
    except KeyboardInterrupt:
        if is_main_process():
            logger.info("\nTraining interrupted by user")
    except Exception as e:
        logger.error(f"Training failed with error: {e}", exc_info=True)
        raise
    finally:
        # Finish W&B run
        if is_main_process() and wandb_run is not None:
            wandb.finish()
        cleanup_distributed()


if __name__ == "__main__":
    main()
