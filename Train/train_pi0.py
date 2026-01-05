#!/usr/bin/env python
"""
Training script for π0 (Pi Zero) Policy on New HDF5 VLA Dataset

This script trains π0 for precise needle insertion task.
π0 uses flow matching and vision-language conditioning.

Usage:
    python train_pi0.py --config train_config_pi0.yaml

DDP Usage:
    torchrun --nproc_per_node=5 train_pi0.py --config train_config_pi0.yaml
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional
import yaml

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

# LeRobot imports
from lerobot.policies.pi0.modeling_pi0 import PI0Policy
from lerobot.policies.pi0.configuration_pi0 import PI0Config
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.utils.utils import get_safe_torch_device, init_logging

# HDF5 VLA imports
from hdf5_lerobot_adapter import create_hdf5_lerobot_dataset, hdf5_lerobot_collate_fn

# Initialize logging
init_logging()
logger = logging.getLogger(__name__)


def setup_distributed():
    """Initialize distributed training environment."""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
    else:
        # Not running with torchrun, single GPU mode
        rank = 0
        world_size = 1
        local_rank = 0

    if world_size > 1:
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(local_rank)
        logger.info(f"Distributed training initialized: rank {rank}/{world_size}, local_rank {local_rank}")
    else:
        logger.info("Single GPU training mode")

    return rank, world_size, local_rank


def cleanup_distributed():
    """Cleanup distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process():
    """Check if current process is main process (rank 0)."""
    return not dist.is_initialized() or dist.get_rank() == 0


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
        # Specific file list
        hdf5_files = [root_dir / f for f in dataset_cfg["hdf5_files"]]
    else:
        # Use all HDF5 files in directory
        hdf5_files = sorted(root_dir.glob("*.h5"))

    logger.info(f"Creating dataset with {len(hdf5_files)} HDF5 episodes")

    # Get n_obs_steps from policy config (π0 uses 1)
    n_obs_steps = policy_cfg.get("n_obs_steps", 1)
    logger.info(f"Using n_obs_steps={n_obs_steps} for π0 (will be squeezed)")

    # Create dataset with squeeze enabled for π0 compatibility
    dataset = create_hdf5_lerobot_dataset(
        hdf5_paths=hdf5_files,
        horizon=dataset_cfg.get("horizon", 1),
        n_obs_steps=n_obs_steps,
        use_qpos=dataset_cfg.get("use_qpos", False),
        use_ee_pose=dataset_cfg.get("use_ee_pose", True),
        task_instruction=dataset_cfg.get("task_instruction", "Insert needle"),
        # Augmentation
        augment=dataset_cfg.get("augment", False),
        augment_brightness=dataset_cfg.get("augment_brightness", 0.0),
        augment_contrast=dataset_cfg.get("augment_contrast", 0.0),
        augment_saturation=dataset_cfg.get("augment_saturation", 0.0),
        augment_hue=dataset_cfg.get("augment_hue", 0.0),
        augment_noise=dataset_cfg.get("augment_noise", 0.0),
        # Squeeze temporal dimension for π0 (n_obs_steps=1 -> no temporal dimension)
        squeeze_n_obs_steps=True,
    )

    logger.info(f"Dataset created with {len(dataset)} samples")

    # Log dataset statistics
    if len(dataset) > 0:
        sample = dataset[0]
        logger.info("Sample keys:")
        for key in sample.keys():
            if hasattr(sample[key], 'shape'):
                logger.info(f"  {key}: {sample[key].shape}")
            else:
                logger.info(f"  {key}: {type(sample[key])}")

    return dataset


def create_dataloader(
    dataset: torch.utils.data.Dataset,
    batch_size: int,
    num_workers: int,
    shuffle: bool = True,
    pin_memory: bool = True,
    use_distributed: bool = False
) -> DataLoader:
    """Create DataLoader with optional distributed sampler."""
    sampler = DistributedSampler(dataset, shuffle=shuffle) if use_distributed else None

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(shuffle and sampler is None),
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=hdf5_lerobot_collate_fn,
        drop_last=True,
    )


def create_policy_from_config(config: Dict, device: torch.device) -> PI0Policy:
    """Create and configure π0 Policy."""
    policy_cfg = config["policy"]

    logger.info("Creating π0 Policy from scratch...")

    # Create config for π0 Policy
    pi0_config = PI0Config(
        # Model architecture
        paligemma_variant=policy_cfg.get("paligemma_variant", "gemma_2b"),
        action_expert_variant=policy_cfg.get("action_expert_variant", "gemma_300m"),
        dtype=policy_cfg.get("dtype", "float32"),

        # Observation and action
        n_obs_steps=policy_cfg.get("n_obs_steps", 1),
        chunk_size=policy_cfg.get("chunk_size", 50),
        n_action_steps=policy_cfg.get("n_action_steps", 50),

        # Dimensions
        max_state_dim=policy_cfg.get("max_state_dim", 32),
        max_action_dim=policy_cfg.get("max_action_dim", 32),

        # Flow matching parameters
        num_inference_steps=policy_cfg.get("num_inference_steps", 10),
        time_sampling_beta_alpha=policy_cfg.get("time_sampling_beta_alpha", 1.5),
        time_sampling_beta_beta=policy_cfg.get("time_sampling_beta_beta", 1.0),
        time_sampling_scale=policy_cfg.get("time_sampling_scale", 0.999),
        time_sampling_offset=policy_cfg.get("time_sampling_offset", 0.001),
        min_period=policy_cfg.get("min_period", 0.004),
        max_period=policy_cfg.get("max_period", 4.0),

        # Image resolution
        image_resolution=tuple(policy_cfg.get("image_resolution", [224, 224])),

        # Training optimization
        gradient_checkpointing=policy_cfg.get("gradient_checkpointing", False),
        compile_model=policy_cfg.get("compile_model", False),

        # Define input/output features
        input_features={
            "observation.images.camera1": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224)),
            "observation.images.camera2": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224)),
            "observation.images.camera3": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224)),
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(6,)),
        },
        output_features={
            "action": PolicyFeature(type=FeatureType.ACTION, shape=(6,)),
        },
    )

    logger.info(f"  n_obs_steps: {pi0_config.n_obs_steps}")
    logger.info(f"  chunk_size: {pi0_config.chunk_size}")
    logger.info(f"  n_action_steps: {pi0_config.n_action_steps}")
    logger.info(f"  paligemma_variant: {pi0_config.paligemma_variant}")
    logger.info(f"  action_expert_variant: {pi0_config.action_expert_variant}")

    # Create policy
    policy = PI0Policy(pi0_config)

    # Move to device
    policy.to(device)

    return policy


def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    peak_lr: float,
    final_lr: float,
) -> LambdaLR:
    """Cosine learning rate schedule with warmup."""
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            # Linear warmup
            return float(current_step) / float(max(1, num_warmup_steps))
        # Cosine decay
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        cosine_decay = 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.141592653589793)))
        return (final_lr / peak_lr) + (1.0 - final_lr / peak_lr) * cosine_decay

    return LambdaLR(optimizer, lr_lambda)


def save_checkpoint(
    step: int,
    epoch: int,
    policy: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[LambdaLR],
    config: Dict,
    output_dir: Path,
    is_best: bool = False,
):
    """Save training checkpoint."""
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Extract state dict (handle DDP wrapper)
    if isinstance(policy, DDP):
        policy_state_dict = policy.module.state_dict()
    else:
        policy_state_dict = policy.state_dict()

    checkpoint = {
        "step": step,
        "epoch": epoch,
        "policy_state_dict": policy_state_dict,
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "config": config,
    }

    # Save latest checkpoint
    checkpoint_path = checkpoint_dir / "checkpoint_latest.pt"
    torch.save(checkpoint, checkpoint_path)
    logger.info(f"Saved checkpoint to {checkpoint_path}")

    # Save best checkpoint
    if is_best:
        best_path = checkpoint_dir / "checkpoint_best.pt"
        torch.save(checkpoint, best_path)
        logger.info(f"Saved best checkpoint to {best_path}")

    # Save periodic checkpoint
    if step % 5000 == 0:
        periodic_path = checkpoint_dir / f"checkpoint_step_{step}.pt"
        torch.save(checkpoint, periodic_path)


def train_one_epoch(
    epoch: int,
    policy: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[LambdaLR],
    device: torch.device,
    config: Dict,
    output_dir: Path,
    global_step: int,
    max_steps: Optional[int] = None,
) -> int:
    """Train for one epoch."""
    policy.train()

    training_cfg = config["training"]
    optimizer_cfg = config["optimizer"]
    log_freq = training_cfg.get("log_freq", 100)
    grad_clip_norm = optimizer_cfg.get("grad_clip_norm", 1.0)

    # Progress bar (only on main process)
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}", disable=not is_main_process())

    running_loss = 0.0
    running_count = 0

    for batch_idx, batch in enumerate(pbar):
        # Move batch to device
        batch = {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()}

        # Forward pass (returns tuple: (loss, loss_dict))
        loss, loss_dict = policy.forward(batch)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        if grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(policy.parameters(), grad_clip_norm)

        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        # Logging
        running_loss += loss.item()
        running_count += 1
        global_step += 1

        if global_step % log_freq == 0 and is_main_process():
            avg_loss = running_loss / running_count
            current_lr = optimizer.param_groups[0]['lr']

            logger.info(
                f"Step {global_step} | Epoch {epoch} | "
                f"Loss: {avg_loss:.6f} | LR: {current_lr:.2e}"
            )

            running_loss = 0.0
            running_count = 0

        # Update progress bar
        if is_main_process():
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "step": global_step})

        # Check if max steps reached
        if max_steps is not None and global_step >= max_steps:
            logger.info(f"Reached max steps: {max_steps}")
            break

    return global_step


def main():
    parser = argparse.ArgumentParser(description="Train π0 Policy on HDF5 VLA Dataset")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML file")
    parser.add_argument("--batch_size", type=int, default=None, help="Override batch size")
    parser.add_argument("--steps", type=int, default=None, help="Override max training steps")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--output_dir", type=str, default=None, help="Override output directory")
    args = parser.parse_args()

    # Setup distributed training
    rank, world_size, local_rank = setup_distributed()

    try:
        # Load config
        config = load_config(args.config)

        # Override config with CLI args
        if args.batch_size:
            config["training"]["batch_size"] = args.batch_size
        if args.steps:
            config["training"]["steps"] = args.steps
        if args.lr:
            config["optimizer"]["lr"] = args.lr
        if args.output_dir:
            config["output_dir"] = args.output_dir

        output_dir = Path(config["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save config
        if is_main_process():
            config_save_path = output_dir / "config.yaml"
            with open(config_save_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)
            logger.info(f"Saved config to {config_save_path}")

        # Set random seed
        seed = config.get("seed", 1000)
        torch.manual_seed(seed + rank)

        # Create dataset
        dataset = create_dataset_from_config(config)

        # Create dataloader
        training_cfg = config["training"]
        dataloader = create_dataloader(
            dataset,
            batch_size=training_cfg["batch_size"],
            num_workers=training_cfg.get("num_workers", 4),
            shuffle=training_cfg.get("shuffle", True),
            pin_memory=training_cfg.get("pin_memory", True),
            use_distributed=(world_size > 1),
        )

        # Create device
        if world_size > 1:
            device = torch.device(f"cuda:{local_rank}")
        else:
            device = get_safe_torch_device("cuda" if torch.cuda.is_available() else "cpu")

        logger.info(f"Using device: {device}")

        # Create policy
        policy = create_policy_from_config(config, device)

        # Wrap with DDP
        if world_size > 1:
            policy = DDP(policy, device_ids=[local_rank], output_device=local_rank)
            logger.info(f"Wrapped policy with DDP on rank {rank}")

        # Create optimizer
        optimizer_cfg = config["optimizer"]
        optimizer = AdamW(
            policy.parameters(),
            lr=optimizer_cfg["lr"],
            betas=optimizer_cfg.get("betas", [0.9, 0.95]),
            weight_decay=optimizer_cfg.get("weight_decay", 0.01),
            eps=optimizer_cfg.get("eps", 1e-8),
        )

        # Create scheduler
        scheduler_cfg = config.get("scheduler", {})
        max_steps = training_cfg["steps"]
        num_warmup_steps = scheduler_cfg.get("num_warmup_steps", max_steps // 10)

        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=max_steps,
            peak_lr=optimizer_cfg["lr"],
            final_lr=scheduler_cfg.get("decay_lr", optimizer_cfg["lr"] * 0.1),
        )

        logger.info(f"Training for {max_steps} steps with {num_warmup_steps} warmup steps")

        # Training loop
        global_step = 0
        epoch = 0

        while global_step < max_steps:
            if world_size > 1:
                dataloader.sampler.set_epoch(epoch)

            global_step = train_one_epoch(
                epoch=epoch,
                policy=policy,
                dataloader=dataloader,
                optimizer=optimizer,
                scheduler=scheduler,
                device=device,
                config=config,
                output_dir=output_dir,
                global_step=global_step,
                max_steps=max_steps,
            )

            # Save checkpoint
            if is_main_process():
                save_checkpoint(
                    step=global_step,
                    epoch=epoch,
                    policy=policy,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    config=config,
                    output_dir=output_dir,
                )

            epoch += 1

        # Save final model
        if is_main_process():
            final_dir = output_dir / "final_model"
            final_dir.mkdir(parents=True, exist_ok=True)

            if hasattr(policy, 'module'):
                policy.module.save_pretrained(final_dir)
            else:
                policy.save_pretrained(final_dir)

            logger.info(f"Saved final model to {final_dir}")

        logger.info("Training completed successfully!")

    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
