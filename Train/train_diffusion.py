#!/usr/bin/env python
"""
Training script for Diffusion Policy on New HDF5 VLA Dataset

This script trains Diffusion Policy for precise needle insertion task.
Diffusion Policy excels at contact-rich manipulation and smooth trajectories.

Usage:
    python train_diffusion.py --config train_config_diffusion.yaml

DDP Usage:
    torchrun --nproc_per_node=5 train_diffusion.py --config train_config_diffusion.yaml
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
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.policies.factory import make_pre_post_processors
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

    # Get n_obs_steps from policy config
    n_obs_steps = policy_cfg.get("n_obs_steps", 1)
    logger.info(f"Using n_obs_steps={n_obs_steps} for temporal observation stacking")

    # Create dataset
    dataset = create_hdf5_lerobot_dataset(
        hdf5_paths=hdf5_files,  # Fixed: hdf5_files -> hdf5_paths
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


def create_policy_from_config(config: Dict, device: torch.device) -> DiffusionPolicy:
    """Create and configure Diffusion Policy."""
    from dataclasses import replace

    policy_cfg = config["policy"]

    # Option 1: Load from pretrained (if available)
    # Option 2: Create from scratch with config
    pretrained_model_id = policy_cfg.get("pretrained_model_id", None)

    if pretrained_model_id:
        logger.info(f"Loading Diffusion Policy from: {pretrained_model_id}")
        try:
            pretrained_policy = DiffusionPolicy.from_pretrained(pretrained_model_id)

            # Create new config with our dataset's camera names
            logger.info("Creating new config with our settings...")
            new_config = replace(
                pretrained_policy.config,
                n_obs_steps=policy_cfg.get("n_obs_steps", 2),
                horizon=policy_cfg.get("horizon", 16),
                n_action_steps=policy_cfg.get("n_action_steps", 8),
                num_inference_steps=policy_cfg.get("num_inference_steps", 100),
                down_dims=tuple(policy_cfg.get("down_dims", [256, 512, 1024])),
            )

            # Create new policy with updated config
            logger.info("Creating new policy with updated config...")
            policy = DiffusionPolicy(new_config)

            # Try to load compatible weights from pretrained model
            logger.info("Loading compatible weights from pretrained model...")
            try:
                pretrained_state = pretrained_policy.state_dict()
                policy_state = policy.state_dict()

                # Copy compatible weights
                loaded_keys = []
                for key in policy_state.keys():
                    if key in pretrained_state and policy_state[key].shape == pretrained_state[key].shape:
                        policy_state[key] = pretrained_state[key]
                        loaded_keys.append(key)

                policy.load_state_dict(policy_state, strict=False)
                logger.info(f"  Loaded {len(loaded_keys)}/{len(policy_state)} compatible weights")
            except Exception as e:
                logger.warning(f"  Could not load pretrained weights: {e}")
                logger.warning("  Training from scratch...")

        except Exception as e:
            logger.warning(f"Failed to load pretrained model: {e}")
            logger.info("Creating policy from scratch...")
            policy = create_policy_from_scratch(policy_cfg)
    else:
        logger.info("Creating Diffusion Policy from scratch...")
        policy = create_policy_from_scratch(policy_cfg)

    logger.info(f"  n_obs_steps: {policy.config.n_obs_steps}")
    logger.info(f"  horizon: {policy.config.horizon}")
    logger.info(f"  n_action_steps: {policy.config.n_action_steps}")
    logger.info(f"  num_inference_steps: {policy.config.num_inference_steps}")
    logger.info(f"  down_dims: {policy.config.down_dims}")

    # Move to device
    policy.to(device)

    return policy


def create_policy_from_scratch(policy_cfg: Dict) -> DiffusionPolicy:
    """Create Diffusion Policy from configuration (no pretrained weights)."""
    from lerobot.configs.types import FeatureType, PolicyFeature

    # Create config for Diffusion Policy
    config = DiffusionConfig(
        n_obs_steps=policy_cfg.get("n_obs_steps", 2),
        horizon=policy_cfg.get("horizon", 16),
        n_action_steps=policy_cfg.get("n_action_steps", 8),
        num_inference_steps=policy_cfg.get("num_inference_steps", 100),
        down_dims=tuple(policy_cfg.get("down_dims", [256, 512, 1024])),
        input_features={
            "observation.images.camera1": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 480, 640)),
            "observation.images.camera2": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 480, 640)),
            "observation.images.camera3": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 480, 640)),
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(6,)),
        },
        output_features={
            "action": PolicyFeature(type=FeatureType.ACTION, shape=(6,)),
        },
    )

    return DiffusionPolicy(config)


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
    log_freq = training_cfg.get("log_freq", 100)
    grad_clip_norm = training_cfg.get("grad_clip_norm", 10.0)

    # Progress bar (only on main process)
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}", disable=not is_main_process())

    running_loss = 0.0
    running_count = 0

    for batch_idx, batch in enumerate(pbar):
        # Move batch to device
        batch = {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()}

        # Forward pass (returns tuple: (loss, None))
        loss, _ = policy.forward(batch)

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
    parser = argparse.ArgumentParser(description="Train Diffusion Policy on HDF5 VLA Dataset")
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
        if hasattr(policy, 'module'):
            optim_params = policy.module.get_optim_params()
        else:
            optim_params = policy.get_optim_params()

        optimizer_cfg = config["optimizer"]
        optimizer = AdamW(
            optim_params,
            lr=optimizer_cfg["lr"],
            betas=optimizer_cfg.get("betas", [0.9, 0.999]),
            weight_decay=optimizer_cfg.get("weight_decay", 1e-6),
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
            final_lr=scheduler_cfg.get("decay_lr", optimizer_cfg["lr"] * 0.01),
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
