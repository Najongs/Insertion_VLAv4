#!/usr/bin/env python
"""
LoRA Training script for π0 (Pi Zero) Policy on New HDF5 VLA Dataset

This script trains π0 using LoRA (Low-Rank Adaptation) for memory efficiency.
π0 uses flow matching and vision-language conditioning.

Memory Requirements:
- Full Fine-Tuning: > 70 GB
- LoRA Fine-Tuning: > 22.5 GB (fits on RTX 3090/4090)

Usage:
    python train_pi0_lora.py --config train_config_pi0_lora.yaml

DDP Usage:
    torchrun --nproc_per_node=5 train_pi0_lora.py --config train_config_pi0_lora.yaml
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

# PEFT imports for LoRA
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training

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


def apply_lora_to_policy(policy: PI0Policy, lora_config_dict: Dict) -> PI0Policy:
    """
    Apply LoRA adapters to π0 policy.

    LoRA targets:
    - PaliGemma vision encoder: query, value projections
    - Action expert (Gemma): query, value, output projections

    This reduces trainable parameters from ~2.5B to ~10-50M.
    """
    # Extract LoRA hyperparameters
    lora_r = lora_config_dict.get("lora_r", 8)
    lora_alpha = lora_config_dict.get("lora_alpha", 16)
    lora_dropout = lora_config_dict.get("lora_dropout", 0.05)
    target_modules = lora_config_dict.get("target_modules", [
        "q_proj", "v_proj", "k_proj", "o_proj",  # Attention layers
        "gate_proj", "up_proj", "down_proj"      # MLP layers
    ])

    if is_main_process():
        logger.info("=" * 80)
        logger.info("Applying LoRA to π0 Policy")
        logger.info("=" * 80)
        logger.info(f"LoRA rank (r): {lora_r}")
        logger.info(f"LoRA alpha: {lora_alpha}")
        logger.info(f"LoRA dropout: {lora_dropout}")
        logger.info(f"Target modules: {target_modules}")

    # Configure LoRA
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,  # π0 uses causal language modeling structure
        r=lora_r,                       # LoRA rank
        lora_alpha=lora_alpha,          # LoRA scaling factor
        lora_dropout=lora_dropout,      # Dropout for LoRA layers
        target_modules=target_modules,  # Modules to apply LoRA
        bias="none",                    # Don't train bias terms
        inference_mode=False,           # Training mode
    )

    # Apply LoRA to policy
    policy = get_peft_model(policy, peft_config)

    if is_main_process():
        # Print trainable parameters
        policy.print_trainable_parameters()
        logger.info("=" * 80)

    return policy


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
        # Auto-discover all .h5/.hdf5 files in root_dir
        hdf5_files = sorted(list(root_dir.glob("*.h5")) + list(root_dir.glob("*.hdf5")))

    if len(hdf5_files) == 0:
        raise FileNotFoundError(f"No HDF5 files found in {root_dir}")

    if is_main_process():
        logger.info(f"Found {len(hdf5_files)} HDF5 files:")
        for f in hdf5_files:
            logger.info(f"  - {f.name}")

    # Create tokenizer for language conditioning (π0 requirement)
    from transformers import AutoTokenizer
    tokenizer_max_length = policy_cfg.get("tokenizer_max_length", 48)
    tokenizer = AutoTokenizer.from_pretrained("google/paligemma-3b-pt-224")
    if is_main_process():
        logger.info(f"Created PaliGemma tokenizer (max_length={tokenizer_max_length})")

    # Create dataset
    dataset = create_hdf5_lerobot_dataset(
        hdf5_paths=hdf5_files,
        horizon=dataset_cfg.get("horizon", 50),
        n_obs_steps=policy_cfg.get("n_obs_steps", 1),  # π0 uses single observation
        squeeze_n_obs_steps=(policy_cfg.get("n_obs_steps", 1) == 1),  # Squeeze if n_obs_steps=1
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
        # Log sample to verify data format
        sample = dataset[0]
        logger.info("Sample data shapes:")
        for key, value in sample.items():
            if isinstance(value, torch.Tensor):
                logger.info(f"  {key}: {value.shape}")

    return dataset


def create_policy_from_config(config: Dict, device: torch.device) -> PI0Policy:
    """Create π0 policy from configuration."""
    policy_cfg = config["policy"]

    if is_main_process():
        logger.info("=" * 80)
        logger.info("Creating π0 Policy")
        logger.info("=" * 80)
        logger.info(f"paligemma_variant: {policy_cfg.get('paligemma_variant', 'gemma_2b')}")
        logger.info(f"action_expert_variant: {policy_cfg.get('action_expert_variant', 'gemma_300m')}")
        logger.info(f"dtype: {policy_cfg.get('dtype', 'float32')}")
        logger.info(f"n_obs_steps: {policy_cfg.get('n_obs_steps', 1)}")
        logger.info(f"chunk_size: {policy_cfg.get('chunk_size', 50)}")
        logger.info(f"gradient_checkpointing: {policy_cfg.get('gradient_checkpointing', False)}")

    # Create PI0Config
    pi0_config = PI0Config(
        paligemma_variant=policy_cfg.get("paligemma_variant", "gemma_2b"),
        action_expert_variant=policy_cfg.get("action_expert_variant", "gemma_300m"),
        dtype=policy_cfg.get("dtype", "float32"),
        n_obs_steps=policy_cfg.get("n_obs_steps", 1),
        chunk_size=policy_cfg.get("chunk_size", 50),
        n_action_steps=policy_cfg.get("n_action_steps", 50),
        max_state_dim=policy_cfg.get("max_state_dim", 32),
        max_action_dim=policy_cfg.get("max_action_dim", 32),
        num_inference_steps=policy_cfg.get("num_inference_steps", 10),
        time_sampling_beta_alpha=policy_cfg.get("time_sampling_beta_alpha", 1.5),
        time_sampling_beta_beta=policy_cfg.get("time_sampling_beta_beta", 1.0),
        time_sampling_scale=policy_cfg.get("time_sampling_scale", 0.999),
        time_sampling_offset=policy_cfg.get("time_sampling_offset", 0.001),
        min_period=policy_cfg.get("min_period", 0.004),
        max_period=policy_cfg.get("max_period", 4.0),
        image_resolution=tuple(policy_cfg.get("image_resolution", [224, 224])),
        gradient_checkpointing=policy_cfg.get("gradient_checkpointing", False),
        compile_model=policy_cfg.get("compile_model", False),
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

    # Create policy
    policy = PI0Policy(pi0_config)

    # Move to device
    policy.to(device)

    if is_main_process():
        logger.info(f"Policy created and moved to {device}")
        logger.info("=" * 80)

    return policy


def create_optimizer_from_config(policy: nn.Module, config: Dict) -> AdamW:
    """Create AdamW optimizer."""
    optimizer_cfg = config["optimizer"]

    # Only optimize LoRA parameters (trainable parameters)
    trainable_params = [p for p in policy.parameters() if p.requires_grad]

    if is_main_process():
        total_params = sum(p.numel() for p in policy.parameters())
        trainable_params_count = sum(p.numel() for p in trainable_params)
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters (LoRA): {trainable_params_count:,}")
        logger.info(f"Trainable %: {100 * trainable_params_count / total_params:.2f}%")

    optimizer = AdamW(
        trainable_params,
        lr=optimizer_cfg.get("lr", 2.5e-5),
        betas=tuple(optimizer_cfg.get("betas", [0.9, 0.95])),
        eps=optimizer_cfg.get("eps", 1e-8),
        weight_decay=optimizer_cfg.get("weight_decay", 0.01),
    )

    return optimizer


def create_scheduler_from_config(optimizer: AdamW, config: Dict) -> LambdaLR:
    """Create learning rate scheduler with warmup and cosine decay."""
    scheduler_cfg = config["scheduler"]

    num_warmup_steps = scheduler_cfg.get("num_warmup_steps", 1000)
    num_decay_steps = scheduler_cfg.get("num_decay_steps", 30000)
    peak_lr = scheduler_cfg.get("peak_lr", 2.5e-5)
    decay_lr = scheduler_cfg.get("decay_lr", 2.5e-6)

    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            # Linear warmup
            return float(current_step) / float(max(1, num_warmup_steps))
        else:
            # Cosine decay
            progress = float(current_step - num_warmup_steps) / float(max(1, num_decay_steps - num_warmup_steps))
            progress = min(progress, 1.0)
            cosine_decay = 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.14159)))
            return (decay_lr / peak_lr) + (1.0 - decay_lr / peak_lr) * cosine_decay

    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
    return scheduler


def train(
    policy: nn.Module,
    dataloader: DataLoader,
    optimizer: AdamW,
    scheduler: LambdaLR,
    config: Dict,
    device: torch.device,
    rank: int,
    world_size: int,
):
    """Main training loop."""
    training_cfg = config["training"]
    optimizer_cfg = config["optimizer"]

    total_steps = training_cfg.get("steps", 30000)
    log_freq = training_cfg.get("log_freq", 100)
    save_freq_epochs = training_cfg.get("save_freq_epochs", 1.0)
    grad_clip_norm = optimizer_cfg.get("grad_clip_norm", 1.0)

    output_dir = Path(config["output_dir"])
    checkpoints_dir = output_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    policy.train()
    global_step = 0
    epoch = 0
    running_loss = 0.0
    running_losses_dict = {}

    if is_main_process():
        logger.info("=" * 80)
        logger.info("Starting LoRA Training")
        logger.info("=" * 80)
        logger.info(f"Total steps: {total_steps}")
        logger.info(f"Batch size per GPU: {training_cfg.get('batch_size', 8)}")
        logger.info(f"Effective batch size: {training_cfg.get('batch_size', 8) * world_size}")
        logger.info(f"Number of GPUs: {world_size}")
        logger.info(f"Gradient clip norm: {grad_clip_norm}")
        logger.info("=" * 80)

    start_time = time.time()

    while global_step < total_steps:
        epoch += 1
        if is_main_process():
            logger.info(f"\nEpoch {epoch}")

        for batch_idx, batch in enumerate(dataloader):
            # Move batch to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            # Forward pass
            output_dict = policy.forward(batch)
            loss = output_dict["loss"]

            # Backward pass
            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            if grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(policy.parameters(), grad_clip_norm)

            # Optimizer step
            optimizer.step()
            scheduler.step()

            # Accumulate loss
            running_loss += loss.item()
            for key, value in output_dict.items():
                if key != "loss" and isinstance(value, torch.Tensor) and value.numel() == 1:
                    if key not in running_losses_dict:
                        running_losses_dict[key] = 0.0
                    running_losses_dict[key] += value.item()

            global_step += 1

            # Logging
            if global_step % log_freq == 0 and is_main_process():
                avg_loss = running_loss / log_freq
                elapsed_time = time.time() - start_time
                steps_per_sec = log_freq / elapsed_time
                lr = scheduler.get_last_lr()[0]

                log_msg = (
                    f"Step {global_step}/{total_steps} | "
                    f"Loss: {avg_loss:.4f} | "
                    f"LR: {lr:.2e} | "
                    f"Steps/sec: {steps_per_sec:.2f}"
                )

                # Add other losses
                for key, value in running_losses_dict.items():
                    avg_value = value / log_freq
                    log_msg += f" | {key}: {avg_value:.4f}"

                logger.info(log_msg)

                # Reset
                running_loss = 0.0
                running_losses_dict = {}
                start_time = time.time()

            # Save checkpoint
            if global_step % int(len(dataloader) * save_freq_epochs) == 0 and is_main_process():
                checkpoint_path = checkpoints_dir / f"checkpoint_step_{global_step}.pt"

                # Save LoRA adapters only (much smaller than full model)
                policy.save_pretrained(checkpoints_dir / f"lora_step_{global_step}")
                logger.info(f"Saved LoRA checkpoint to {checkpoints_dir / f'lora_step_{global_step}'}")

            # Check if training is complete
            if global_step >= total_steps:
                break

    # Save final checkpoint
    if is_main_process():
        final_checkpoint_path = checkpoints_dir / "lora_final"
        policy.save_pretrained(final_checkpoint_path)
        logger.info(f"Saved final LoRA checkpoint to {final_checkpoint_path}")
        logger.info("=" * 80)
        logger.info("Training completed!")
        logger.info("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Train π0 Policy with LoRA on HDF5 VLA Dataset")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML file")
    parser.add_argument("--batch_size", type=int, default=None, help="Override batch size")
    parser.add_argument("--steps", type=int, default=None, help="Override training steps")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--output_dir", type=str, default=None, help="Override output directory")
    args = parser.parse_args()

    # Setup distributed training
    rank, world_size, local_rank = setup_distributed()

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
    device = get_safe_torch_device(device_type="cuda", device_id=local_rank)

    if is_main_process():
        logger.info(f"Using device: {device}")

    # Create dataset
    dataset = create_dataset_from_config(config)

    # Create dataloader with distributed sampler
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True) if world_size > 1 else None
    dataloader = DataLoader(
        dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=(sampler is None),  # Only shuffle if not using distributed sampler
        num_workers=config["training"].get("num_workers", 4),
        pin_memory=config["training"].get("pin_memory", True),
        sampler=sampler,
        collate_fn=hdf5_lerobot_collate_fn,
    )

    # Create policy
    policy = create_policy_from_config(config, device)

    # Apply LoRA to policy
    lora_config = config.get("lora", {})
    policy = apply_lora_to_policy(policy, lora_config)

    # Wrap policy with DDP if using distributed training
    if world_size > 1:
        policy = DDP(policy, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
        if is_main_process():
            logger.info("Policy wrapped with DistributedDataParallel")

    # Create optimizer and scheduler
    optimizer = create_optimizer_from_config(policy, config)
    scheduler = create_scheduler_from_config(optimizer, config)

    # Train
    try:
        train(policy, dataloader, optimizer, scheduler, config, device, rank, world_size)
    except KeyboardInterrupt:
        if is_main_process():
            logger.info("\nTraining interrupted by user")
    except Exception as e:
        logger.error(f"Training failed with error: {e}", exc_info=True)
        raise
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
