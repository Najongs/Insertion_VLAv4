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
import gc
from pathlib import Path
from typing import Dict, List, Optional
from contextlib import nullcontext
import yaml

# Memory monitoring
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

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

# Normalization imports
from normalization_utils import Normalizer, load_stats

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
        hdf5_files = sorted(list(root_dir.rglob("*.h5")) + list(root_dir.rglob("*.hdf5")))

    if len(hdf5_files) == 0:
        raise FileNotFoundError(f"No HDF5 files found in {root_dir}")

    if is_main_process():
        logger.info(f"Loading dataset from {len(hdf5_files)} HDF5 files...")

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
        use_ee_pose_delta_as_action=dataset_cfg.get("use_ee_pose_delta_as_action", False),
        task_instruction=dataset_cfg.get("task_instruction", "Insert needle into eye trocar."),
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


def normalize_batch(batch: dict, normalizer: Optional[Normalizer], device: torch.device) -> dict:
    """Apply normalization to state and action in batch."""
    if normalizer is None:
        return batch

    normalized_batch = {}
    for key, value in batch.items():
        if key == "observation.state":
            # Normalize state
            normalized_batch[key] = normalizer.normalize(value.to(device), key)
        elif key == "action":
            # Normalize action
            normalized_batch[key] = normalizer.normalize(value.to(device), key)
        else:
            # Keep other keys as-is
            normalized_batch[key] = value.to(device) if isinstance(value, torch.Tensor) else value

    return normalized_batch


def update_policy(
    policy: nn.Module,
    batch: dict,
    optimizer: AdamW,
    grad_scaler: GradScaler,
    grad_clip_norm: float,
    lr_scheduler: LambdaLR,
    lambda_temporal: float,
    normalizer: Optional[Normalizer] = None,
    use_amp: bool = False,
    gradient_accumulation_steps: int = 1,
    accumulation_step: int = 0,
):
    """
    Performs a single training step with gradient accumulation support.

    Args:
        gradient_accumulation_steps: Number of steps to accumulate gradients
        accumulation_step: Current step in accumulation cycle (0 to gradient_accumulation_steps-1)
        normalizer: Optional normalizer for state and action
    """
    device = get_device_from_parameters(policy)
    policy.train()

    # Apply normalization to batch
    batch = normalize_batch(batch, normalizer, device)

    # Scale loss by accumulation steps to maintain gradient magnitude
    loss_scale = 1.0 / gradient_accumulation_steps

    # Use float16 explicitly for AMP (not bfloat16, as GradScaler doesn't support bfloat16)
    with torch.autocast(device_type=device.type, enabled=use_amp, dtype=torch.float16):
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

        # Scale loss for gradient accumulation
        loss = loss * loss_scale

    # Backward pass (use GradScaler for float16 stability)
    if use_amp:
        grad_scaler.scale(loss).backward()
    else:
        loss.backward()

    # Only update weights after accumulating gradients
    is_update_step = (accumulation_step + 1) % gradient_accumulation_steps == 0

    if is_update_step:
        if use_amp:
            # AMP: unscale gradients before clipping
            grad_scaler.unscale_(optimizer)

        # Clip gradients
        grad_norm = torch.nn.utils.clip_grad_norm_(
            policy.parameters(),
            grad_clip_norm,
            error_if_nonfinite=False,
        )

        # Update weights
        if use_amp:
            grad_scaler.step(optimizer)
            grad_scaler.update()
        else:
            optimizer.step()

        optimizer.zero_grad(set_to_none=True)

        if lr_scheduler is not None:
            lr_scheduler.step()
    else:
        grad_norm = 0.0

    # Return unscaled loss for logging
    return loss.item() / loss_scale, action_loss.item(), temporal_loss.item(), grad_norm if isinstance(grad_norm, float) else grad_norm.item()


def train(
    policy: nn.Module,
    dataloader: DataLoader,
    optimizer: AdamW,
    scheduler: LambdaLR,
    grad_scaler: GradScaler,
    config: Dict,
    rank: int,
    world_size: int,
    normalizer: Optional[Normalizer] = None,
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
    gradient_accumulation_steps = training_cfg.get("gradient_accumulation_steps", 1)
    use_amp = training_cfg.get("use_amp", False)

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
    # DISABLED: wandb.watch causes memory accumulation over long training runs
    # We'll log metrics manually instead
    if is_main_process() and wandb_run is not None:
        logger.info("W&B watch disabled to prevent memory accumulation")
        # try:
        #     wandb.watch(policy, log="gradients", log_freq=log_freq * 10)
        #     logger.info(f"W&B watch enabled: log='gradients', log_freq={log_freq * 10}")
        # except Exception as e:
        #     logger.warning(f"Failed to setup wandb.watch: {e}")

    if is_main_process():
        batch_size = training_cfg.get('batch_size', 8)
        effective_batch = batch_size * world_size * gradient_accumulation_steps
        logger.info("=" * 80)
        logger.info("Starting SmolVLA Expert-Only Training")
        logger.info("=" * 80)
        logger.info(f"Total steps: {total_steps}")
        logger.info(f"Batch size per GPU: {batch_size}")
        logger.info(f"Number of GPUs: {world_size}")
        logger.info(f"Gradient accumulation steps: {gradient_accumulation_steps}")
        logger.info(f"Effective batch size: {effective_batch}")
        logger.info(f"Gradient clip norm: {grad_clip_norm}")
        logger.info(f"Temporal consistency weight (λ): {lambda_temporal}")
        logger.info(f"Mixed Precision (AMP): {use_amp} (dtype=float16)")
        if wandb_run is not None:
            logger.info(f"W&B tracking: {wandb_run.url}")
        logger.info("=" * 80)

    start_time = time.time()

    # Create DataLoader iterator (regenerate periodically to prevent memory buildup)
    # Using manual epoch looping instead of itertools.cycle to avoid buffer accumulation
    def get_dataloader_iterator():
        """Create a fresh dataloader iterator to prevent memory buildup."""
        while True:
            for batch in dataloader:
                yield batch

    dl_iter = get_dataloader_iterator()

    # Progress bar (only on main process)
    if is_main_process():
        pbar = tqdm(total=total_steps, desc="Training", unit="step",
                   bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
    else:
        pbar = None

    for step in range(total_steps):
        # Calculate accumulation step (0 to gradient_accumulation_steps-1)
        accumulation_step = step % gradient_accumulation_steps

        # Get batch
        batch = next(dl_iter)

        # Move batch to device
        batch = {k: v.to(get_device_from_parameters(policy)) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}

        # Update policy with gradient accumulation
        loss, action_loss, temporal_loss, grad_norm = update_policy(
            policy,
            batch,
            optimizer,
            grad_scaler,
            grad_clip_norm,
            scheduler,
            lambda_temporal,
            normalizer=normalizer,
            use_amp=use_amp,
            gradient_accumulation_steps=gradient_accumulation_steps,
            accumulation_step=accumulation_step,
        )

        # Accumulate losses (detach from computation graph to prevent memory buildup)
        running_loss += loss
        running_action_loss += action_loss
        running_temporal_loss += temporal_loss
        if grad_norm > 0:  # Only accumulate when weights are updated
            running_grad_norm += grad_norm

        # MEMORY FIX: Explicitly delete batch and clear caches periodically
        del batch

        # Every 50 steps: aggressive memory cleanup
        if (step + 1) % 50 == 0:
            torch.cuda.empty_cache()  # Clear CUDA cache
            gc.collect()  # Force Python garbage collection to prevent RAM buildup

        # Update progress bar
        if pbar is not None:
            pbar.update(1)
            # Update postfix every step with current metrics
            if (step + 1) % 10 == 0:
                current_lr = scheduler.get_last_lr()[0]
                pbar.set_postfix({
                    'loss': f'{running_loss / min(step % log_freq + 1, log_freq):.4f}',
                    'lr': f'{current_lr:.2e}'
                })

        # Logging
        if (step + 1) % log_freq == 0 and is_main_process():
            avg_loss = running_loss / log_freq
            avg_action_loss = running_action_loss / log_freq
            avg_temporal_loss = running_temporal_loss / log_freq
            avg_grad_norm = running_grad_norm / log_freq
            elapsed_time = time.time() - start_time
            steps_per_sec = log_freq / elapsed_time
            lr = scheduler.get_last_lr()[0]

            # Memory monitoring
            if PSUTIL_AVAILABLE:
                process = psutil.Process()
                ram_gb = process.memory_info().rss / (1024 ** 3)  # Convert to GB
                ram_percent = process.memory_percent()
            else:
                ram_gb = 0.0
                ram_percent = 0.0

            # CUDA memory
            if torch.cuda.is_available():
                cuda_mem_allocated = torch.cuda.memory_allocated() / (1024 ** 3)  # GB
                cuda_mem_reserved = torch.cuda.memory_reserved() / (1024 ** 3)  # GB
            else:
                cuda_mem_allocated = 0.0
                cuda_mem_reserved = 0.0

            # Progress percentage
            progress_pct = (step + 1) / total_steps * 100

            log_msg = (
                f"[{progress_pct:5.1f}%] Step {step + 1}/{total_steps} | "
                f"Loss: {avg_loss:.4f} (Action: {avg_action_loss:.4f}, Temporal: {avg_temporal_loss:.4f}) | "
                f"GradNorm: {avg_grad_norm:.3f} | "
                f"LR: {lr:.2e} | "
                f"{steps_per_sec:.2f} steps/sec | "
                f"VRAM: {cuda_mem_allocated:.1f}GB"
            )

            pbar.write(log_msg)  # Write to tqdm output

            # W&B logging (create dict without holding references)
            if wandb_run is not None:
                log_dict = {
                    "train/loss": float(avg_loss),
                    "train/action_loss": float(avg_action_loss),
                    "train/temporal_loss": float(avg_temporal_loss),
                    "train/grad_norm": float(avg_grad_norm),
                    "train/learning_rate": float(lr),
                    "train/steps_per_sec": float(steps_per_sec),
                    "system/ram_gb": float(ram_gb),
                    "system/ram_percent": float(ram_percent),
                    "system/vram_allocated_gb": float(cuda_mem_allocated),
                    "system/vram_reserved_gb": float(cuda_mem_reserved),
                    "step": int(step + 1),
                }
                wandb.log(log_dict)
                del log_dict  # Explicitly delete to free memory

            # Reset
            running_loss = 0.0
            running_action_loss = 0.0
            running_temporal_loss = 0.0
            running_grad_norm = 0.0
            start_time = time.time()

        # Intermediate checkpoint saving
        if (step + 1) % save_freq == 0 and is_main_process():
            logger.info(f"Saving checkpoint at step {step + 1}...")

            # Get the actual policy (unwrap DDP if necessary)
            policy_to_save = policy.module if isinstance(policy, DDP) else policy

            checkpoint_path = checkpoints_dir / f"checkpoint_step_{step + 1}.pt"

            checkpoint_dict = {
                "step": step + 1,
                "epoch": 0,
                "policy_state_dict": policy_to_save.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "config": config,
            }

            # Save normalization stats if normalizer is used
            if normalizer is not None:
                checkpoint_dict["normalization_stats"] = normalizer.stats

            torch.save(checkpoint_dict, checkpoint_path)
            logger.info(f"Saved checkpoint to: {checkpoint_path}")

    # Close progress bar
    if pbar is not None:
        pbar.close()

    # Save final checkpoint as .pt (matching inference format)
    if is_main_process():
        logger.info("=" * 80)
        logger.info("Saving final model...")
        logger.info("=" * 80)

        # Get the actual policy (unwrap DDP if necessary)
        policy_to_save = policy.module if isinstance(policy, DDP) else policy

        # Compute normalization statistics from dataset
        logger.info("Computing action normalization statistics from dataset...")
        all_actions = []
        sample_count = 0
        max_samples = 10000  # Sample up to 10k actions for statistics

        for batch in dataloader:
            if "action" in batch:
                actions = batch["action"]
                # Handle different action shapes: (B, T, D) or (B, D)
                if actions.ndim == 3:
                    actions = actions.reshape(-1, actions.shape[-1])  # (B*T, D)
                elif actions.ndim == 2:
                    pass  # Already (B, D)
                all_actions.append(actions.cpu().numpy())
                sample_count += actions.shape[0]

            if sample_count >= max_samples:
                break

        # Compute statistics
        if len(all_actions) > 0:
            all_actions = np.concatenate(all_actions, axis=0)
            action_min = all_actions.min(axis=0)  # (D,)
            action_max = all_actions.max(axis=0)  # (D,)
            action_mean = all_actions.mean(axis=0)  # (D,)
            action_std = all_actions.std(axis=0)  # (D,)

            logger.info(f"Action statistics computed from {len(all_actions)} samples:")
            logger.info(f"  Min: {action_min}")
            logger.info(f"  Max: {action_max}")
            logger.info(f"  Mean: {action_mean}")
            logger.info(f"  Std: {action_std}")
        else:
            logger.warning("No actions found in dataset, using default normalization")
            action_dim = 6
            action_min = np.array([-1.0] * action_dim)
            action_max = np.array([1.0] * action_dim)
            action_mean = np.array([0.0] * action_dim)
            action_std = np.array([1.0] * action_dim)

        # Save as .pt checkpoint (matching lerobot_to_MECA.py format)
        final_checkpoint_path = checkpoints_dir / "checkpoint_latest.pt"

        checkpoint_dict = {
            "step": total_steps,
            "epoch": 0,  # Not used in this training script
            "policy_state_dict": policy_to_save.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "config": config,
            # Normalization statistics (for inference)
            "action_min": action_min,
            "action_max": action_max,
            "action_mean": action_mean,
            "action_std": action_std,
        }

        # Save normalization stats if normalizer is used
        if normalizer is not None:
            checkpoint_dict["normalization_stats"] = normalizer.stats

        torch.save(checkpoint_dict, final_checkpoint_path)
        logger.info(f"Saved final checkpoint (.pt) to: {final_checkpoint_path}")

        # Also save HuggingFace format for compatibility
        hf_checkpoint_path = checkpoints_dir / "final_hf"
        policy_to_save.save_pretrained(hf_checkpoint_path)
        logger.info(f"Saved HuggingFace format to: {hf_checkpoint_path}")

        logger.info("=" * 80)
        logger.info("Training completed successfully!")
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
    sampler = DistributedSampler(dataset, shuffle=False) if is_distributed else None

    # Use num_workers from config (set to 0 to avoid shared memory OOM in DDP)
    # With 5 GPUs and large images (512x512x3), multi-worker DataLoader uses too much shared memory
    num_workers = config["training"].get("num_workers", 0)

    if is_main_process():
        if num_workers == 0:
            logger.info("DataLoader: num_workers=0 (loading in main process to avoid shared memory OOM)")
        else:
            logger.info(f"DataLoader: num_workers={num_workers}")

    dataloader = DataLoader(
        dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=(sampler is None),
        num_workers=num_workers,
        pin_memory=config["training"].get("pin_memory", True),  # Enable for faster GPU transfer
        sampler=sampler,
        collate_fn=hdf5_lerobot_collate_fn,
        drop_last=False,
        prefetch_factor=2 if num_workers > 0 else None,  # Prefetch 2 batches per worker
        persistent_workers=(num_workers > 0),  # Keep workers alive for better performance
    )

    # Create normalizer if normalization is enabled
    normalizer = None
    if config.get("normalization", {}).get("enable", False):
        stats_file = config["normalization"].get("stats_file", "dataset_stats.yaml")
        stats_path = Path(__file__).parent / stats_file

        if stats_path.exists():
            if is_main_process():
                logger.info("=" * 80)
                logger.info("Loading Normalization Statistics")
                logger.info("=" * 80)
                logger.info(f"Stats file: {stats_path}")

            stats = load_stats(str(stats_path))
            normalizer = Normalizer(stats).to(device)

            if is_main_process():
                logger.info("Normalizer created successfully")
                logger.info(f"  Action mean: {stats['action']['mean'][:3]}...")
                logger.info(f"  Action std:  {stats['action']['std'][:3]}...")
                logger.info(f"  State mean:  {stats['observation.state']['mean'][:3]}...")
                logger.info(f"  State std:   {stats['observation.state']['std'][:3]}...")
                logger.info("=" * 80)
        else:
            if is_main_process():
                logger.warning(f"Normalization enabled but stats file not found: {stats_path}")
                logger.warning("Training without normalization!")

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

    # Create GradScaler for Mixed Precision Training
    use_amp = config["training"].get("use_amp", False)
    grad_scaler = GradScaler(device.type, enabled=use_amp)

    # Setup W&B (main process only)
    wandb_run = None
    if is_main_process():
        wandb_run = setup_wandb(config)

    # Train
    try:
        train(policy, dataloader, optimizer, scheduler, grad_scaler, config, rank, world_size, normalizer, wandb_run)
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
