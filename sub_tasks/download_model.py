#!/usr/bin/env python
"""
Download trained SmolVLA model from Hugging Face Hub

This script downloads a model from Hugging Face Hub and optionally
converts it to a checkpoint format compatible with training scripts.

Usage:
    python download_model.py \
        --repo_id "username/model-name" \
        --output_dir "downloads/model" \
        --save_checkpoint

Requirements:
    - huggingface_hub
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional

import torch
import yaml

# Add Train directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "Train"))

from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.utils.utils import init_logging

# Initialize logging
init_logging()
logger = logging.getLogger(__name__)


def download_model(
    repo_id: str,
    output_dir: Path,
    token: Optional[str] = None,
    revision: Optional[str] = None,
) -> SmolVLAPolicy:
    """Download model from Hugging Face Hub.

    Args:
        repo_id: Hugging Face repository ID
        output_dir: Directory to save model
        token: Hugging Face API token (optional)
        revision: Git revision (branch/tag/commit) to download

    Returns:
        Loaded policy model
    """
    logger.info(f"Downloading model from: {repo_id}")

    if revision:
        logger.info(f"Using revision: {revision}")

    try:
        # Download and load model
        policy = SmolVLAPolicy.from_pretrained(
            repo_id,
            token=token,
            revision=revision,
        )

        logger.info("Model downloaded successfully!")

        # Save to local directory
        output_dir.mkdir(parents=True, exist_ok=True)
        policy.save_pretrained(output_dir)

        logger.info(f"Model saved to: {output_dir}")

        return policy

    except Exception as e:
        logger.error(f"Failed to download model: {e}")
        raise


def convert_to_checkpoint(
    policy: SmolVLAPolicy,
    output_path: Path,
    repo_id: str,
) -> None:
    """Convert downloaded model to checkpoint format.

    Args:
        policy: Loaded policy model
        output_path: Path to save checkpoint
        repo_id: Original repository ID
    """
    logger.info(f"Converting model to checkpoint format...")

    # Create checkpoint dictionary
    checkpoint = {
        "policy_state_dict": policy.state_dict(),
        "config": {
            "policy": {
                "pretrained_model_id": repo_id,
                "n_obs_steps": getattr(policy.config, "n_obs_steps", 1),
                "chunk_size": getattr(policy.config, "chunk_size", 1),
                "n_action_steps": getattr(policy.config, "n_action_steps", 1),
            }
        },
        "step": getattr(policy.config, "training_step", 0),
        "epoch": getattr(policy.config, "training_epoch", 0),
    }

    # Save checkpoint
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output_path)

    logger.info(f"Checkpoint saved to: {output_path}")


def print_model_info(policy: SmolVLAPolicy, repo_id: str) -> None:
    """Print model information.

    Args:
        policy: Loaded policy model
        repo_id: Repository ID
    """
    print("\n" + "="*80)
    print("MODEL INFORMATION")
    print("="*80)
    print(f"Repository: {repo_id}")
    print(f"Model URL: https://huggingface.co/{repo_id}")
    print()

    # Config information
    print("Configuration:")
    if hasattr(policy.config, "n_obs_steps"):
        print(f"  Observation Steps: {policy.config.n_obs_steps}")
    if hasattr(policy.config, "chunk_size"):
        print(f"  Chunk Size: {policy.config.chunk_size}")
    if hasattr(policy.config, "n_action_steps"):
        print(f"  Action Steps: {policy.config.n_action_steps}")
    if hasattr(policy.config, "training_step"):
        print(f"  Training Step: {policy.config.training_step}")
    if hasattr(policy.config, "training_epoch"):
        print(f"  Training Epoch: {policy.config.training_epoch}")

    # Model size
    total_params = sum(p.numel() for p in policy.parameters())
    trainable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    print()
    print(f"Model Size:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")

    print("="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Download trained SmolVLA model from Hugging Face Hub"
    )

    parser.add_argument(
        "--repo_id",
        type=str,
        required=True,
        help='Hugging Face repository ID (e.g., "username/smolvla-insertion")'
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="downloads/model",
        help="Directory to save downloaded model"
    )

    parser.add_argument(
        "--save_checkpoint",
        action="store_true",
        help="Also save as checkpoint format (.pt)"
    )

    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="Path to save checkpoint file (default: output_dir/checkpoint.pt)"
    )

    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Hugging Face API token (optional, for private models)"
    )

    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        help="Git revision to download (branch/tag/commit)"
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    logger.info("="*80)
    logger.info("SmolVLA Model Downloader")
    logger.info("="*80)
    logger.info(f"Repository: {args.repo_id}")
    logger.info(f"Output directory: {output_dir}")
    if args.revision:
        logger.info(f"Revision: {args.revision}")
    logger.info("="*80)

    # Download model
    try:
        policy = download_model(
            repo_id=args.repo_id,
            output_dir=output_dir,
            token=args.token,
            revision=args.revision,
        )
    except Exception as e:
        logger.error(f"Download failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Print model information
    print_model_info(policy, args.repo_id)

    # Convert to checkpoint if requested
    if args.save_checkpoint:
        if args.checkpoint_path:
            checkpoint_path = Path(args.checkpoint_path)
        else:
            checkpoint_path = output_dir / "checkpoint.pt"

        try:
            convert_to_checkpoint(
                policy=policy,
                output_path=checkpoint_path,
                repo_id=args.repo_id,
            )
        except Exception as e:
            logger.error(f"Checkpoint conversion failed: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

    # Success message
    logger.info("\n" + "="*80)
    logger.info("SUCCESS!")
    logger.info("="*80)
    logger.info(f"Model downloaded to: {output_dir}")
    if args.save_checkpoint:
        logger.info(f"Checkpoint saved to: {checkpoint_path}")
    logger.info("\nTo use this model:")
    logger.info(f'  from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy')
    logger.info(f'  policy = SmolVLAPolicy.from_pretrained("{output_dir}")')
    logger.info("="*80)


if __name__ == "__main__":
    main()
