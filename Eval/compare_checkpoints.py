#!/usr/bin/env python
"""
Compare multiple checkpoints by evaluating them and comparing metrics.

This script evaluates multiple checkpoints and generates a comparison report.

Usage:
    python compare_checkpoints.py \
        --checkpoints checkpoint1.pt checkpoint2.pt checkpoint3.pt \
        --config eval_config.yaml \
        --output_dir outputs/comparison
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Dict
import yaml
import pandas as pd
import matplotlib.pyplot as plt

# Add Train directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "Train"))

from evaluate_smolvla import load_checkpoint, create_eval_dataloader, evaluate
from lerobot.policies.factory import make_pre_post_processors
from lerobot.utils.utils import get_safe_torch_device, init_logging

# Initialize logging
init_logging()
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def evaluate_checkpoint(
    checkpoint_path: Path,
    config: Dict,
    device,
    eval_dataloader,
) -> Dict:
    """Evaluate a single checkpoint."""
    logger.info(f"\n{'='*80}")
    logger.info(f"Evaluating checkpoint: {checkpoint_path.name}")
    logger.info(f"{'='*80}")

    # Load checkpoint
    policy, checkpoint = load_checkpoint(str(checkpoint_path), device)

    # Create preprocessor
    pretrained_model_id = checkpoint.get("config", {}).get("policy", {}).get(
        "pretrained_model_id", "lerobot/smolvla_base"
    )
    preprocessor, _ = make_pre_post_processors(
        policy.config,
        pretrained_model_id,
        preprocessor_overrides={"device_processor": {"device": str(device)}}
    )

    # Evaluate
    metrics = evaluate(
        policy=policy,
        dataloader=eval_dataloader,
        preprocessor=preprocessor,
        device=device,
        save_predictions=False,
    )

    # Add checkpoint info to metrics
    metrics["checkpoint_name"] = checkpoint_path.name
    metrics["checkpoint_step"] = checkpoint.get("step", "unknown")

    return metrics


def create_comparison_table(all_metrics: List[Dict]) -> pd.DataFrame:
    """Create a pandas DataFrame comparing all checkpoints."""
    data = []

    for metrics in all_metrics:
        row = {
            "Checkpoint": metrics.get("checkpoint_name", "unknown"),
            "Step": metrics.get("checkpoint_step", "unknown"),
            "Loss": metrics.get("loss_mean", 0.0),
            "Action MSE": metrics.get("action_mse_mean", 0.0),
            "Position MSE": metrics.get("position_mse_mean", 0.0),
            "Rotation MSE": metrics.get("rotation_mse_mean", 0.0),
            "Gripper Acc": metrics.get("gripper_accuracy_mean", 0.0),
        }
        data.append(row)

    df = pd.DataFrame(data)
    return df


def plot_comparison(all_metrics: List[Dict], output_dir: Path):
    """Create comparison plots."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Extract data
    steps = [m.get("checkpoint_step", 0) for m in all_metrics]
    losses = [m.get("loss_mean", 0.0) for m in all_metrics]
    action_mses = [m.get("action_mse_mean", 0.0) for m in all_metrics]
    pos_mses = [m.get("position_mse_mean", 0.0) for m in all_metrics]
    rot_mses = [m.get("rotation_mse_mean", 0.0) for m in all_metrics]

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Checkpoint Comparison", fontsize=16)

    # Plot 1: Loss
    axes[0, 0].plot(steps, losses, marker='o')
    axes[0, 0].set_xlabel("Training Step")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].set_title("Loss vs Training Step")
    axes[0, 0].grid(True)

    # Plot 2: Action MSE
    axes[0, 1].plot(steps, action_mses, marker='o', color='orange')
    axes[0, 1].set_xlabel("Training Step")
    axes[0, 1].set_ylabel("Action MSE")
    axes[0, 1].set_title("Action MSE vs Training Step")
    axes[0, 1].grid(True)

    # Plot 3: Position MSE
    axes[1, 0].plot(steps, pos_mses, marker='o', color='green')
    axes[1, 0].set_xlabel("Training Step")
    axes[1, 0].set_ylabel("Position MSE")
    axes[1, 0].set_title("Position MSE vs Training Step")
    axes[1, 0].grid(True)

    # Plot 4: Rotation MSE
    axes[1, 1].plot(steps, rot_mses, marker='o', color='red')
    axes[1, 1].set_xlabel("Training Step")
    axes[1, 1].set_ylabel("Rotation MSE")
    axes[1, 1].set_title("Rotation MSE vs Training Step")
    axes[1, 1].grid(True)

    plt.tight_layout()

    # Save plot
    plot_path = output_dir / "checkpoint_comparison.png"
    plt.savefig(plot_path, dpi=150)
    logger.info(f"Comparison plot saved to: {plot_path}")

    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Compare multiple checkpoints")

    parser.add_argument(
        "--checkpoints",
        type=str,
        nargs='+',
        required=True,
        help="Paths to checkpoint files (.pt)"
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
        default="outputs/comparison",
        help="Output directory for comparison results"
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (cpu/cuda)"
    )

    args = parser.parse_args()

    # Setup
    checkpoint_paths = [Path(cp) for cp in args.checkpoints]

    # Check all checkpoints exist
    for cp_path in checkpoint_paths:
        if not cp_path.exists():
            logger.error(f"Checkpoint not found: {cp_path}")
            sys.exit(1)

    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"Config not found: {config_path}")
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Comparing {len(checkpoint_paths)} checkpoints")
    logger.info(f"Output directory: {output_dir}")

    # Load config
    config = load_config(str(config_path))

    # Device
    device = get_safe_torch_device(args.device)
    logger.info(f"Using device: {device}")

    # Create evaluation dataloader (shared across all checkpoints)
    logger.info("Creating evaluation dataloader...")
    eval_dataloader = create_eval_dataloader(config)

    # Evaluate all checkpoints
    all_metrics = []

    for cp_path in checkpoint_paths:
        try:
            metrics = evaluate_checkpoint(
                checkpoint_path=cp_path,
                config=config,
                device=device,
                eval_dataloader=eval_dataloader,
            )
            all_metrics.append(metrics)
        except Exception as e:
            logger.error(f"Failed to evaluate {cp_path}: {e}")
            continue

    # Create comparison table
    logger.info("\nCreating comparison table...")
    comparison_df = create_comparison_table(all_metrics)

    # Print table
    print("\n" + "="*80)
    print("CHECKPOINT COMPARISON")
    print("="*80)
    print(comparison_df.to_string(index=False))
    print("="*80 + "\n")

    # Save table
    table_path = output_dir / "comparison_table.csv"
    comparison_df.to_csv(table_path, index=False)
    logger.info(f"Comparison table saved to: {table_path}")

    # Create plots
    logger.info("Creating comparison plots...")
    try:
        plot_comparison(all_metrics, output_dir)
    except Exception as e:
        logger.warning(f"Failed to create plots: {e}")

    # Save detailed metrics
    detailed_path = output_dir / "detailed_metrics.yaml"
    with open(detailed_path, 'w') as f:
        yaml.dump(all_metrics, f, default_flow_style=False)
    logger.info(f"Detailed metrics saved to: {detailed_path}")

    logger.info("\nComparison complete!")


if __name__ == "__main__":
    main()
