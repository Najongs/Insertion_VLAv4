#!/usr/bin/env python
"""
Upload trained SmolVLA checkpoint to Hugging Face Hub

This script:
1. Loads a trained checkpoint
2. Converts it to Hugging Face format
3. Creates model card and metadata
4. Uploads to Hugging Face Hub

Usage:
    python upload_to_huggingface.py \
        --checkpoint /path/to/checkpoint.pt \
        --repo_id "username/model-name" \
        --private

Requirements:
    - huggingface_hub
    - Hugging Face account with write token
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Optional

import torch
import yaml
from huggingface_hub import HfApi, create_repo, upload_folder

# Add Train directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "Train"))

from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.utils.utils import init_logging

# Initialize logging
init_logging()
logger = logging.getLogger(__name__)


def load_checkpoint_info(checkpoint_path: str) -> Dict:
    """Load checkpoint and extract training information.

    Args:
        checkpoint_path: Path to checkpoint file

    Returns:
        Dictionary containing checkpoint information
    """
    logger.info(f"Loading checkpoint: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    info = {
        "step": checkpoint.get("step", "unknown"),
        "epoch": checkpoint.get("epoch", "unknown"),
        "config": checkpoint.get("config", {}),
        "policy_state_dict": checkpoint.get("policy_state_dict"),
    }

    # Get optimizer state size (optional, for info only)
    if "optimizer_state_dict" in checkpoint:
        info["has_optimizer"] = True

    logger.info(f"Checkpoint info:")
    logger.info(f"  Step: {info['step']}")
    logger.info(f"  Epoch: {info['epoch']}")

    return info


def create_model_card(
    checkpoint_info: Dict,
    repo_id: str,
    checkpoint_path: str,
    output_path: Path,
) -> None:
    """Create README.md model card for Hugging Face.

    Args:
        checkpoint_info: Checkpoint information dictionary
        repo_id: Hugging Face repository ID
        checkpoint_path: Path to checkpoint file
        output_path: Path to save README.md
    """
    config = checkpoint_info.get("config", {})
    policy_cfg = config.get("policy", {})
    dataset_cfg = config.get("dataset", {})

    # Count training episodes
    num_episodes = len(dataset_cfg.get("episode_dirs", []))

    # Extract key training parameters
    n_obs_steps = policy_cfg.get("n_obs_steps", 1)
    chunk_size = policy_cfg.get("chunk_size", 1)
    n_action_steps = policy_cfg.get("n_action_steps", 1)
    pretrained_model_id = policy_cfg.get("pretrained_model_id", "lerobot/smolvla_base")

    # Create model card content
    model_card = f"""---
license: apache-2.0
tags:
- robotics
- vision-language-action
- smolvla
- lerobot
- insertion-task
library_name: lerobot
pipeline_tag: robotics
---

# SmolVLA for VLA Insertion Task

This model is a fine-tuned version of [{pretrained_model_id}](https://huggingface.co/{pretrained_model_id})
on a robot insertion task dataset with multi-camera views and sensor feedback.

## Model Description

**SmolVLA (Small Vision-Language-Action)** is a compact vision-language-action model designed for robot manipulation tasks.
This checkpoint has been trained on a custom VLA insertion dataset with 5 different colored insertion points.

- **Base Model:** {pretrained_model_id}
- **Training Framework:** LeRobot
- **Task:** Precision needle insertion with visual and force feedback
- **Robot:** Meca500 (6-DOF collaborative robot)

## Training Details

### Training Data

- **Dataset:** VLA Insertion Task Dataset
- **Episodes:** {num_episodes} demonstrations
- **Colors:** 5 insertion targets (Blue, Green, Red, White, Yellow)
- **Observations:**
  - 5 camera views (640x480 RGB images)
  - Robot state (6-DOF pose)
  - OCT sensor data (1025 A-lines)
  - Force feedback
- **Actions:** 6-DOF delta pose commands (position + rotation)

### Training Configuration

```yaml
Training Steps: {checkpoint_info.get('step', 'unknown')}
Epochs: {checkpoint_info.get('epoch', 'unknown')}
Observation Steps: {n_obs_steps}
Action Chunk Size: {chunk_size}
Action Prediction Steps: {n_action_steps}
```

### Training Procedure

The model was trained using:
- **Optimizer:** AdamW
- **Learning Rate:** from training config
- **Hardware:** Multi-GPU training with DataParallel
- **Framework:** LeRobot + PyTorch

## Intended Use

This model is intended for:
- Research in vision-language-action models
- Robot manipulation with visual feedback
- Precision insertion tasks
- Multi-modal robot learning

### Example Usage

```python
import torch
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

# Load the trained model
model_id = "{repo_id}"
policy = SmolVLAPolicy.from_pretrained(model_id)
policy.eval()

# Prepare observation
observation = {{
    "observation.images.camera1": image_tensor,  # Shape: (1, 3, 480, 640)
    "observation.state": state_tensor,           # Shape: (1, 6)
    "task": "Insert needle into Red point",
    "robot_type": "meca500",
}}

# Get action prediction
with torch.no_grad():
    action = policy.select_action(observation)

print(f"Predicted action: {{action}}")  # Shape: (1, 6) - delta pose
```

## Model Architecture

SmolVLA combines:
1. **Vision Encoder:** Processes multi-camera RGB images
2. **Language Encoder:** Processes task instructions
3. **Action Decoder:** Predicts robot actions from visual and language inputs

Key features:
- Multi-camera fusion
- Vision-language alignment
- Action chunking for temporal consistency
- Efficient architecture for real-time inference

## Limitations and Biases

- Trained specifically for insertion tasks with the Meca500 robot
- Performance may vary with different lighting conditions
- Requires similar camera setup (5 views) for best results
- Limited to the insertion target colors seen during training

## Training Infrastructure

- **Framework:** LeRobot + PyTorch
- **Hardware:** Multi-GPU setup
- **Checkpoint:** {Path(checkpoint_path).name}

## Citation

If you use this model, please cite:

```bibtex
@misc{{smolvla-insertion-task,
  title={{SmolVLA for VLA Insertion Task}},
  author={{Your Name}},
  year={{2024}},
  publisher={{Hugging Face}},
  howpublished={{\\url{{https://huggingface.co/{repo_id}}}}},
}}
```

## Model Card Authors

Created by the VLA Insertion Task team.

## Model Card Contact

For questions or issues, please open an issue in the repository.
"""

    # Save model card
    with open(output_path, 'w') as f:
        f.write(model_card)

    logger.info(f"Model card created: {output_path}")


def create_config_json(checkpoint_info: Dict, output_path: Path) -> None:
    """Create config.json for Hugging Face.

    Args:
        checkpoint_info: Checkpoint information dictionary
        output_path: Path to save config.json
    """
    config = checkpoint_info.get("config", {})
    policy_cfg = config.get("policy", {})

    # Create Hugging Face compatible config
    hf_config = {
        "model_type": "smolvla",
        "architectures": ["SmolVLAPolicy"],
        "pretrained_model_id": policy_cfg.get("pretrained_model_id", "lerobot/smolvla_base"),
        "n_obs_steps": policy_cfg.get("n_obs_steps", 1),
        "chunk_size": policy_cfg.get("chunk_size", 1),
        "n_action_steps": policy_cfg.get("n_action_steps", 1),
        "training_step": checkpoint_info.get("step", 0),
        "training_epoch": checkpoint_info.get("epoch", 0),
        "task": "vla_insertion",
        "robot_type": "meca500",
    }

    with open(output_path, 'w') as f:
        json.dump(hf_config, f, indent=2)

    logger.info(f"Config file created: {output_path}")


def prepare_model_for_upload(
    checkpoint_path: str,
    output_dir: Path,
    repo_id: str,
) -> None:
    """Prepare model files for Hugging Face upload.

    Args:
        checkpoint_path: Path to checkpoint file
        output_dir: Directory to save prepared files
        repo_id: Hugging Face repository ID
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load checkpoint info
    checkpoint_info = load_checkpoint_info(checkpoint_path)

    # Load and save policy in Hugging Face format
    logger.info("Loading policy model...")
    policy_cfg = checkpoint_info["config"].get("policy", {})
    pretrained_model_id = policy_cfg.get("pretrained_model_id", "lerobot/smolvla_base")

    # Load base policy
    policy = SmolVLAPolicy.from_pretrained(pretrained_model_id)

    # Update policy config
    policy.config.n_obs_steps = policy_cfg.get("n_obs_steps", 1)
    policy.config.chunk_size = policy_cfg.get("chunk_size", 1)
    policy.config.n_action_steps = policy_cfg.get("n_action_steps", 1)

    # Load trained weights
    policy_state_dict = checkpoint_info["policy_state_dict"]

    # Handle DataParallel wrapper
    if any(k.startswith("module.") for k in policy_state_dict.keys()):
        new_state_dict = {}
        for k, v in policy_state_dict.items():
            if k.startswith("module.model."):
                new_key = k.replace("module.model.", "")
                new_state_dict[new_key] = v
            elif k.startswith("module."):
                new_key = k.replace("module.", "")
                new_state_dict[new_key] = v
            else:
                new_state_dict[k] = v
        policy_state_dict = new_state_dict

    # Load weights
    policy.load_state_dict(policy_state_dict, strict=False)

    # Save policy using Hugging Face method
    logger.info(f"Saving policy to {output_dir}...")
    policy.save_pretrained(output_dir)

    # Create model card
    readme_path = output_dir / "README.md"
    create_model_card(checkpoint_info, repo_id, checkpoint_path, readme_path)

    # Create config.json
    config_path = output_dir / "config.json"
    create_config_json(checkpoint_info, config_path)

    # Save training config as YAML for reference
    training_config_path = output_dir / "training_config.yaml"
    with open(training_config_path, 'w') as f:
        yaml.dump(checkpoint_info["config"], f, default_flow_style=False)

    logger.info(f"Training config saved: {training_config_path}")

    logger.info("Model preparation complete!")


def upload_to_hub(
    local_dir: Path,
    repo_id: str,
    private: bool = False,
    token: Optional[str] = None,
) -> str:
    """Upload model to Hugging Face Hub.

    Args:
        local_dir: Local directory with model files
        repo_id: Hugging Face repository ID (e.g., "username/model-name")
        private: Whether to create a private repository
        token: Hugging Face API token (optional, uses HF_TOKEN env var if not provided)

    Returns:
        URL of the uploaded model
    """
    logger.info(f"Uploading to Hugging Face Hub: {repo_id}")

    # Create repository if it doesn't exist
    try:
        api = HfApi(token=token)
        repo_url = create_repo(
            repo_id=repo_id,
            private=private,
            token=token,
            exist_ok=True,
        )
        logger.info(f"Repository created/accessed: {repo_url}")
    except Exception as e:
        logger.error(f"Failed to create repository: {e}")
        raise

    # Upload folder
    try:
        logger.info(f"Uploading files from {local_dir}...")
        api.upload_folder(
            folder_path=str(local_dir),
            repo_id=repo_id,
            repo_type="model",
            token=token,
        )
        logger.info("Upload complete!")

        model_url = f"https://huggingface.co/{repo_id}"
        logger.info(f"Model available at: {model_url}")

        return model_url

    except Exception as e:
        logger.error(f"Failed to upload: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(
        description="Upload trained SmolVLA checkpoint to Hugging Face Hub"
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to checkpoint file (.pt)"
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
        default="outputs/hf_upload",
        help="Directory to prepare model files before upload"
    )

    parser.add_argument(
        "--private",
        action="store_true",
        help="Create a private repository"
    )

    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Hugging Face API token (optional, uses HF_TOKEN env var if not provided)"
    )

    parser.add_argument(
        "--no_upload",
        action="store_true",
        help="Only prepare files, do not upload to Hub"
    )

    args = parser.parse_args()

    # Validate checkpoint path
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        sys.exit(1)

    output_dir = Path(args.output_dir)

    logger.info("="*80)
    logger.info("SmolVLA to Hugging Face Hub Uploader")
    logger.info("="*80)
    logger.info(f"Checkpoint: {checkpoint_path}")
    logger.info(f"Repository: {args.repo_id}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Private: {args.private}")
    logger.info("="*80)

    # Prepare model files
    logger.info("\n[1/2] Preparing model files...")
    try:
        prepare_model_for_upload(
            checkpoint_path=str(checkpoint_path),
            output_dir=output_dir,
            repo_id=args.repo_id,
        )
    except Exception as e:
        logger.error(f"Failed to prepare model: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Upload to Hub
    if not args.no_upload:
        logger.info("\n[2/2] Uploading to Hugging Face Hub...")
        try:
            model_url = upload_to_hub(
                local_dir=output_dir,
                repo_id=args.repo_id,
                private=args.private,
                token=args.token,
            )

            logger.info("\n" + "="*80)
            logger.info("SUCCESS!")
            logger.info("="*80)
            logger.info(f"Model uploaded to: {model_url}")
            logger.info(f"View your model at: {model_url}")
            logger.info("\nTo use this model:")
            logger.info(f'  from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy')
            logger.info(f'  policy = SmolVLAPolicy.from_pretrained("{args.repo_id}")')
            logger.info("="*80)

        except Exception as e:
            logger.error(f"Failed to upload: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    else:
        logger.info("\n" + "="*80)
        logger.info("Files prepared (upload skipped)")
        logger.info("="*80)
        logger.info(f"Model files saved to: {output_dir}")
        logger.info("\nTo upload manually, run:")
        logger.info(f"  python {__file__} --checkpoint {checkpoint_path} --repo_id {args.repo_id}")
        logger.info("="*80)


if __name__ == "__main__":
    main()
