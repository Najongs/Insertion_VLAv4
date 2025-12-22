#!/usr/bin/env python
"""
Upload collected HDF5 episodes to Hugging Face Hub

This script uploads VLA dataset from collected_data directory to Hugging Face.

Usage:
    # Set your Hugging Face token
    export HF_TOKEN=your_token_here

    # Upload dataset
    python upload_to_hf.py \
        --data_dir collected_data \
        --repo_id "your-username/vla-meca500-insertion" \
        --dataset_name "VLA MECA500 Insertion Dataset"

    # Upload as private
    python upload_to_hf.py \
        --data_dir collected_data \
        --repo_id "your-username/vla-meca500-insertion" \
        --private
"""

import argparse
import h5py
import logging
import os
import sys
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np
from PIL import Image
from tqdm import tqdm

try:
    from datasets import Dataset, DatasetDict, Features, Value, Image as ImageFeature, Sequence, Array2D
    from huggingface_hub import HfApi, create_repo
except ImportError:
    print("Error: Required packages not found. Install with:")
    print("  pip install datasets huggingface_hub pillow h5py")
    sys.exit(1)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_h5_episode(h5_path: Path) -> Dict:
    """Load data from a single HDF5 episode file.

    Args:
        h5_path: Path to HDF5 file

    Returns:
        Dictionary with episode data
    """
    logger.info(f"Loading: {h5_path.name}")

    with h5py.File(h5_path, 'r') as f:
        # Load observations
        qpos = f['observations/qpos'][:]
        ee_pose = f['observations/ee_pose'][:]

        # Load actions
        actions = f['action'][:]

        # Load timestamps
        timestamps = f['timestamp'][:]

        # Load images from all cameras
        images_grp = f['observations/images']
        camera_keys = sorted(list(images_grp.keys()))

        images_by_camera = {}
        for cam_key in camera_keys:
            images_by_camera[cam_key] = images_grp[cam_key][:]

        num_frames = len(qpos)

        return {
            "episode_id": h5_path.stem,
            "qpos": qpos,
            "ee_pose": ee_pose,
            "actions": actions,
            "timestamps": timestamps,
            "images": images_by_camera,
            "camera_keys": camera_keys,
            "num_frames": num_frames,
        }


def create_hf_dataset(
    h5_files: List[Path],
    output_dir: Optional[Path] = None,
) -> DatasetDict:
    """Create Hugging Face Dataset from HDF5 files.

    Args:
        h5_files: List of HDF5 file paths
        output_dir: Optional directory to save intermediate data

    Returns:
        DatasetDict with train split
    """
    logger.info(f"Processing {len(h5_files)} episodes...")

    all_samples = []

    for h5_file in tqdm(h5_files, desc="Loading episodes"):
        try:
            episode_data = load_h5_episode(h5_file)

            # Create frame-level samples
            for frame_idx in range(episode_data["num_frames"]):
                sample = {
                    "episode_id": episode_data["episode_id"],
                    "frame_index": frame_idx,
                    "timestamp": float(episode_data["timestamps"][frame_idx]),
                    "qpos": episode_data["qpos"][frame_idx].tolist(),
                    "ee_pose": episode_data["ee_pose"][frame_idx].tolist(),
                    "action": episode_data["actions"][frame_idx].tolist(),
                }

                # Add images from each camera
                for cam_key in episode_data["camera_keys"]:
                    img_array = episode_data["images"][cam_key][frame_idx]
                    # Convert to PIL Image
                    sample[f"image_{cam_key}"] = Image.fromarray(img_array)

                all_samples.append(sample)

        except Exception as e:
            logger.warning(f"Failed to load {h5_file.name}: {e}")
            continue

    logger.info(f"Total frames: {len(all_samples)}")

    if not all_samples:
        raise ValueError("No samples were loaded!")

    # Get camera keys from first sample
    camera_keys = [k.replace("image_", "") for k in all_samples[0].keys() if k.startswith("image_")]

    # Define features schema
    features = Features({
        "episode_id": Value("string"),
        "frame_index": Value("int32"),
        "timestamp": Value("float64"),
        "qpos": Sequence(Value("float32"), length=6),
        "ee_pose": Sequence(Value("float32"), length=6),
        "action": Sequence(Value("float32"), length=6),
    })

    # Add image features
    for cam_key in camera_keys:
        features[f"image_{cam_key}"] = ImageFeature()

    # Create dataset
    dataset = Dataset.from_list(all_samples, features=features)

    # Create DatasetDict
    dataset_dict = DatasetDict({
        "train": dataset,
    })

    logger.info(f"Dataset created: {len(dataset)} samples")

    return dataset_dict


def create_readme(
    repo_id: str,
    dataset_name: str,
    num_episodes: int,
    num_frames: int,
    camera_keys: List[str],
) -> str:
    """Create README.md for dataset card.

    Args:
        repo_id: Repository ID
        dataset_name: Dataset name
        num_episodes: Number of episodes
        num_frames: Total number of frames
        camera_keys: List of camera names

    Returns:
        README content as string
    """
    return f"""---
license: apache-2.0
task_categories:
- robotics
tags:
- robotics
- vision-language-action
- imitation-learning
- meca500
pretty_name: "{dataset_name}"
size_categories:
- 1K<n<10K
---

# {dataset_name}

This dataset contains robot demonstration data for vision-language-action (VLA) training using a MECA500 6-DOF collaborative robot.

## Dataset Description

- **Task:** Precision insertion/manipulation task
- **Robot:** MECA500 (6-DOF collaborative robot)
- **Episodes:** {num_episodes} demonstrations
- **Total Frames:** {num_frames:,}
- **Camera Views:** {len(camera_keys)} ({', '.join(camera_keys)})
- **Control Frequency:** 15 Hz
- **Camera Frequency:** 30 FPS

## Data Collection

The data was collected using:
- **Robot:** MECA500 at 100 Hz state sampling
- **Vision:** {len(camera_keys)} OAK-D cameras at 30 FPS
- **Control:** Xbox/PS controller with 6-axis control + D-pad
- **Format:** HDF5 files with images, robot states, and actions

## Dataset Structure

Each sample contains:
- `episode_id`: Episode identifier (string)
- `frame_index`: Frame number within episode (int)
- `timestamp`: Time in seconds (float)
- `qpos`: Joint positions in degrees [j1, j2, j3, j4, j5, j6] (6 floats)
- `ee_pose`: End-effector pose [x, y, z, rx, ry, rz] in mm and degrees (6 floats)
- `action`: Action command [vx, vy, vz, wx, wy, wz] (6 floats)
- `image_*`: RGB images from each camera view (480x640x3)

### Example Usage

```python
from datasets import load_dataset

# Load dataset
dataset = load_dataset("{repo_id}")

# Access samples
sample = dataset["train"][0]
print(f"Episode: {{sample['episode_id']}}")
print(f"Joint positions: {{sample['qpos']}}")
print(f"EE pose: {{sample['ee_pose']}}")
print(f"Action: {{sample['action']}}")
print(f"Image shape: {{sample['image_camera1'].size}}")

# Iterate through episodes
for sample in dataset["train"]:
    episode_id = sample["episode_id"]
    frame_idx = sample["frame_index"]
    # ... process sample
```

### Training Example

```python
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from datasets import load_dataset

# Load dataset
dataset = load_dataset("{repo_id}")

# Initialize policy
policy = SmolVLAPolicy.from_pretrained("lerobot/smolvla_base")

# Train on dataset
# ... your training code
```

## Splits

- **train:** {num_frames:,} frames from {num_episodes} episodes

## Use Cases

This dataset is suitable for:
- Vision-Language-Action (VLA) model training
- Imitation learning / Behavioral cloning
- Multi-view visual servoing
- Robot manipulation research
- End-to-end visuomotor policy learning

## Dataset Format

The data follows LeRobot-compatible structure:
- Observations: `qpos` (joint states), `ee_pose` (end-effector pose), `image_*` (camera views)
- Actions: 6-DOF Cartesian velocity commands
- Temporal structure: Sequential frames with timestamps

## Citation

If you use this dataset, please cite:

```bibtex
@dataset{{vla_insertion_dataset,
  title={{{dataset_name}}},
  year={{2024}},
  publisher={{Hugging Face}},
  howpublished={{\\url{{https://huggingface.co/datasets/{repo_id}}}}}
}}
```

## License

Apache 2.0

## Dataset Card Authors

Created with VLA data collection framework.
"""


def upload_to_huggingface(
    data_dir: Path,
    repo_id: str,
    dataset_name: str,
    private: bool = False,
    token: Optional[str] = None,
) -> str:
    """Upload dataset to Hugging Face Hub.

    Args:
        data_dir: Directory containing HDF5 files
        repo_id: Hugging Face repository ID (e.g., "username/dataset-name")
        dataset_name: Human-readable dataset name
        private: Whether to create private repository
        token: Hugging Face API token (or use HF_TOKEN env var)

    Returns:
        URL of uploaded dataset
    """
    # Get token from env if not provided
    if token is None:
        token = os.getenv("HF_TOKEN")
        if token is None:
            raise ValueError(
                "Hugging Face token not found. Either:\n"
                "  1. Set HF_TOKEN environment variable: export HF_TOKEN=your_token\n"
                "  2. Pass --token argument\n"
                "Get your token at: https://huggingface.co/settings/tokens"
            )

    # Find all HDF5 files
    h5_files = sorted(data_dir.glob("episode_*.h5"))

    if not h5_files:
        raise ValueError(f"No episode_*.h5 files found in {data_dir}")

    logger.info(f"Found {len(h5_files)} episodes in {data_dir}")

    # Create dataset
    dataset_dict = create_hf_dataset(h5_files)

    # Get camera keys from first sample
    sample = dataset_dict["train"][0]
    camera_keys = [k.replace("image_", "") for k in sample.keys() if k.startswith("image_")]

    # Create README
    readme_content = create_readme(
        repo_id=repo_id,
        dataset_name=dataset_name,
        num_episodes=len(h5_files),
        num_frames=len(dataset_dict["train"]),
        camera_keys=camera_keys,
    )

    # Create repository
    logger.info(f"Creating repository: {repo_id}")
    try:
        api = HfApi(token=token)
        repo_url = create_repo(
            repo_id=repo_id,
            repo_type="dataset",
            private=private,
            token=token,
            exist_ok=True,
        )
        logger.info(f"Repository created/updated: {repo_url}")
    except Exception as e:
        logger.error(f"Failed to create repository: {e}")
        raise

    # Push dataset to hub
    logger.info("Uploading dataset to Hugging Face Hub...")
    try:
        dataset_dict.push_to_hub(
            repo_id=repo_id,
            token=token,
            private=private,
        )

        # Upload README
        with open("README.md", "w") as f:
            f.write(readme_content)

        api.upload_file(
            path_or_fileobj="README.md",
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="dataset",
            token=token,
        )

        os.remove("README.md")

        dataset_url = f"https://huggingface.co/datasets/{repo_id}"
        logger.info(f"✅ Dataset uploaded successfully: {dataset_url}")

        return dataset_url

    except Exception as e:
        logger.error(f"Failed to upload dataset: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(
        description="Upload VLA dataset to Hugging Face Hub",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Upload dataset (requires HF_TOKEN env var)
  export HF_TOKEN=your_token_here
  python upload_to_hf.py \\
      --data_dir collected_data \\
      --repo_id "username/vla-meca500-insertion" \\
      --dataset_name "VLA MECA500 Insertion Dataset"

  # Upload as private
  python upload_to_hf.py \\
      --data_dir collected_data \\
      --repo_id "username/vla-meca500-insertion" \\
      --private
        """
    )

    parser.add_argument(
        "--data_dir",
        type=str,
        default="collected_data",
        help="Directory containing episode_*.h5 files"
    )

    parser.add_argument(
        "--repo_id",
        type=str,
        required=True,
        help='Hugging Face repository ID (e.g., "username/dataset-name")'
    )

    parser.add_argument(
        "--dataset_name",
        type=str,
        default="VLA MECA500 Dataset",
        help="Human-readable dataset name for README"
    )

    parser.add_argument(
        "--private",
        action="store_true",
        help="Create private repository (default: public)"
    )

    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Hugging Face API token (or use HF_TOKEN env var)"
    )

    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    if not data_dir.exists():
        logger.error(f"Directory not found: {data_dir}")
        sys.exit(1)

    logger.info("="*80)
    logger.info("VLA Dataset Upload to Hugging Face")
    logger.info("="*80)
    logger.info(f"Data directory: {data_dir}")
    logger.info(f"Repository: {args.repo_id}")
    logger.info(f"Dataset name: {args.dataset_name}")
    logger.info(f"Private: {args.private}")
    logger.info("="*80)

    try:
        dataset_url = upload_to_huggingface(
            data_dir=data_dir,
            repo_id=args.repo_id,
            dataset_name=args.dataset_name,
            private=args.private,
            token=args.token,
        )

        logger.info("\n" + "="*80)
        logger.info("✅ SUCCESS!")
        logger.info("="*80)
        logger.info(f"Dataset URL: {dataset_url}")
        logger.info("\nTo use this dataset:")
        logger.info(f'  from datasets import load_dataset')
        logger.info(f'  dataset = load_dataset("{args.repo_id}")')
        logger.info("="*80)

    except Exception as e:
        logger.error(f"\n❌ Upload failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
