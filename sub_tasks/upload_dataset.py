#!/usr/bin/env python
"""
Upload VLA Dataset to Hugging Face Hub

This script uploads VLA insertion dataset episodes to Hugging Face Hub.
It processes multiple episodes, creates dataset metadata, and uploads everything.

Usage:
    # Upload specific episodes
    python upload_dataset.py \
        --episode_dirs /path/to/episode1 /path/to/episode2 \
        --repo_id "username/vla-insertion-dataset" \
        --dataset_name "VLA Insertion - Blue Point"

    # Upload all episodes from a color directory
    python upload_dataset.py \
        --episode_dir /home/najo/NAS/VLA/dataset/New_dataset2/Blue_point \
        --repo_id "username/vla-insertion-blue" \
        --max_episodes 10

Requirements:
    - datasets
    - huggingface_hub
    - pillow
"""

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

try:
    from datasets import Dataset, DatasetDict, Features, Value, Image as ImageFeature, Sequence
    from huggingface_hub import HfApi, create_repo, upload_folder
except ImportError:
    print("Error: Required packages not found. Install with:")
    print("  pip install datasets huggingface_hub pillow")
    sys.exit(1)

# Add Train directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "Train"))

from lerobot.utils.utils import init_logging

# Initialize logging
init_logging()
logger = logging.getLogger(__name__)


def find_episodes(base_dir: Path, max_episodes: Optional[int] = None) -> List[Path]:
    """Find episode directories in a base directory.

    Args:
        base_dir: Base directory to search
        max_episodes: Maximum number of episodes to find

    Returns:
        List of episode directory paths
    """
    if not base_dir.exists():
        raise FileNotFoundError(f"Directory not found: {base_dir}")

    # Check if base_dir is itself an episode
    if (base_dir / "metadata.json").exists():
        return [base_dir]

    # Find episode subdirectories
    episode_dirs = []
    for subdir in sorted(base_dir.iterdir()):
        if subdir.is_dir() and (subdir / "metadata.json").exists():
            episode_dirs.append(subdir)
            if max_episodes and len(episode_dirs) >= max_episodes:
                break

    if not episode_dirs:
        raise ValueError(f"No episodes found in {base_dir}")

    return episode_dirs


def load_episode_data(episode_dir: Path) -> Dict:
    """Load data from a single episode.

    Args:
        episode_dir: Episode directory path

    Returns:
        Dictionary with episode data
    """
    logger.info(f"Loading episode: {episode_dir.name}")

    # Load metadata
    with open(episode_dir / "metadata.json", "r") as f:
        metadata = json.load(f)

    # Load robot states
    robot_states_path = episode_dir / "robot_states.npz"
    if robot_states_path.exists():
        with np.load(robot_states_path) as data:
            joints = data.get("joints", data["robot_states"][:, :6])
            poses = data.get("poses", data["robot_states"][:, 6:])
    else:
        # Try CSV
        csv_path = episode_dir / "robot_states.csv"
        df = pd.read_csv(csv_path)
        joint_cols = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"]
        pose_cols = ["pose_x", "pose_y", "pose_z", "pose_a", "pose_b", "pose_r"]
        joints = df[joint_cols].to_numpy(dtype=np.float32)
        poses = df[pose_cols].to_numpy(dtype=np.float32)

    # Load images from all camera views
    camera_views = metadata.get("camera_views", [])
    images_by_view = {}

    for view in camera_views:
        view_dir = episode_dir / "images" / view
        if not view_dir.exists():
            view_dir = episode_dir / view

        if view_dir.exists():
            image_files = sorted(view_dir.glob("*.jpg"))
            images_by_view[view] = image_files

    # Load sensor data (optional)
    sensor_data = None
    sensor_files = list(episode_dir.glob("sensor_data_*.npz"))
    if sensor_files:
        with np.load(sensor_files[0]) as data:
            sensor_data = {
                "alines": data.get("alines"),
                "forces": data.get("forces"),
                "timestamps": data.get("timestamps"),
            }

    return {
        "episode_id": episode_dir.name,
        "metadata": metadata,
        "joints": joints,
        "poses": poses,
        "images": images_by_view,
        "sensor_data": sensor_data,
        "num_frames": len(poses),
    }


def create_dataset_dict(
    episode_dirs: List[Path],
    output_dir: Path,
) -> DatasetDict:
    """Create Hugging Face Dataset from episodes.

    Args:
        episode_dirs: List of episode directories
        output_dir: Directory to save processed data

    Returns:
        DatasetDict with train split
    """
    logger.info(f"Processing {len(episode_dirs)} episodes...")

    all_data = []

    for episode_dir in tqdm(episode_dirs, desc="Loading episodes"):
        try:
            episode_data = load_episode_data(episode_dir)

            # Create frame-level samples
            num_frames = episode_data["num_frames"]
            camera_views = list(episode_data["images"].keys())

            for frame_idx in range(num_frames):
                sample = {
                    "episode_id": episode_data["episode_id"],
                    "frame_index": frame_idx,
                    "timestamp": frame_idx / episode_data["metadata"].get("robot_hz", 100),
                }

                # Add robot state
                sample["joint_positions"] = episode_data["joints"][frame_idx].tolist()
                sample["end_effector_pose"] = episode_data["poses"][frame_idx].tolist()

                # Add images
                for view_idx, view in enumerate(camera_views):
                    images = episode_data["images"][view]
                    if frame_idx < len(images):
                        sample[f"image_{view}"] = str(images[frame_idx])

                # Add sensor data if available
                if episode_data["sensor_data"] is not None:
                    sensor_ratio = len(episode_data["sensor_data"]["timestamps"]) / num_frames
                    sensor_idx = int(frame_idx * sensor_ratio)
                    if sensor_idx < len(episode_data["sensor_data"]["alines"]):
                        sample["sensor_alines"] = episode_data["sensor_data"]["alines"][sensor_idx].tolist()
                        sample["sensor_force"] = float(episode_data["sensor_data"]["forces"][sensor_idx])

                all_data.append(sample)

        except Exception as e:
            logger.warning(f"Failed to load episode {episode_dir}: {e}")
            continue

    logger.info(f"Total frames: {len(all_data)}")

    # Convert to pandas DataFrame
    df = pd.DataFrame(all_data)

    # Define features schema
    features = Features({
        "episode_id": Value("string"),
        "frame_index": Value("int32"),
        "timestamp": Value("float32"),
        "joint_positions": Sequence(Value("float32"), length=6),
        "end_effector_pose": Sequence(Value("float32"), length=6),
    })

    # Add image features
    camera_views = [k for k in df.columns if k.startswith("image_")]
    for view in camera_views:
        features[view] = ImageFeature()

    # Add sensor features if present
    if "sensor_alines" in df.columns:
        features["sensor_alines"] = Sequence(Value("float32"))
        features["sensor_force"] = Value("float32")

    # Create dataset
    dataset = Dataset.from_pandas(df, features=features)

    # Create DatasetDict
    dataset_dict = DatasetDict({
        "train": dataset,
    })

    logger.info("Dataset created successfully!")

    return dataset_dict


def create_dataset_card(
    repo_id: str,
    dataset_name: str,
    episode_dirs: List[Path],
    num_frames: int,
    camera_views: List[str],
) -> str:
    """Create README.md dataset card.

    Args:
        repo_id: Repository ID
        dataset_name: Dataset name
        episode_dirs: List of episode directories
        num_frames: Total number of frames
        camera_views: List of camera views

    Returns:
        Dataset card markdown content
    """
    # Analyze episodes
    color_name = episode_dirs[0].parent.name if episode_dirs else "Unknown"
    num_episodes = len(episode_dirs)

    card = f"""---
license: apache-2.0
task_categories:
- robotics
tags:
- robotics
- vision-language-action
- insertion-task
- meca500
pretty_name: "{dataset_name}"
size_categories:
- 1K<n<10K
---

# {dataset_name}

This dataset contains robot demonstrations for a precision needle insertion task using a Meca500 robot.

## Dataset Description

- **Task:** Precision needle insertion into colored insertion points
- **Robot:** Meca500 (6-DOF collaborative robot)
- **Target:** {color_name}
- **Episodes:** {num_episodes} demonstrations
- **Total Frames:** {num_frames:,}
- **Camera Views:** {len(camera_views)} ({', '.join(camera_views)})

## Data Collection

The data was collected using:
- **Robot Control:** Meca500 at 100 Hz
- **Vision:** {len(camera_views)} OAK cameras at 30 FPS
- **Sensor:** OCT (Optical Coherence Tomography) sensor at 650 Hz
- **Force Feedback:** Contact force measurements

## Dataset Structure

Each sample contains:
- `episode_id`: Episode identifier
- `frame_index`: Frame number within episode
- `timestamp`: Time in seconds
- `joint_positions`: Robot joint angles (6 values)
- `end_effector_pose`: End-effector pose [x, y, z, alpha, beta, gamma]
- `image_*`: RGB images from each camera view (640x480)
- `sensor_alines`: OCT A-line data (1025 values) [optional]
- `sensor_force`: Contact force measurement [optional]

### Example

```python
from datasets import load_dataset

# Load dataset
dataset = load_dataset("{repo_id}")

# Access samples
sample = dataset["train"][0]
print(f"Episode: {{sample['episode_id']}}")
print(f"Pose: {{sample['end_effector_pose']}}")
print(f"Image shape: {{sample['image_View1'].size}}")
```

## Splits

- **train:** {num_frames:,} frames from {num_episodes} episodes

## Use Cases

This dataset is suitable for:
- Vision-Language-Action (VLA) model training
- Imitation learning
- Visual servoing research
- Multi-modal robot learning
- Contact-rich manipulation

## Citation

If you use this dataset, please cite:

```bibtex
@dataset{{vla_insertion_dataset,
  title={{{dataset_name}}},
  author={{Your Name}},
  year={{2024}},
  publisher={{Hugging Face}},
  howpublished={{\\url{{https://huggingface.co/datasets/{repo_id}}}}},
}}
```

## Dataset Card Authors

Created by the VLA Insertion Task team.

## License

Apache 2.0
"""

    return card


def upload_dataset(
    episode_dirs: List[Path],
    repo_id: str,
    dataset_name: str,
    output_dir: Path,
    private: bool = False,
    token: Optional[str] = None,
) -> str:
    """Upload dataset to Hugging Face Hub.

    Args:
        episode_dirs: List of episode directories
        repo_id: Hugging Face repository ID
        dataset_name: Dataset name
        output_dir: Directory to save processed data
        private: Whether to create private repository
        token: Hugging Face API token

    Returns:
        URL of uploaded dataset
    """
    # Create dataset
    dataset_dict = create_dataset_dict(episode_dirs, output_dir)

    # Get camera views from first sample
    sample = dataset_dict["train"][0]
    camera_views = [k.replace("image_", "") for k in sample.keys() if k.startswith("image_")]

    # Create dataset card
    card_content = create_dataset_card(
        repo_id=repo_id,
        dataset_name=dataset_name,
        episode_dirs=episode_dirs,
        num_frames=len(dataset_dict["train"]),
        camera_views=camera_views,
    )

    # Save dataset card
    card_path = output_dir / "README.md"
    with open(card_path, "w") as f:
        f.write(card_content)

    # Save dataset to disk
    logger.info(f"Saving dataset to {output_dir}...")
    dataset_dict.save_to_disk(str(output_dir / "dataset"))

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
        logger.info(f"Repository created: {repo_url}")
    except Exception as e:
        logger.error(f"Failed to create repository: {e}")
        raise

    # Push to hub
    logger.info("Uploading dataset to Hub...")
    try:
        dataset_dict.push_to_hub(
            repo_id=repo_id,
            token=token,
            private=private,
        )

        # Upload README separately
        api.upload_file(
            path_or_fileobj=str(card_path),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="dataset",
            token=token,
        )

        dataset_url = f"https://huggingface.co/datasets/{repo_id}"
        logger.info(f"Dataset uploaded: {dataset_url}")

        return dataset_url

    except Exception as e:
        logger.error(f"Failed to upload dataset: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(
        description="Upload VLA Dataset to Hugging Face Hub"
    )

    # Episode selection
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--episode_dirs",
        type=str,
        nargs="+",
        help="List of episode directories to upload"
    )
    group.add_argument(
        "--episode_dir",
        type=str,
        help="Base directory containing episodes (will upload all or up to max_episodes)"
    )

    parser.add_argument(
        "--max_episodes",
        type=int,
        default=None,
        help="Maximum number of episodes to upload (only with --episode_dir)"
    )

    # Repository settings
    parser.add_argument(
        "--repo_id",
        type=str,
        required=True,
        help='Hugging Face repository ID (e.g., "username/vla-insertion-blue")'
    )

    parser.add_argument(
        "--dataset_name",
        type=str,
        default="VLA Insertion Dataset",
        help="Dataset name for README"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/dataset_upload",
        help="Directory to save processed dataset"
    )

    parser.add_argument(
        "--private",
        action="store_true",
        help="Create private repository"
    )

    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Hugging Face API token"
    )

    parser.add_argument(
        "--no_upload",
        action="store_true",
        help="Only prepare dataset, do not upload"
    )

    args = parser.parse_args()

    # Find episodes
    if args.episode_dirs:
        episode_dirs = [Path(d) for d in args.episode_dirs]
    else:
        episode_dirs = find_episodes(Path(args.episode_dir), args.max_episodes)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("="*80)
    logger.info("VLA Dataset Upload to Hugging Face")
    logger.info("="*80)
    logger.info(f"Episodes: {len(episode_dirs)}")
    for ep in episode_dirs[:5]:
        logger.info(f"  - {ep.name}")
    if len(episode_dirs) > 5:
        logger.info(f"  ... and {len(episode_dirs) - 5} more")
    logger.info(f"Repository: {args.repo_id}")
    logger.info(f"Output directory: {output_dir}")
    logger.info("="*80)

    # Upload dataset
    if not args.no_upload:
        try:
            dataset_url = upload_dataset(
                episode_dirs=episode_dirs,
                repo_id=args.repo_id,
                dataset_name=args.dataset_name,
                output_dir=output_dir,
                private=args.private,
                token=args.token,
            )

            logger.info("\n" + "="*80)
            logger.info("SUCCESS!")
            logger.info("="*80)
            logger.info(f"Dataset uploaded to: {dataset_url}")
            logger.info("\nTo use this dataset:")
            logger.info(f'  from datasets import load_dataset')
            logger.info(f'  dataset = load_dataset("{args.repo_id}")')
            logger.info("="*80)

        except Exception as e:
            logger.error(f"Upload failed: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    else:
        # Just prepare dataset
        try:
            dataset_dict = create_dataset_dict(episode_dirs, output_dir)
            logger.info(f"Dataset prepared in: {output_dir}")
            logger.info("Use --no-upload flag removed to upload")
        except Exception as e:
            logger.error(f"Failed to prepare dataset: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)


if __name__ == "__main__":
    main()
