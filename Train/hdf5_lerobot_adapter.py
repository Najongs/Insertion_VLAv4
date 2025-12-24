"""
HDF5 LeRobot Adapter for New VLA Dataset

This module provides an adapter to convert HDF5-format VLA dataset to LeRobot format
for training SmolVLA and other vision-language-action models.
"""

import torch
import numpy as np
import h5py
from pathlib import Path
from typing import Dict, List, Optional, Union
from torch.utils.data import Dataset, ConcatDataset
import logging
import random
import cv2

logger = logging.getLogger(__name__)


class HDF5LeRobotDataset(Dataset):
    """
    Adapter that loads HDF5 VLA episodes and provides LeRobot-compatible samples.

    HDF5 structure:
    - action: (N, 6) - delta pose actions
    - observations/
        - ee_pose: (N, 6) - end-effector poses
        - qpos: (N, 6) - joint positions
        - images/
            - camera1: (N, 480, 640, 3)
            - camera2: (N, 480, 640, 3)
            - camera3: (N, 480, 640, 3)
    - timestamp: (N,)

    LeRobot expects samples with:
    {
        "observation.images.camera1": Tensor (C, H, W),
        "observation.images.camera2": Tensor (C, H, W),
        "observation.images.camera3": Tensor (C, H, W),
        "observation.state": Tensor (state_dim,),
        "action": Tensor (action_dim,),
        "task": str,
        "timestamp": float,
        "frame_index": int,
        "episode_index": int,
        "index": int,
    }
    """

    def __init__(
        self,
        hdf5_path: str,
        episode_index: int = 0,
        horizon: int = 1,
        use_qpos: bool = False,
        use_ee_pose: bool = True,
        task_instruction: str = "Insert the needle into the target point",
        camera_dropout_prob: float = 0.0,
        min_cameras: int = 1,
        augment: bool = True,
        augment_brightness: float = 0.2,
        augment_contrast: float = 0.2,
        augment_saturation: float = 0.2,
        augment_hue: float = 0.05,
        augment_noise: float = 0.02,
    ):
        """
        Args:
            hdf5_path: Path to HDF5 episode file
            episode_index: Episode index for this dataset
            horizon: Action prediction horizon (for future multi-step actions)
            use_qpos: Use joint positions for state (6 dims)
            use_ee_pose: Use end-effector pose for state (6 dims, default)
            task_instruction: Task instruction text
            camera_dropout_prob: Probability of dropping out cameras (0.0 = disabled)
            min_cameras: Minimum number of cameras to keep active
            augment: Enable image augmentation
            augment_brightness: Max brightness adjustment factor
            augment_contrast: Max contrast adjustment factor
            augment_saturation: Max saturation adjustment factor
            augment_hue: Max hue adjustment (in degrees / 360)
            augment_noise: Gaussian noise std deviation
        """
        super().__init__()

        self.hdf5_path = Path(hdf5_path)
        self.episode_index = episode_index
        self.horizon = horizon
        self.use_qpos = use_qpos
        self.use_ee_pose = use_ee_pose
        self.task = task_instruction

        # Augmentation settings
        self.camera_dropout_prob = camera_dropout_prob
        self.min_cameras = min_cameras
        self.augment = augment
        self.augment_brightness = augment_brightness
        self.augment_contrast = augment_contrast
        self.augment_saturation = augment_saturation
        self.augment_hue = augment_hue
        self.augment_noise = augment_noise

        # Load HDF5 file
        self.h5file = h5py.File(str(self.hdf5_path), 'r')

        # Get dataset shapes
        self.num_frames = self.h5file['action'].shape[0]
        self.num_cameras = len(self.h5file['observations']['images'].keys())

        # Determine state dimension
        if use_qpos and use_ee_pose:
            self.state_dim = 12  # qpos (6) + ee_pose (6)
        elif use_qpos:
            self.state_dim = 6  # qpos only
        else:
            self.state_dim = 6  # ee_pose only (default)

        logger.info(f"Loaded HDF5 episode: {self.hdf5_path.name}")
        logger.info(f"  Episode index: {self.episode_index}")
        logger.info(f"  Total frames: {self.num_frames}")
        logger.info(f"  Cameras: {self.num_cameras}")
        logger.info(f"  State dim: {self.state_dim} ({'qpos+ee_pose' if use_qpos and use_ee_pose else 'qpos' if use_qpos else 'ee_pose'})")
        logger.info(f"  Task: {self.task}")

    def __len__(self) -> int:
        # Subtract horizon-1 to ensure we can always get full action sequences
        return max(0, self.num_frames - self.horizon + 1)

    def apply_image_augmentation(self, img_np: np.ndarray) -> np.ndarray:
        """
        Apply image augmentation to a single image.

        Args:
            img_np: Image array in [0, 1] range, shape (H, W, C)

        Returns:
            Augmented image in [0, 1] range
        """
        if not self.augment:
            return img_np

        # Convert to uint8 for cv2 operations
        img_uint8 = (img_np * 255).astype(np.uint8)

        # Random brightness and contrast adjustment
        if self.augment_brightness > 0 or self.augment_contrast > 0:
            alpha = 1.0 + random.uniform(-self.augment_contrast, self.augment_contrast)  # Contrast
            beta = random.uniform(-self.augment_brightness, self.augment_brightness) * 255  # Brightness
            img_uint8 = cv2.convertScaleAbs(img_uint8, alpha=alpha, beta=beta)

        # Random color augmentation (HSV)
        if self.augment_saturation > 0 or self.augment_hue > 0:
            img_hsv = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2HSV).astype(np.float32)

            # Adjust hue
            if self.augment_hue > 0:
                hue_shift = random.uniform(-self.augment_hue, self.augment_hue) * 180
                img_hsv[:, :, 0] = (img_hsv[:, :, 0] + hue_shift) % 180

            # Adjust saturation
            if self.augment_saturation > 0:
                sat_scale = 1.0 + random.uniform(-self.augment_saturation, self.augment_saturation)
                img_hsv[:, :, 1] = np.clip(img_hsv[:, :, 1] * sat_scale, 0, 255)

            img_uint8 = cv2.cvtColor(img_hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

        # Convert back to float [0, 1]
        img_float = img_uint8.astype(np.float32) / 255.0

        # Add Gaussian noise
        if self.augment_noise > 0:
            noise = np.random.normal(0, self.augment_noise, img_float.shape).astype(np.float32)
            img_float = np.clip(img_float + noise, 0.0, 1.0)

        return img_float

    def __getitem__(self, idx: int) -> Dict:
        """
        Get a sample in LeRobot format.

        Returns:
            Dictionary with LeRobot-compatible keys and values.
        """
        # Create LeRobot sample
        lerobot_sample = {
            # Metadata
            "task": self.task,
            "timestamp": float(self.h5file['timestamp'][idx]),
            "frame_index": idx,
            "episode_index": self.episode_index,
            "index": idx,  # Will be updated by ConcatDataset
        }

        # Determine which cameras to dropout
        active_cameras = list(range(1, self.num_cameras + 1))
        if self.camera_dropout_prob > 0 and random.random() < self.camera_dropout_prob:
            # Calculate how many cameras to keep active
            num_to_keep = max(self.min_cameras, random.randint(self.min_cameras, self.num_cameras))
            if num_to_keep < self.num_cameras:
                active_cameras = random.sample(active_cameras, num_to_keep)

        # Process images: Load from HDF5 and convert to (C, H, W) tensors
        for cam_idx in range(1, self.num_cameras + 1):
            cam_key = f"camera{cam_idx}"
            try:
                # Check if this camera should be dropped out
                if cam_idx not in active_cameras:
                    # Camera dropout: use black image
                    lerobot_sample[f"observation.images.{cam_key}"] = torch.zeros(3, 512, 512)
                    continue

                # Load image from HDF5: (H, W, C) in [0, 255]
                img_np = self.h5file['observations']['images'][cam_key][idx]

                # Resize to 512x512 (required by SmolVLA model)
                img_np = cv2.resize(img_np, (512, 512), interpolation=cv2.INTER_LINEAR)

                # Convert to float and normalize to [0, 1]
                img_np = img_np.astype(np.float32) / 255.0

                # Apply augmentation
                img_np = self.apply_image_augmentation(img_np)

                # Convert to tensor: (H, W, C) -> (C, H, W)
                img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).contiguous()

                # Add to sample with LeRobot naming convention
                lerobot_sample[f"observation.images.{cam_key}"] = img_tensor

            except Exception as e:
                logger.warning(f"Failed to load {cam_key} at frame {idx}: {e}")
                # Create dummy image if loading fails
                lerobot_sample[f"observation.images.{cam_key}"] = torch.zeros(3, 512, 512)

        # Process robot state
        state_parts = []

        if self.use_qpos:
            qpos = self.h5file['observations']['qpos'][idx]
            state_parts.append(qpos)

        if self.use_ee_pose:
            ee_pose = self.h5file['observations']['ee_pose'][idx]
            state_parts.append(ee_pose)

        # Concatenate state parts
        if len(state_parts) > 1:
            state = np.concatenate(state_parts, axis=0)
        else:
            state = state_parts[0]

        # Convert to tensor
        lerobot_sample["observation.state"] = torch.from_numpy(state.astype(np.float32))

        # Process actions
        if self.horizon == 1:
            # Single-step action
            action = self.h5file['action'][idx]
            lerobot_sample["action"] = torch.from_numpy(action.astype(np.float32))
        else:
            # Multi-step action chunk
            end_idx = min(idx + self.horizon, self.num_frames)
            actions = self.h5file['action'][idx:end_idx]

            # Pad if necessary
            if actions.shape[0] < self.horizon:
                padding = np.repeat(actions[-1:], self.horizon - actions.shape[0], axis=0)
                actions = np.concatenate([actions, padding], axis=0)

            lerobot_sample["action"] = torch.from_numpy(actions.astype(np.float32))

        return lerobot_sample

    def __del__(self):
        """Close HDF5 file when dataset is destroyed."""
        if hasattr(self, 'h5file'):
            self.h5file.close()


def create_hdf5_lerobot_dataset(
    hdf5_paths: List[Union[str, Path]],
    horizon: int = 1,
    use_qpos: bool = False,
    use_ee_pose: bool = True,
    task_instruction: str = "Insert the needle into the target point",
    camera_dropout_prob: float = 0.0,
    min_cameras: int = 1,
    augment: bool = True,
    augment_brightness: float = 0.2,
    augment_contrast: float = 0.2,
    augment_saturation: float = 0.2,
    augment_hue: float = 0.05,
    augment_noise: float = 0.02,
) -> Dataset:
    """
    Create a combined HDF5 LeRobot dataset from multiple episodes.

    Args:
        hdf5_paths: List of HDF5 file paths
        horizon: Action prediction horizon
        use_qpos: Use joint positions for state
        use_ee_pose: Use end-effector pose for state (default)
        task_instruction: Task instruction text
        camera_dropout_prob: Probability of dropping out cameras
        min_cameras: Minimum number of cameras to keep active
        augment: Enable image augmentation
        augment_brightness: Max brightness adjustment factor
        augment_contrast: Max contrast adjustment factor
        augment_saturation: Max saturation adjustment factor
        augment_hue: Max hue adjustment (in degrees / 360)
        augment_noise: Gaussian noise std deviation

    Returns:
        Combined dataset (ConcatDataset if multiple episodes)
    """
    datasets = []

    for episode_idx, hdf5_path in enumerate(hdf5_paths):
        file_path = Path(hdf5_path)

        # Check if file exists
        if not file_path.exists():
            logger.warning(f"HDF5 file does not exist: {file_path}")
            continue

        # Create dataset for this episode
        try:
            dataset = HDF5LeRobotDataset(
                hdf5_path=str(file_path),
                episode_index=episode_idx,
                horizon=horizon,
                use_qpos=use_qpos,
                use_ee_pose=use_ee_pose,
                task_instruction=task_instruction,
                camera_dropout_prob=camera_dropout_prob,
                min_cameras=min_cameras,
                augment=augment,
                augment_brightness=augment_brightness,
                augment_contrast=augment_contrast,
                augment_saturation=augment_saturation,
                augment_hue=augment_hue,
                augment_noise=augment_noise,
            )
            datasets.append(dataset)
        except Exception as e:
            logger.error(f"Failed to load episode {file_path}: {e}")
            continue

    if len(datasets) == 0:
        raise ValueError("No valid episodes found!")

    logger.info(f"Created combined dataset with {len(datasets)} episodes")

    # Combine all episodes
    if len(datasets) == 1:
        combined_dataset = datasets[0]
    else:
        combined_dataset = ConcatDataset(datasets)

    logger.info(f"Total samples: {len(combined_dataset)}")

    return combined_dataset


def hdf5_lerobot_collate_fn(batch: List[Dict]) -> Dict:
    """
    Custom collate function for HDF5 LeRobot batches.

    Handles:
    - Stacking tensors (images, state, actions)
    - Keeping lists (task strings)
    - Maintaining metadata

    Args:
        batch: List of samples from HDF5LeRobotDataset

    Returns:
        Batched dictionary
    """
    if len(batch) == 0:
        return {}

    # Get all keys from first sample
    keys = batch[0].keys()

    batched = {}

    for key in keys:
        values = [sample[key] for sample in batch]

        # Handle different types
        if key == "task":
            # Keep as list of strings
            batched[key] = values
        elif key in ["timestamp", "frame_index", "episode_index", "index"]:
            # Stack scalars into tensor
            batched[key] = torch.tensor(values)
        elif isinstance(values[0], torch.Tensor):
            # Stack tensors
            batched[key] = torch.stack(values, dim=0)
        else:
            # Keep as list for other types
            batched[key] = values

    return batched


if __name__ == "__main__":
    """
    Test the HDF5 LeRobot adapter.
    """
    import logging
    logging.basicConfig(level=logging.INFO)

    print("🧪 Testing HDF5 LeRobot Adapter...\n")

    # Test with a single episode
    test_file = "/home/irom/NAS/VLA/Insertion_VLAv4/New_dataset/collected_data/episode_20251222_225152.h5"

    try:
        dataset = HDF5LeRobotDataset(
            hdf5_path=test_file,
            episode_index=0,
            use_ee_pose=True,
            use_qpos=False,
        )

        print(f"✅ Dataset created: {len(dataset)} samples\n")

        # Test first sample
        sample = dataset[0]

        print("📊 Sample structure:")
        for key, value in sample.items():
            if isinstance(value, torch.Tensor):
                print(f"  {key}: {value.shape} ({value.dtype})")
            else:
                print(f"  {key}: {type(value).__name__} = {value if not isinstance(value, str) else value[:50]}")

        print("\n✅ Sample loaded successfully!\n")

        # Test dataloader
        from torch.utils.data import DataLoader

        dataloader = DataLoader(
            dataset,
            batch_size=4,
            shuffle=False,
            collate_fn=hdf5_lerobot_collate_fn,
            num_workers=0,
        )

        batch = next(iter(dataloader))

        print("📦 Batch structure:")
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                print(f"  {key}: {value.shape} ({value.dtype})")
            elif isinstance(value, list):
                print(f"  {key}: List[{type(value[0]).__name__}] (len={len(value)})")

        print("\n✅ DataLoader works!\n")

        print("\n🎉 All tests passed!")

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
