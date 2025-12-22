"""
VLA Dataset for Insertion Task
통합 데이터셋 모듈 - Old/New format 지원

데이터셋 경로: /home/najo/NAS/VLA/dataset
- New_dataset2, New_dataset3, New_dataset4, New_dataset5, New_dataset6

Features:
- Metadata.json + NPZ 기반 데이터 로딩
- Multi-camera view support
- Sensor data (OCT alines + forces)
- Robot state (joints + poses)
- Action generation (delta pose + rotation)
- VL feature caching (optional)
"""

import json
from pathlib import Path
from typing import Optional, List, Dict, Any, Literal

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from scipy.spatial.transform import Rotation
from PIL import Image


class VLADataset(Dataset):
    """
    VLA Dataset for robot insertion task

    Args:
        data_dir: Episode directory path (contains metadata.json)
        horizon: Action prediction horizon (default: 8)
        sensor_window_size: Sensor history window size (default: 65)
        robot_window_size: Robot state history window size (default: 100)
        action_expert_hz: Action frequency (default: 10 Hz)
        instruction: Task instruction text (optional, auto-generated from task name)
        use_cache: Enable VL feature caching (default: False for training script)
    """

    def __init__(
        self,
        data_dir: str,
        horizon: int = 8,
        sensor_window_size: int = 65,
        robot_window_size: int = 100,
        action_expert_hz: int = 10,
        instruction: Optional[str] = None,
        use_cache: bool = False,
        cache_root: Optional[str] = None,
    ):
        self.data_dir = Path(data_dir)
        self.horizon = horizon
        self.sensor_window_size = sensor_window_size
        self.robot_window_size = robot_window_size
        self.action_expert_hz = action_expert_hz
        self.use_cache = use_cache
        self.cache_root = Path(cache_root) if cache_root else None

        # Load metadata
        self._load_metadata()

        # Generate instruction
        if instruction is None:
            task_name = self.data_dir.parent.name.replace("_", " ")
            self.instruction = self._generate_instruction(task_name)
        else:
            self.instruction = instruction

        # Load data files
        self._load_sensor_data()
        self._load_robot_states()
        self._load_images()

        # Calculate dataset size
        self.num_poses = len(self.poses) if self.has_robot_states else 0
        self.action_interval = int(self.robot_hz / self.action_expert_hz)
        self.num_actions = max(0, (self.num_poses - self.action_interval) // self.action_interval)
        self._total_samples = self.num_actions

    def _load_metadata(self):
        """Load metadata.json"""
        meta_path = self.data_dir / "metadata.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"metadata.json not found in {self.data_dir}")

        with open(meta_path, "r") as f:
            self.meta = json.load(f)

        self.robot_hz = self.meta.get("robot_hz", 100)
        self.sensor_hz = self.meta.get("sensor_hz", 650)

    def _generate_instruction(self, task_name: str) -> str:
        """Generate task instruction from task name"""
        return f"""Environment Context:
- This is a Meca500 robot.
- The end-effector made by 3d printer the needle tip have to contact with {task_name}.
- The scene is an optical table with many holes, but these are NOT targets.
- The ONLY true insertion target is the {task_name}.

Task:
You must analyze the views and determine the needle's relative position to the {task_name}.
Identify:
1) needle tip location
2) alignment relative to the {task_name} center
3) required direction to align for insertion
4) If the needle tip is inserted at the {task_name}, it is Done of task

Respond with:
- target visibility
- needle alignment
- required adjustment direction
- distance with {task_name} and needle tip point"""

    def _load_sensor_data(self):
        """Load sensor data (OCT alines + forces)"""
        # Find timestamped sensor file: sensor_data_YYYYMMDD_HHMMSS.npz
        sensor_files = list(self.data_dir.glob("sensor_data_*.npz"))
        if sensor_files:
            sensor_path = sensor_files[0]
        else:
            sensor_path = self.data_dir / "sensor_data.npz"

        if not sensor_path.exists():
            print(f"⚠️ Sensor file not found: {sensor_path}")
            self.has_sensor = False
            self.sensor_data = None
            self.sensor_length = 0
            return

        try:
            # Load sensor data with mmap for memory efficiency
            self.sensor_npz = np.load(sensor_path, mmap_mode="r")

            if 'alines' in self.sensor_npz and 'forces' in self.sensor_npz:
                # Raw sensor data format
                self.sensor_timestamps = self.sensor_npz["timestamps"][:]
                self.alines = self.sensor_npz["alines"]  # Shape: (N, 1025)
                self.forces = self.sensor_npz["forces"]  # Shape: (N,)
                self.sensor_length = len(self.sensor_timestamps)
                self.has_sensor = True
            else:
                print(f"⚠️ Unknown sensor data format in {sensor_path}")
                self.has_sensor = False
                self.sensor_data = None
                self.sensor_length = 0
        except Exception as e:
            print(f"⚠️ Error loading sensor data {sensor_path}: {e}")
            self.has_sensor = False
            self.sensor_data = None
            self.sensor_length = 0

    def _load_robot_states(self):
        """Load robot states (joints + poses)"""
        npz_path = self.data_dir / "robot_states.npz"
        csv_path = self.data_dir / "robot_states.csv"

        if npz_path.exists():
            try:
                with np.load(npz_path, mmap_mode="r") as data:
                    self.robot_states = np.array(data["robot_states"], dtype=np.float32)
                    self.joints = (
                        np.array(data["joints"], dtype=np.float32)
                        if "joints" in data
                        else self.robot_states[:, :6]
                    )
                    self.poses = (
                        np.array(data["poses"], dtype=np.float32)
                        if "poses" in data
                        else self.robot_states[:, 6:]
                    )
                self.has_robot_states = True
            except Exception as e:
                print(f"⚠️ Error loading robot states from NPZ: {e}")
                self.robot_states = np.zeros((1, 12), dtype=np.float32)
                self.joints = np.zeros((1, 6), dtype=np.float32)
                self.poses = np.zeros((1, 6), dtype=np.float32)
                self.has_robot_states = False

        elif csv_path.exists():
            try:
                joint_cols = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"]
                pose_cols = ["pose_x", "pose_y", "pose_z", "pose_a", "pose_b", "pose_r"]
                df = pd.read_csv(csv_path, usecols=joint_cols + pose_cols)

                self.joints = df[joint_cols].to_numpy(dtype=np.float32)
                self.poses = df[pose_cols].to_numpy(dtype=np.float32)
                self.robot_states = np.concatenate([self.joints, self.poses], axis=1)
                self.has_robot_states = True
            except Exception as e:
                print(f"⚠️ Error loading robot states from CSV: {e}")
                self.robot_states = np.zeros((1, 12), dtype=np.float32)
                self.joints = np.zeros((1, 6), dtype=np.float32)
                self.poses = np.zeros((1, 6), dtype=np.float32)
                self.has_robot_states = False
        else:
            self.robot_states = np.zeros((1, 12), dtype=np.float32)
            self.joints = np.zeros((1, 6), dtype=np.float32)
            self.poses = np.zeros((1, 6), dtype=np.float32)
            self.has_robot_states = False

    def _load_images(self):
        """Load image paths for each camera view"""
        self.images = {}
        for view_name in self.meta.get("camera_views", []):
            # Try images/ViewX directory first
            img_dir = self.data_dir / "images"
            view_dir = img_dir / view_name

            if not view_dir.exists():
                # Try ViewX directory directly
                view_dir = self.data_dir / view_name

            if view_dir.exists():
                files = sorted(
                    view_dir.glob("*.jpg"),
                    key=lambda x: self._extract_timestamp(x.stem)
                )
                self.images[view_name] = [str(f) for f in files]

    def _extract_timestamp(self, filename: str) -> float:
        """Extract timestamp from image filename"""
        try:
            return float(filename)
        except ValueError:
            # Extract last numeric part (timestamp)
            parts = filename.split('_')
            for part in reversed(parts):
                try:
                    return float(part)
                except ValueError:
                    continue
            return 0.0

    def __len__(self):
        return self._total_samples

    def __getitem__(self, idx: int):
        """
        Get a sample from the dataset

        Returns:
            dict with keys:
                - instruction: str
                - images: List[str] - image paths
                - sensor_data: torch.Tensor (sensor_window_size, 1026)
                - robot_states: torch.Tensor (robot_window_size, 12)
                - actions: torch.Tensor (horizon, 7)
                - has_sensor: bool
                - has_robot_states: bool
                - episode_id: str
                - timestamp: float
        """
        if idx >= self._total_samples:
            raise IndexError(f"Index {idx} out of range for dataset of size {self._total_samples}")

        # Get data for this index
        image_paths = self._get_images(idx)
        sensor_window = self._get_sensor_window(idx)
        robot_state_window = self._get_robot_state_window(idx)
        actions = self._get_actions(idx)

        # Get timestamp from first image if available
        timestamp = 0.0
        if image_paths and image_paths[0]:
            try:
                timestamp = float(Path(image_paths[0]).stem)
            except (ValueError, IndexError):
                pass

        return {
            "instruction": self.instruction,
            "images": image_paths,
            "sensor_data": torch.from_numpy(sensor_window),
            "robot_states": torch.from_numpy(robot_state_window),
            "actions": torch.from_numpy(actions),
            "has_sensor": self.has_sensor,
            "has_robot_states": self.has_robot_states,
            "episode_id": self.data_dir.name,
            "timestamp": timestamp,
        }

    def _get_images(self, idx: int) -> List[str]:
        """Get image paths for all views at given index"""
        image_paths = []

        if not self.images:
            return image_paths

        # Calculate image index from action index
        img_idx = idx * self.action_interval

        for view_name in sorted(self.images.keys()):
            view_images = self.images[view_name]
            if len(view_images) > 0:
                # Use closest available image
                actual_idx = min(img_idx, len(view_images) - 1)
                image_paths.append(view_images[actual_idx])

        return image_paths

    def _get_sensor_window(self, idx: int) -> np.ndarray:
        """
        Get sensor window (trailing/historical data only)

        Returns:
            np.ndarray: Shape (sensor_window_size, 1026)
                where 1026 = 1025 (alines) + 1 (forces)
        """
        if not self.has_sensor:
            return np.zeros((self.sensor_window_size, 1026), dtype=np.float32)

        # Calculate sensor index from robot index
        robot_idx = idx * self.action_interval
        sensor_ratio = self.sensor_hz / self.robot_hz  # ~6.5
        sensor_center_idx = int(robot_idx * sensor_ratio)

        # Extract TRAILING window (past only, no future data)
        sensor_end = sensor_center_idx
        sensor_start = max(0, sensor_end - self.sensor_window_size)

        # Ensure bounds
        sensor_end = min(sensor_end, self.sensor_length)
        sensor_start = max(0, sensor_end - self.sensor_window_size)

        # Load data slice
        alines_window = np.array(
            self.alines[sensor_start:sensor_end],
            dtype=np.float32
        )  # (N, 1025)
        forces_window = np.array(
            self.forces[sensor_start:sensor_end],
            dtype=np.float32
        )  # (N,)

        # Combine: alines (1025) + forces (1) = 1026 dimensions
        forces_expanded = forces_window[:, np.newaxis]  # (N, 1)
        sensor_window = np.concatenate([alines_window, forces_expanded], axis=1)  # (N, 1026)

        # Pad if necessary
        if sensor_window.shape[0] < self.sensor_window_size:
            pad_size = self.sensor_window_size - sensor_window.shape[0]
            pad = np.zeros((pad_size, 1026), dtype=np.float32)
            sensor_window = np.concatenate([sensor_window, pad], axis=0)

        return sensor_window

    def _get_robot_state_window(self, idx: int) -> np.ndarray:
        """
        Get robot state window (trailing/historical data only)

        Returns:
            np.ndarray: Shape (robot_window_size, 12)
        """
        if not self.has_robot_states:
            return np.zeros((self.robot_window_size, 12), dtype=np.float32)

        # Extract TRAILING window (past only)
        center_idx = idx * self.action_interval
        end_idx = center_idx
        start_idx = max(0, end_idx - self.robot_window_size)

        # Ensure bounds
        end_idx = min(end_idx, len(self.robot_states))
        robot_window = self.robot_states[start_idx:end_idx]

        # Pad if necessary
        if robot_window.shape[0] < self.robot_window_size:
            pad_size = self.robot_window_size - robot_window.shape[0]
            pad = np.zeros((pad_size, 12), dtype=np.float32)
            robot_window = np.concatenate([robot_window, pad], axis=0)

        return robot_window

    def _get_actions(self, action_step: int) -> np.ndarray:
        """
        Compute action sequence (delta pose + rotation)

        Returns:
            np.ndarray: Shape (horizon, 7) where 7 = [dx, dy, dz, drx, dry, drz, gripper]
        """
        actions = []

        for i in range(self.horizon):
            current_action_idx = action_step + i
            start_pose_idx = current_action_idx * self.action_interval
            end_pose_idx = start_pose_idx + self.action_interval

            if end_pose_idx >= len(self.poses):
                break

            # Check if near end of trajectory (terminal action)
            if (self.num_actions - current_action_idx) <= 5:
                # Terminal action: no movement
                delta_action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
            else:
                start_pose = self.poses[start_pose_idx]
                end_pose = self.poses[end_pose_idx]

                # Compute delta translation
                delta_translation = end_pose[:3] - start_pose[:3]

                # Compute delta rotation (as rotation vector)
                r_start = Rotation.from_euler("xyz", start_pose[3:], degrees=True)
                r_end = Rotation.from_euler("xyz", end_pose[3:], degrees=True)
                r_delta = r_end * r_start.inv()
                delta_rotation = r_delta.as_rotvec()

                # Combine: [dx, dy, dz, drx, dry, drz, gripper]
                delta_pose = np.concatenate([delta_translation, delta_rotation])
                delta_action = np.concatenate([delta_pose, [1.0]], axis=0)

            actions.append(delta_action)

        # Pad if necessary
        if not actions:
            default_action = np.array([0.0] * 6 + [1.0], dtype=np.float32)
            return np.tile(default_action, (self.horizon, 1))

        if len(actions) < self.horizon:
            # Repeat last action
            last_action = actions[-1]
            for _ in range(self.horizon - len(actions)):
                actions.append(last_action.copy())

        return np.array(actions, dtype=np.float32)


def collate_fn(batch):
    """
    Collate function for DataLoader

    Args:
        batch: List of samples from VLADataset

    Returns:
        Dictionary with batched tensors
    """
    instructions = [b["instruction"] for b in batch]

    # Sanitize image lists
    image_lists = []
    for b in batch:
        img_list = b.get("images", [])
        if img_list is None:
            image_lists.append([])
        else:
            image_lists.append([img for img in img_list if img is not None])

    # Pad sensor data to max length in batch
    sensor_tensors = [b["sensor_data"] for b in batch]
    max_sensor_len = max(t.shape[0] for t in sensor_tensors)

    padded_sensors = []
    for sensor in sensor_tensors:
        if sensor.shape[0] < max_sensor_len:
            pad_size = max_sensor_len - sensor.shape[0]
            pad = torch.zeros((pad_size, sensor.shape[1]), dtype=sensor.dtype)
            padded_sensors.append(torch.cat([sensor, pad], dim=0))
        else:
            padded_sensors.append(sensor)
    sensor_data = torch.stack(padded_sensors, dim=0)

    # Pad robot states to max length in batch
    robot_state_tensors = [b["robot_states"] for b in batch]
    max_robot_len = max(t.shape[0] for t in robot_state_tensors)

    padded_robot_states = []
    for robot_state in robot_state_tensors:
        if robot_state.shape[0] < max_robot_len:
            pad_size = max_robot_len - robot_state.shape[0]
            pad = torch.zeros((pad_size, robot_state.shape[1]), dtype=robot_state.dtype)
            padded_robot_states.append(torch.cat([robot_state, pad], dim=0))
        else:
            padded_robot_states.append(robot_state)
    robot_states = torch.stack(padded_robot_states, dim=0)

    # Stack actions
    actions = torch.stack([b["actions"] for b in batch], dim=0)

    # Masks
    has_sensor_mask = torch.tensor([b["has_sensor"] for b in batch], dtype=torch.bool)
    has_robot_states_mask = torch.tensor([b["has_robot_states"] for b in batch], dtype=torch.bool)

    # Metadata
    episode_ids = [b["episode_id"] for b in batch]
    timestamps = [b["timestamp"] for b in batch]

    return {
        "instruction": instructions,
        "images": image_lists,
        "sensor_data": sensor_data,
        "robot_states": robot_states,
        "actions": actions,
        "has_sensor_mask": has_sensor_mask,
        "has_robot_states_mask": has_robot_states_mask,
        "episode_ids": episode_ids,
        "timestamps": timestamps,
    }


def create_dataloader(
    dataset_paths: List[str],
    batch_size: int = 4,
    num_workers: int = 4,
    shuffle: bool = True,
    horizon: int = 8,
    sensor_window_size: int = 65,
    robot_window_size: int = 100,
    action_expert_hz: int = 10,
) -> DataLoader:
    """
    Create DataLoader from multiple dataset paths

    Args:
        dataset_paths: List of episode directory paths
        batch_size: Batch size
        num_workers: Number of worker processes
        shuffle: Whether to shuffle data
        horizon: Action prediction horizon
        sensor_window_size: Sensor history window size
        robot_window_size: Robot state history window size
        action_expert_hz: Action frequency (Hz)

    Returns:
        DataLoader instance
    """
    datasets = []

    for path in dataset_paths:
        path = Path(path)

        # Check if path is a task directory or episode directory
        if (path / "metadata.json").exists():
            # Single episode directory
            episode_dirs = [path]
        else:
            # Task directory containing multiple episodes
            episode_dirs = [
                d for d in path.iterdir()
                if d.is_dir() and (
                    d.name.startswith('episode_') or
                    d.name.startswith('data_collection_')
                )
            ]

        # Load each episode
        for episode_dir in episode_dirs:
            try:
                ds = VLADataset(
                    data_dir=str(episode_dir),
                    horizon=horizon,
                    sensor_window_size=sensor_window_size,
                    robot_window_size=robot_window_size,
                    action_expert_hz=action_expert_hz,
                )

                if len(ds) > 0:
                    datasets.append(ds)
                    print(f"✅ Loaded {episode_dir.name}: {len(ds)} samples")
                else:
                    print(f"⚠️ Empty dataset: {episode_dir.name}")
            except Exception as e:
                print(f"❌ Failed to load {episode_dir}: {e}")

    if not datasets:
        raise ValueError("No datasets loaded! Check dataset paths.")

    # Combine all datasets
    full_dataset = ConcatDataset(datasets)
    print(f"\n📊 Total dataset: {len(full_dataset)} samples from {len(datasets)} episodes")

    # Create DataLoader
    dataloader = DataLoader(
        full_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )

    return dataloader


# =====================================
# Test Code
# =====================================

if __name__ == "__main__":
    print("🧪 Testing VLA Dataset...")
    print()

    # Test dataset
    test_episode = "/home/najo/NAS/VLA/dataset/New_dataset2/Green_point/data_collection_20251108_053848"

    if not Path(test_episode).exists():
        print(f"❌ Test episode not found: {test_episode}")
        exit(1)

    print(f"📂 Testing with: {test_episode}")
    print()

    try:
        # Create dataset
        ds = VLADataset(
            data_dir=test_episode,
            horizon=8,
            sensor_window_size=65,
            robot_window_size=100,
            action_expert_hz=10,
        )

        print(f"✅ Dataset created: {len(ds)} samples")
        print(f"   Has sensor: {ds.has_sensor}")
        print(f"   Has robot states: {ds.has_robot_states}")
        print(f"   Camera views: {list(ds.images.keys())}")
        print()

        # Test sample
        if len(ds) > 0:
            sample = ds[0]
            print("📦 Sample 0:")
            print(f"   Instruction: {sample['instruction'][:100]}...")
            print(f"   Images: {len(sample['images'])} views")
            print(f"   Sensor data: {sample['sensor_data'].shape}")
            print(f"   Robot states: {sample['robot_states'].shape}")
            print(f"   Actions: {sample['actions'].shape}")
            print(f"   Episode ID: {sample['episode_id']}")
            print()

        # Test DataLoader
        print("🔄 Testing DataLoader...")
        task_path = "/home/najo/NAS/VLA/dataset/New_dataset2/Green_point"

        dataloader = create_dataloader(
            dataset_paths=[task_path],
            batch_size=2,
            num_workers=0,
            shuffle=False,
        )

        batch = next(iter(dataloader))
        print(f"   Batch size: {len(batch['instruction'])}")
        print(f"   Sensor data: {batch['sensor_data'].shape}")
        print(f"   Robot states: {batch['robot_states'].shape}")
        print(f"   Actions: {batch['actions'].shape}")
        print()

        print("✅ All tests passed!")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
