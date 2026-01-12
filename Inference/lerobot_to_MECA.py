#!/usr/bin/env python
"""
Real-time VLA inference with OAK cameras and MECA500 robot control.

This script:
1. Captures frames from multiple OAK cameras
2. Runs SmolVLA model inference
3. Directly controls MECA500 robot
"""
import cv2
import depthai as dai
import contextlib
import numpy as np
import torch
import time
import logging
import pathlib
import sys
from typing import Dict, List, Optional
from termcolor import colored

import mecademicpy.robot as mdr
import mecademicpy.robot_initializer as initializer
import mecademicpy.tools as tools

from lerobot.policies.factory import make_pre_post_processors
from lerobot.utils.utils import get_safe_torch_device, init_logging

init_logging()
logger = logging.getLogger(__name__)

# =========================
# Configuration
# =========================

# Robot Configuration
ROBOT_ADDRESS = "192.168.0.100"
ROBOT_TYPE = "meca500"

# Model Configuration
MODEL_ID = "lerobot/smolvla_base"  # Base model architecture (for preprocessor)
# Use trained checkpoint (PRIMARY) - always uses latest checkpoint
CHECKPOINT_PATH = "/home/irom/NAS/VLA/Insertion_VLAv4/Inference/checkpoints/checkpoints/checkpoint_latest.pt"
# Use downloaded Hugging Face model directly (FALLBACK)
MODEL_PATH = "/home/irom/NAS/VLA/Insertion_VLAv4/sub_tasks/downloads/model"
USE_HF_MODEL = False  # Set to True to use Hugging Face model, False to use checkpoint (default: checkpoint)

# Target Configuration (choose one: Blue point, Green point, Red point, White point, Yellow point)
TARGET_COLOR = "red point"  # Change this to target different colored insertion points (lowercase to match training)

# Camera Configuration
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 30
CAMERA_MANUAL_FOCUS = 101

# Control Loop Configuration
CONTROL_FREQUENCY = 8  # Hz (realistic for VLA inference, was 15 but too fast)
MAX_EPISODES = 100
MAX_STEPS_PER_EPISODE = 1000

# Action Scaling (safety limits)
# NOTE: Model outputs are already in the same range as collected data (scaled by joystick)
# So we should NOT scale again! Set to 1.0 to use model output directly.
ACTION_SCALE_XYZ = 1.0  # No additional scaling (model output already scaled)
ACTION_SCALE_ROT = 1.0  # No additional scaling (model output already scaled)

# Time Correction: Data was collected at 15Hz, but inference runs at 8Hz
# Model learned actions for 66.6ms steps, but we apply them for 125ms steps
# DEPRECATED: Time scaling amplifies noise. Use temporal interpolation instead.
# So we need to INCREASE actions proportionally: 125ms / 66.6ms = 1.875x
ENABLE_TIME_SCALE = False  # Set to True to use legacy time scaling (NOT recommended)
TIME_SCALE = 15.0 / 8.0 if ENABLE_TIME_SCALE else 1.0  # = 1.875 (data_collection_hz / inference_hz)

# Action Normalization Configuration
# IMPORTANT: If you know your dataset statistics, set these to the actual min/max values
# from your training data. Otherwise, leave at None to use model output directly.
# You can extract these using: python scripts/extract_dataset_stats.py
DATA_ACTION_MIN = None  # Example: np.array([-2.0, -2.0, -1.0, -5.0, -5.0, -5.0])
DATA_ACTION_MAX = None  # Example: np.array([2.0, 2.0, 1.0, 5.0, 5.0, 5.0])

# Action Smoothing Configuration
ENABLE_EMA_FILTER = True  # Enable Exponential Moving Average filter
EMA_ALPHA = 0.2  # EMA smoothing factor (0.1-0.3 recommended, lower = smoother)
ENABLE_TEMPORAL_ENSEMBLE = False  # Enable temporal ensemble (action chunking)
TEMPORAL_ENSEMBLE_WEIGHTS = [0.5, 0.3, 0.2]  # Weights for temporal ensemble (most recent first)

# Robot Motion Parameters
ROBOT_BLENDING = 100  # Blending factor (0-100, higher = smoother trajectory transitions)
ROBOT_CART_LIN_VEL = 100  # Cartesian linear velocity (mm/s)
ROBOT_JOINT_VEL = 1  # Joint velocity (%)
ENABLE_ACC_CTRL = True  # Enable acceleration control for smoother motion

# Robot Motion Timeouts (seconds)
MOVE_TIMEOUT = 180
IDLE_TIMEOUT = 30
ERROR_RECOVERY_DELAY = 0.5

# Performance Profiling
ENABLE_PROFILING = True  # Show detailed timing for camera/inference/etc

# Visualization
SHOW_CAMERA_PREVIEW = True  # Show camera frames in real-time


class ActionSmoother:
    """Applies temporal filtering to smooth robot actions.

    Implements:
    1. Exponential Moving Average (EMA) for high-frequency noise filtering
    2. Temporal Ensemble for action chunking
    """

    def __init__(self, action_dim: int = 6, ema_alpha: float = 0.2,
                 temporal_ensemble_weights: Optional[List[float]] = None):
        """
        Args:
            action_dim: Dimension of action space (default: 6 for 6-DOF)
            ema_alpha: EMA smoothing factor (0.0-1.0). Lower = smoother, higher = more responsive
            temporal_ensemble_weights: Weights for temporal ensemble (most recent first)
        """
        self.action_dim = action_dim
        self.ema_alpha = ema_alpha
        self.ema_state = None  # Stores running average

        # Temporal ensemble: stores action chunks from past inferences
        # Each entry is a list of actions predicted at that timestep
        self.action_chunks_history = []  # List of [chunk_idx, action_chunk]
        self.temporal_ensemble_weights = temporal_ensemble_weights or [0.5, 0.3, 0.2]
        self.chunk_step = 0  # Current step within chunk

    def reset(self):
        """Reset filter state for new episode."""
        self.ema_state = None
        self.action_chunks_history = []
        self.chunk_step = 0

    def apply_ema(self, action: np.ndarray) -> np.ndarray:
        """Apply Exponential Moving Average filter.

        Args:
            action: Raw action from policy (6-DOF)

        Returns:
            Smoothed action
        """
        if self.ema_state is None:
            # Initialize with first action
            self.ema_state = action.copy()
            return action

        # EMA formula: S_t = α * x_t + (1 - α) * S_{t-1}
        self.ema_state = self.ema_alpha * action + (1.0 - self.ema_alpha) * self.ema_state
        return self.ema_state.copy()

    def add_action_chunk(self, action_chunk: np.ndarray):
        """Store a new action chunk from policy prediction.

        Args:
            action_chunk: Array of shape (chunk_size, action_dim) or (action_dim,)
        """
        if action_chunk.ndim == 1:
            # Single action, treat as chunk of size 1
            action_chunk = action_chunk.reshape(1, -1)

        self.action_chunks_history.append(action_chunk)

        # Keep only recent chunks (window size = len(weights))
        max_history = len(self.temporal_ensemble_weights)
        if len(self.action_chunks_history) > max_history:
            self.action_chunks_history.pop(0)

    def get_temporal_ensemble_action(self, current_step: int) -> Optional[np.ndarray]:
        """Get action at current_step using temporal ensemble.

        For each past prediction, we look at what it predicted for current_step,
        then compute a weighted average with more recent predictions having higher weight.

        Args:
            current_step: Global step counter

        Returns:
            Ensembled action, or None if not enough history
        """
        if len(self.action_chunks_history) == 0:
            return None

        actions_for_current_step = []
        weights_for_current_step = []

        # Look back through history (most recent first)
        for i, chunk in enumerate(reversed(self.action_chunks_history)):
            # How many steps ago was this prediction made?
            steps_ago = i

            # What index in that chunk corresponds to current_step?
            chunk_size = len(chunk)
            chunk_start_step = current_step - steps_ago
            index_in_chunk = current_step - chunk_start_step

            if 0 <= index_in_chunk < chunk_size:
                # This chunk has a prediction for current_step
                actions_for_current_step.append(chunk[index_in_chunk])

                # Assign weight (more recent = higher weight)
                if i < len(self.temporal_ensemble_weights):
                    weights_for_current_step.append(self.temporal_ensemble_weights[i])
                else:
                    # Beyond defined weights, use small weight
                    weights_for_current_step.append(0.1)

        if len(actions_for_current_step) == 0:
            return None

        # Normalize weights
        actions_array = np.array(actions_for_current_step)
        weights_array = np.array(weights_for_current_step)
        weights_array = weights_array / weights_array.sum()

        # Weighted average
        ensembled_action = np.sum(actions_array * weights_array[:, np.newaxis], axis=0)
        return ensembled_action

    def smooth(self, action: np.ndarray, enable_ema: bool = True,
               enable_temporal_ensemble: bool = False,
               action_chunk: Optional[np.ndarray] = None,
               current_step: int = 0) -> np.ndarray:
        """Apply smoothing filters to action.

        Args:
            action: Raw action from policy (current timestep action)
            enable_ema: Whether to apply EMA filter
            enable_temporal_ensemble: Whether to apply temporal ensemble
            action_chunk: Full action chunk from policy (for temporal ensemble)
            current_step: Global step counter (for temporal ensemble)

        Returns:
            Smoothed action
        """
        smoothed_action = action.copy()

        # Apply temporal ensemble first (if enabled and chunk available)
        if enable_temporal_ensemble and action_chunk is not None:
            self.add_action_chunk(action_chunk)
            ensemble_action = self.get_temporal_ensemble_action(current_step)
            if ensemble_action is not None:
                smoothed_action = ensemble_action

        # Then apply EMA
        if enable_ema:
            smoothed_action = self.apply_ema(smoothed_action)

        return smoothed_action


class OAKCameraManager:
    """Manages multiple OAK cameras for simultaneous capture."""

    def __init__(self, width: int = 640, height: int = 480, fps: int = 30):
        self.width = width
        self.height = height
        self.fps = fps
        self.devices: List = []
        self.queues: List = []
        self.device_infos: List = []
        self.stack = contextlib.ExitStack()

    def initialize_cameras(self) -> int:
        """Initialize all available OAK cameras.

        Returns:
            Number of successfully initialized cameras.
        """
        device_infos = dai.Device.getAllAvailableDevices()
        logger.info(f"Found {len(device_infos)} OAK devices")

        if len(device_infos) == 0:
            raise RuntimeError("No OAK devices found")

        for i, info in enumerate(device_infos):
            logger.info(f"Initializing camera {i+1}/{len(device_infos)} (MxId: {info.mxid})")

            pipeline = dai.Pipeline()
            cam = pipeline.create(dai.node.ColorCamera)
            cam.setBoardSocket(dai.CameraBoardSocket.CAM_A)
            cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
            cam.setPreviewSize(self.width, self.height)
            cam.setInterleaved(False)
            cam.initialControl.setManualFocus(CAMERA_MANUAL_FOCUS)

            xout = pipeline.create(dai.node.XLinkOut)
            xout.setStreamName("rgb")
            cam.preview.link(xout.input)

            try:
                device = self.stack.enter_context(dai.Device(pipeline, info, dai.UsbSpeed.SUPER))
                q = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
                self.devices.append(device)
                self.queues.append(q)
                self.device_infos.append(info)
                logger.info(f"  Camera {i+1} initialized successfully")
            except Exception as e:
                logger.error(f"  Failed to initialize camera {i+1}: {e}")

        logger.info(f"Successfully initialized {len(self.queues)} cameras")

        # Wait for all cameras to be ready and producing frames
        logger.info("Waiting for all cameras to start streaming...")
        max_wait_iterations = 50  # ~5 seconds at 0.1s per iteration
        for iteration in range(max_wait_iterations):
            frames = self.get_frames()
            if len(frames) == len(self.queues):
                logger.info(f"All {len(self.queues)} cameras ready and streaming")
                break
            time.sleep(0.1)
        else:
            logger.warning(f"Timeout waiting for all cameras. Got {len(frames)}/{len(self.queues)} cameras")

        # Flush queues and wait for fresh frames
        # This ensures the next get_frames() call will have fresh frames ready
        logger.debug("Flushing camera queues...")
        for q in self.queues:
            while q.tryGet() is not None:
                pass  # Drain the queue
        time.sleep(0.15)  # Wait for new frames (at 30fps, frames come every ~33ms)

        return len(self.queues)

    def get_frames(self) -> Dict[str, np.ndarray]:
        """Get current frames from all cameras.

        Returns:
            Dictionary mapping camera names to RGB frames.
        """
        frames = {}
        for i, q in enumerate(self.queues):
            in_frame = q.tryGet()
            if in_frame is not None:
                frame = in_frame.getCvFrame()
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames[f"camera{i+1}"] = frame
        return frames

    def close(self):
        """Close all camera devices."""
        self.stack.close()
        logger.info("All cameras closed")

class RobotManager:
    """Manages MECA500 robot connection, setup, and control."""

    def __init__(self, address: str = "192.168.0.100"):
        self.address = address
        self.robot: Optional[mdr.Robot] = None
        self.logger = logging.getLogger(__name__)

    def __enter__(self):
        log_file = f'{pathlib.Path(__file__).stem}.log'
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger.info(f"Log file: {log_file}")

        self.robot = initializer.RobotWithTools()
        self.robot.__enter__()
        self.logger.info(f"Connecting to robot at {self.address}...")
        try:
            self.robot.Connect(address=self.address, disconnect_on_exception=False)
            self.logger.info("Robot connected")
        except mdr.MecademicException as e:
            self.logger.error(f"Failed to connect to robot: {e}")
            raise
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.robot and self.robot.IsConnected():
            self.logger.info("Disconnecting robot...")
            try:
                status = self.robot.GetStatusRobot()
                if status.error_status:
                    self.logger.warning('Robot in error state, attempting to reset...')
                    self.robot.ResetError()
                    time.sleep(ERROR_RECOVERY_DELAY)
                    self.robot.ResumeMotion()
                    time.sleep(ERROR_RECOVERY_DELAY)
            except Exception as e:
                self.logger.warning(f'Error check/clear failed during exit: {e}')

            try:
                self.logger.info("Deactivating robot...")
                self.robot.DeactivateRobot()
            except Exception as e:
                self.logger.warning(f'Deactivate failed during exit: {e}')
        if self.robot:
            self.robot.__exit__(exc_type, exc_value, traceback)
        self.logger.info("Robot disconnected")
        logging.shutdown()

    def _verify_robot_model(self):
        """Verify that the connected robot is a MECA500."""
        if not tools.robot_model_is_meca500(self.robot.GetRobotInfo().robot_model):
            raise mdr.MecademicException(
                f'Unsupported robot model: {self.robot.GetRobotInfo().robot_model}'
            )

    def setup(self):
        """Initialize and configure the robot for VLA control."""
        self.logger.info('Activating and homing robot...')
        self._verify_robot_model()

        self.robot.SetJointVel(3)
        initializer.reset_sim_mode(self.robot)
        initializer.reset_motion_queue(self.robot, activate_home=True)
        initializer.reset_vacuum_module(self.robot)
        self.robot.WaitHomed()

        # Configure for Cartesian movements
        self.robot.SetCartLinVel(ROBOT_CART_LIN_VEL)
        self.robot.SetJointVel(ROBOT_JOINT_VEL)
        self.robot.SetBlending(ROBOT_BLENDING)
        self.robot.SetAutoConf(True)  # Auto-select best configuration

        # Configure acceleration control for smoother motion
        if ENABLE_ACC_CTRL:
            try:
                # SetAccCtrl: Set acceleration control (percentage of maximum acceleration)
                # Lower values = smoother but slower acceleration
                self.robot.SetCartAcc(50)  # 50% of max cartesian acceleration
                self.robot.SetJointAcc(50)  # 50% of max joint acceleration
                self.logger.info("Acceleration control enabled (50% of max)")
            except AttributeError:
                self.logger.warning("SetAccCtrl not available in this robot firmware")

        self.robot.WaitIdle(IDLE_TIMEOUT)

        # Move to home position
        self.logger.info('Moving to initial home position [0, 0, 0, 0, 0, 0]...')
        self.move_angle_points([[0, -20, 20, 0, 30, 60]])
        self.logger.info('Robot at home position')
        self.logger.info('Robot setup complete')

    def move_angle_points(self, points: List[List[float]]):
        """Move robot through a sequence of joint angle positions.

        Args:
            points: List of 6-DOF joint angles in degrees.
        """
        self._verify_robot_model()

        for idx, angles in enumerate(points):
            self.logger.info(f'Moving to joint angles {idx+1}: {angles}')
            self.robot.MoveJoints(*angles)
            self.robot.WaitIdle(MOVE_TIMEOUT)

    def move_EE_points(self, points: List[List[float]]):
        """Move robot through a sequence of end-effector poses.

        Args:
            points: List of 6-DOF poses [x, y, z, alpha, beta, gamma].
        """
        self._verify_robot_model()

        for idx, pose in enumerate(points):
            self.logger.info(f'Moving to EE pose {idx+1}: {pose}')
            self.robot.MovePose(*pose)
            self.robot.WaitIdle(MOVE_TIMEOUT)

    def check_and_recover_error(self) -> bool:
        """Check for robot errors and attempt recovery.

        Returns:
            True if error was detected and recovery attempted, False otherwise.
        """
        try:
            status = self.robot.GetStatusRobot()
            if status.error_status:
                self.logger.warning('Robot in error state, attempting recovery...')
                self.robot.ResetError()
                time.sleep(ERROR_RECOVERY_DELAY)
                self.robot.ResumeMotion()
                time.sleep(ERROR_RECOVERY_DELAY)
                self.logger.info('Error recovery completed')
                return True
            return False
        except Exception as e:
            self.logger.error(f'Error recovery failed: {e}')
            return False

    def get_state(self) -> np.ndarray:
        """Get current robot state as 6-DOF pose.

        Returns:
            6-DOF pose [x, y, z, alpha, beta, gamma] in mm and degrees.
        """
        try:
            pose = self.robot.GetPose()
            return np.array(pose, dtype=np.float32)
        except Exception as e:
            self.logger.warning(f'Failed to get robot state: {e}')
            return np.zeros(6, dtype=np.float32)

    def move_EE_single(self, action: np.ndarray):
        """Send single action to robot (non-blocking, for real-time control).

        Args:
            action: 6-DOF delta [dx, dy, dz, da, db, dg] in mm and degrees.
        """
        try:
            self.check_and_recover_error()
            self._verify_robot_model()

            current_pose = self.robot.GetPose()
            target_pose = [current_pose[i] + action[i] for i in range(6)]

            # Non-blocking move command (no WaitIdle for real-time control)
            self.robot.MovePose(*target_pose)

            if self.logger.isEnabledFor(logging.DEBUG):
                self.logger.debug(
                    f'Sent EE pose: [{target_pose[0]:.2f}, {target_pose[1]:.2f}, '
                    f'{target_pose[2]:.2f}, {target_pose[3]:.2f}, {target_pose[4]:.2f}, '
                    f'{target_pose[5]:.2f}]'
                )

        except mdr.MecademicException as e:
            self.logger.error(f'Mecademic error sending action: {e}')
        except Exception as e:
            self.logger.error(f'Failed to send action: {e}')


def generate_instruction(task_name: str) -> str:
    """Generate task instruction matching training data format.

    Args:
        task_name: Target name (e.g., "red point", "blue point")

    Returns:
        Formatted instruction string (MUST match training config format)
    """
    # Simple instruction format matching train_config_new_dataset.yaml
    # Training used: "Insert needle into red point"
    return f"Insert needle into {task_name}"


def load_trained_checkpoint(checkpoint_path: str, device: torch.device):
    """Load trained policy checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file (.pt)
        device: Device to load model to

    Returns:
        Loaded policy model
    """
    logger.info(f"Loading trained checkpoint from: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Extract policy state dict
    policy_state_dict = checkpoint.get("policy_state_dict")
    if policy_state_dict is None:
        raise ValueError("Checkpoint does not contain 'policy_state_dict'")

    # Get config from checkpoint
    train_config = checkpoint.get("config", {})
    policy_cfg = train_config.get("policy", {})

    logger.info(f"Checkpoint info:")
    logger.info(f"  Step: {checkpoint.get('step', 'unknown')}")
    logger.info(f"  Epoch: {checkpoint.get('epoch', 'unknown')}")

    # Load base policy architecture
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    pretrained_model_id = policy_cfg.get("pretrained_model_id", MODEL_ID)

    logger.info(f"Loading base policy from: {pretrained_model_id}")
    policy = SmolVLAPolicy.from_pretrained(pretrained_model_id)

    # Update policy config to match training config
    logger.info("Updating policy config from training config...")
    policy.config.n_obs_steps = policy_cfg.get("n_obs_steps", 1)
    policy.config.chunk_size = policy_cfg.get("chunk_size", 1)
    policy.config.n_action_steps = policy_cfg.get("n_action_steps", 1)

    logger.info(f"  n_obs_steps: {policy.config.n_obs_steps}")
    logger.info(f"  chunk_size: {policy.config.chunk_size}")
    logger.info(f"  n_action_steps: {policy.config.n_action_steps}")

    # Handle DataParallel wrapped models
    if any(k.startswith("module.") for k in policy_state_dict.keys()):
        # Remove "module." prefix from DataParallel
        new_state_dict = {}
        for k, v in policy_state_dict.items():
            if k.startswith("module.model."):
                # DataParallel(DataParallelWrapper(SmolVLAPolicy))
                new_key = k.replace("module.model.", "")
                new_state_dict[new_key] = v
            elif k.startswith("module."):
                new_key = k.replace("module.", "")
                new_state_dict[new_key] = v
            else:
                new_state_dict[k] = v
        policy_state_dict = new_state_dict

    # Load state dict
    missing_keys, unexpected_keys = policy.load_state_dict(policy_state_dict, strict=False)

    if missing_keys:
        logger.warning(f"Missing keys: {missing_keys[:5]}...")
    if unexpected_keys:
        logger.warning(f"Unexpected keys: {unexpected_keys[:5]}...")

    logger.info(colored("Trained checkpoint loaded successfully!", "green"))

    # Move to device
    policy.to(device)
    policy.eval()

    return policy


def draw_action_graph(action_history: List[np.ndarray], width: int = 800, height: int = 400) -> np.ndarray:
    """Draw action history as a graph.

    Args:
        action_history: List of action arrays (6-DOF)
        width: Graph width in pixels
        height: Graph height in pixels

    Returns:
        Graph image as numpy array (BGR)
    """
    # Create blank image
    img = np.ones((height, width, 3), dtype=np.uint8) * 255

    if len(action_history) < 2:
        cv2.putText(img, "Collecting data...", (width//2 - 100, height//2),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (128, 128, 128), 2)
        return img

    # Convert to numpy array
    actions = np.array(action_history)  # Shape: (N, 6)
    num_steps = len(actions)

    # Define colors for each DOF (BGR format)
    colors_xyz = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]  # Blue, Green, Red
    colors_rot = [(255, 128, 0), (255, 0, 255), (128, 255, 0)]  # Cyan, Magenta, Yellow-green

    # Split into translation and rotation
    margin = 40
    graph_height = (height - 3 * margin) // 2

    # Draw XYZ graph (top half)
    xyz_data = actions[:, :3]
    xyz_min, xyz_max = xyz_data.min(), xyz_data.max()
    xyz_range = max(abs(xyz_min), abs(xyz_max), 0.1)  # Ensure non-zero range

    cv2.putText(img, "Translation (XYZ)", (10, 25),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    for dof in range(3):
        points = []
        for i in range(num_steps):
            x = int(margin + (i / max(num_steps - 1, 1)) * (width - 2 * margin))
            y_normalized = xyz_data[i, dof] / (2 * xyz_range)  # Normalize to [-0.5, 0.5]
            y = int(margin + graph_height // 2 - y_normalized * graph_height * 0.9)
            points.append((x, y))

        # Draw line
        for i in range(len(points) - 1):
            cv2.line(img, points[i], points[i+1], colors_xyz[dof], 2)

        # Draw legend
        legend_x = width - 150
        legend_y = margin + dof * 20
        cv2.line(img, (legend_x, legend_y), (legend_x + 30, legend_y), colors_xyz[dof], 2)
        cv2.putText(img, ["X", "Y", "Z"][dof], (legend_x + 35, legend_y + 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

    # Draw zero line for XYZ
    zero_y_xyz = margin + graph_height // 2
    cv2.line(img, (margin, zero_y_xyz), (width - margin, zero_y_xyz), (200, 200, 200), 1)

    # Draw Rotation graph (bottom half)
    rot_data = actions[:, 3:]
    rot_min, rot_max = rot_data.min(), rot_data.max()
    rot_range = max(abs(rot_min), abs(rot_max), 0.1)

    rot_top = margin * 2 + graph_height
    cv2.putText(img, "Rotation (Rx, Ry, Rz)", (10, rot_top + 25),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    for dof in range(3):
        points = []
        for i in range(num_steps):
            x = int(margin + (i / max(num_steps - 1, 1)) * (width - 2 * margin))
            y_normalized = rot_data[i, dof] / (2 * rot_range)
            y = int(rot_top + margin + graph_height // 2 - y_normalized * graph_height * 0.9)
            points.append((x, y))

        # Draw line
        for i in range(len(points) - 1):
            cv2.line(img, points[i], points[i+1], colors_rot[dof], 2)

        # Draw legend
        legend_x = width - 150
        legend_y = rot_top + margin + dof * 20
        cv2.line(img, (legend_x, legend_y), (legend_x + 30, legend_y), colors_rot[dof], 2)
        cv2.putText(img, ["Rx", "Ry", "Rz"][dof], (legend_x + 35, legend_y + 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

    # Draw zero line for rotation
    zero_y_rot = rot_top + margin + graph_height // 2
    cv2.line(img, (margin, zero_y_rot), (width - margin, zero_y_rot), (200, 200, 200), 1)

    # Draw current values
    if num_steps > 0:
        current = actions[-1]
        text = f"Current: XYZ=[{current[0]:+.2f}, {current[1]:+.2f}, {current[2]:+.2f}] "
        text += f"Rot=[{current[3]:+.2f}, {current[4]:+.2f}, {current[5]:+.2f}]"
        cv2.putText(img, text, (10, height - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)

    return img


def create_observation(
    frames: Dict[str, np.ndarray],
    robot_state: np.ndarray,
    num_cameras: int,
    device: torch.device,
    instruction: str
) -> Dict:
    """Create observation dictionary for VLA policy.

    Args:
        frames: Camera frames dictionary.
        robot_state: Current robot state (6-DOF pose).
        num_cameras: Number of cameras.
        device: Torch device.
        instruction: Task instruction string.

    Returns:
        Observation dictionary for policy inference.
    """
    observation = {}

    # Stack and process camera frames in batch
    frame_list = []
    cam_keys = []
    for i in range(1, num_cameras + 1):
        cam_key = f"camera{i}"
        if cam_key in frames:
            frame_list.append(frames[cam_key])
            cam_keys.append(cam_key)

    if len(frame_list) == num_cameras:
        # Resize to 512x512 (REQUIRED: model was trained with 512x512 images)
        import cv2
        resized_frames = []
        for frame in frame_list:
            resized = cv2.resize(frame, (512, 512), interpolation=cv2.INTER_LINEAR)
            resized_frames.append(resized)

        # Convert to tensor and normalize: (N, H, W, C) -> (N, C, H, W)
        stacked_frames = np.stack(resized_frames, axis=0)
        frame_tensor = torch.from_numpy(stacked_frames).float() / 255.0
        frame_tensor = frame_tensor.permute(0, 3, 1, 2).contiguous()
        frame_tensor = frame_tensor.to(device, non_blocking=True)

        # Assign to observation dict
        for i, cam_key in enumerate(cam_keys):
            observation[f"observation.images.{cam_key}"] = frame_tensor[i:i+1]

    # Add robot state
    state_tensor = torch.from_numpy(robot_state).float().to(device).unsqueeze(0)
    observation["observation.state"] = state_tensor

    # Add task and robot type
    observation["task"] = instruction
    observation["robot_type"] = ROBOT_TYPE

    return observation


def main():
    logger.info(colored("Starting LeRobot to MECA500 integration", "green", attrs=["bold"]))

    camera_manager = OAKCameraManager(width=CAMERA_WIDTH, height=CAMERA_HEIGHT, fps=CAMERA_FPS)

    try:
        with RobotManager(ROBOT_ADDRESS) as robot_manager:
            robot_manager.setup()

            # Initialize cameras
            num_cameras = camera_manager.initialize_cameras()
            logger.info(f"Using {num_cameras} cameras for observation")

            # Setup device and load model
            device = get_safe_torch_device("cuda" if torch.cuda.is_available() else "cpu")
            logger.info(f"Using device: {device}")

            # Load model
            if USE_HF_MODEL:
                logger.info(f"Loading Hugging Face model from: {MODEL_PATH}")
                from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
                from lerobot.configs.types import FeatureType, PolicyFeature

                policy = SmolVLAPolicy.from_pretrained(MODEL_PATH)

                # Manually configure input/output features for 3 cameras
                logger.info("Configuring input/output features...")
                policy.config.input_features = {
                    "observation.images.camera1": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 480, 640)),
                    "observation.images.camera2": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 480, 640)),
                    "observation.images.camera3": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 480, 640)),
                    "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(6,)),
                }
                policy.config.output_features = {
                    "action": PolicyFeature(type=FeatureType.ACTION, shape=(6,)),
                }

                policy.to(device)
                policy.eval()
                logger.info(colored("Hugging Face model loaded successfully!", "green"))
            else:
                logger.info(f"Loading trained checkpoint: {CHECKPOINT_PATH}")
                policy = load_trained_checkpoint(CHECKPOINT_PATH, device)

            # Generate task instruction (matching training data format)
            instruction = generate_instruction(TARGET_COLOR)
            logger.info(f"Task instruction generated for: {TARGET_COLOR}")

            # Create preprocessors and postprocessors
            logger.info("Creating preprocessors and postprocessors")

            if ENABLE_PROFILING:
                logger.info(f"Policy input features: {policy.config.input_features}")
                logger.info(f"Policy output features: {policy.config.output_features}")

            # Use base MODEL_ID for preprocessor (architecture identifier)
            preprocessor, postprocessor = make_pre_post_processors(
                policy.config,
                MODEL_ID,
                preprocessor_overrides={"device_processor": {"device": device.type}}
            )

            # Initialize action smoother
            action_smoother = ActionSmoother(
                action_dim=6,
                ema_alpha=EMA_ALPHA,
                temporal_ensemble_weights=TEMPORAL_ENSEMBLE_WEIGHTS
            )
            logger.info(f"Action smoother initialized:")
            logger.info(f"  EMA enabled: {ENABLE_EMA_FILTER}, alpha={EMA_ALPHA}")
            logger.info(f"  Temporal ensemble enabled: {ENABLE_TEMPORAL_ENSEMBLE}")
            if ENABLE_TEMPORAL_ENSEMBLE:
                logger.info(f"  Temporal ensemble weights: {TEMPORAL_ENSEMBLE_WEIGHTS}")

            logger.info(colored("Setup complete!", "green"))
            logger.info(f"Target: {TARGET_COLOR}")
            logger.info(f"Robot type: {ROBOT_TYPE}")
            logger.info(f"Control frequency: {CONTROL_FREQUENCY} Hz")
            logger.info(f"Action scale: XYZ={ACTION_SCALE_XYZ}mm, Rot={ACTION_SCALE_ROT}deg")
            if ENABLE_TIME_SCALE:
                logger.warning(f"Time scale: {TIME_SCALE}x (LEGACY MODE - NOT RECOMMENDED)")
            else:
                logger.info(f"Time scale: {TIME_SCALE}x (time scaling disabled - recommended)")
            if DATA_ACTION_MIN is not None and DATA_ACTION_MAX is not None:
                logger.info(f"Using dataset statistics for normalization")
            else:
                logger.info(f"Using model output directly (no dataset stats normalization)")
            logger.info(f"Robot motion: Blending={ROBOT_BLENDING}, CartLinVel={ROBOT_CART_LIN_VEL}mm/s, JointVel={ROBOT_JOINT_VEL}%")
            logger.info(f"Acceleration control: {'Enabled (50%)' if ENABLE_ACC_CTRL else 'Disabled'}")

            # Warmup inference
            logger.info("Warming up model...")
            warmup_success = False
            for warmup_attempt in range(10):
                warmup_frames = camera_manager.get_frames()
                if len(warmup_frames) >= num_cameras:
                    warmup_state = robot_manager.get_state()
                    warmup_obs = create_observation(warmup_frames, warmup_state, num_cameras, device, instruction)
                    with torch.inference_mode():
                        _ = policy.select_action(preprocessor(warmup_obs))
                    logger.info("Warmup complete")
                    warmup_success = True
                    break
                else:
                    logger.debug(f"Warmup attempt {warmup_attempt+1}: got {len(warmup_frames)}/{num_cameras} frames, retrying...")
                    time.sleep(0.1)

            if not warmup_success:
                logger.warning(f"Warmup incomplete: only got {len(warmup_frames)}/{num_cameras} cameras")

            logger.info("Press Ctrl+C to stop")

            episode_count = 0
            total_steps = 0

            # Action history for plotting (keep last 100 steps)
            action_history = []
            max_history = 100

            while episode_count < MAX_EPISODES:
                logger.info(colored(f"\n=== Episode {episode_count + 1}/{MAX_EPISODES} ===", "cyan"))

                # Reset policy and smoother for new episode
                policy.reset()
                action_smoother.reset()

                episode_start_time = time.time()

                for step in range(MAX_STEPS_PER_EPISODE):
                    step_start = time.time()

                    # Capture frames
                    if ENABLE_PROFILING:
                        t0 = time.time()
                    frames = camera_manager.get_frames()
                    if ENABLE_PROFILING:
                        t_camera = time.time() - t0

                    if len(frames) < num_cameras:
                        missing_cams = [f"camera{i}" for i in range(1, num_cameras+1) if f"camera{i}" not in frames]
                        logger.warning(f"Only got {len(frames)}/{num_cameras} frames (missing: {missing_cams}), skipping")
                        time.sleep(0.01)
                        continue

                    # Display camera preview (will be updated with action info later)
                    display_frames = []
                    if SHOW_CAMERA_PREVIEW:
                        for i in range(1, num_cameras + 1):
                            cam_key = f"camera{i}"
                            if cam_key in frames:
                                # Convert RGB to BGR for OpenCV display
                                frame_bgr = cv2.cvtColor(frames[cam_key], cv2.COLOR_RGB2BGR)
                                # Add camera label
                                cv2.putText(frame_bgr, cam_key, (10, 30),
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                                display_frames.append(frame_bgr)

                    # Create observation
                    if ENABLE_PROFILING:
                        t0 = time.time()
                    robot_state = robot_manager.get_state()
                    observation = create_observation(frames, robot_state, num_cameras, device, instruction)
                    if ENABLE_PROFILING:
                        t_obs_build = time.time() - t0

                    # Preprocess
                    if ENABLE_PROFILING:
                        t0 = time.time()
                    preprocessed_obs = preprocessor(observation)
                    if ENABLE_PROFILING:
                        t_preprocess = time.time() - t0

                    # Inference
                    if ENABLE_PROFILING:
                        t0 = time.time()
                    with torch.inference_mode():
                        action = policy.select_action(preprocessed_obs)
                    if ENABLE_PROFILING:
                        t_inference = time.time() - t0

                    # Postprocess
                    if ENABLE_PROFILING:
                        t0 = time.time()
                    action = postprocessor(action)
                    if ENABLE_PROFILING:
                        t_postprocess = time.time() - t0

                    # Extract and scale action
                    if isinstance(action, dict) and "action" in action:
                        action_np = action["action"].cpu().numpy()
                    elif isinstance(action, torch.Tensor):
                        action_np = action.cpu().numpy()
                    else:
                        logger.error(f"Unexpected action format: {type(action)}")
                        continue

                    # Handle action shape
                    # action_np can be:
                    # - (1, chunk_size, 6) - batched chunk
                    # - (chunk_size, 6) - unbatched chunk
                    # - (1, 6) - batched single action
                    # - (6,) - single action
                    if action_np.ndim == 3:
                        # Batched chunk: (1, chunk_size, 6) -> (chunk_size, 6)
                        action_chunk = action_np[0]
                        current_action = action_chunk[0]  # Use first action in chunk
                    elif action_np.ndim == 2:
                        if action_np.shape[0] == 1:
                            # Batched single action: (1, 6) -> (6,)
                            current_action = action_np[0]
                            action_chunk = current_action.reshape(1, -1)  # Treat as chunk of size 1
                        else:
                            # Unbatched chunk: (chunk_size, 6)
                            action_chunk = action_np
                            current_action = action_chunk[0]
                    else:
                        # Single action: (6,)
                        current_action = action_np
                        action_chunk = current_action.reshape(1, -1)

                    # Apply action and time scaling to entire chunk
                    action_chunk_scaled = action_chunk.copy()
                    action_chunk_scaled[:, :3] *= ACTION_SCALE_XYZ * TIME_SCALE
                    action_chunk_scaled[:, 3:] *= ACTION_SCALE_ROT * TIME_SCALE

                    # Scale current action
                    action_scaled = current_action.copy()
                    action_scaled[:3] *= ACTION_SCALE_XYZ * TIME_SCALE
                    action_scaled[3:] *= ACTION_SCALE_ROT * TIME_SCALE

                    # Apply action smoothing
                    action_smoothed = action_smoother.smooth(
                        action_scaled,
                        enable_ema=ENABLE_EMA_FILTER,
                        enable_temporal_ensemble=ENABLE_TEMPORAL_ENSEMBLE,
                        action_chunk=action_chunk_scaled if ENABLE_TEMPORAL_ENSEMBLE else None,
                        current_step=total_steps
                    )

                    # Execute action
                    robot_manager.move_EE_single(action_smoothed)
                    total_steps += 1

                    # Update camera preview with action info
                    if SHOW_CAMERA_PREVIEW and len(display_frames) == num_cameras:
                        for frame_bgr in display_frames:
                            xyz_text = f"XYZ: [{action_smoothed[0]:+.2f}, {action_smoothed[1]:+.2f}, {action_smoothed[2]:+.2f}]"
                            rot_text = f"Rot: [{action_smoothed[3]:+.2f}, {action_smoothed[4]:+.2f}, {action_smoothed[5]:+.2f}]"
                            cv2.putText(frame_bgr, xyz_text, (10, 60),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
                            cv2.putText(frame_bgr, rot_text, (10, 85),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 200, 255), 1)
                            # Add smoothing indicators
                            if ENABLE_EMA_FILTER or ENABLE_TEMPORAL_ENSEMBLE:
                                smoothing_text = "Smoothing: "
                                if ENABLE_EMA_FILTER:
                                    smoothing_text += f"EMA({EMA_ALPHA})"
                                if ENABLE_TEMPORAL_ENSEMBLE:
                                    if ENABLE_EMA_FILTER:
                                        smoothing_text += " + "
                                    smoothing_text += f"TE({len(TEMPORAL_ENSEMBLE_WEIGHTS)})"
                                cv2.putText(frame_bgr, smoothing_text, (10, 110),
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

                        # Stack and display frames
                        combined = np.hstack(display_frames)
                        if combined.shape[1] > 1920:
                            scale = 1920 / combined.shape[1]
                            new_width = int(combined.shape[1] * scale)
                            new_height = int(combined.shape[0] * scale)
                            combined = cv2.resize(combined, (new_width, new_height))
                        cv2.imshow("VLA Inference - Camera Views", combined)
                        cv2.waitKey(1)

                    # Update action history for graph (use smoothed action)
                    action_history.append(action_smoothed.copy())
                    if len(action_history) > max_history:
                        action_history.pop(0)

                    # Display action graph
                    if SHOW_CAMERA_PREVIEW and total_steps % 5 == 0:  # Update graph every 5 steps
                        graph_img = draw_action_graph(action_history)
                        cv2.imshow("Action History", graph_img)
                        cv2.waitKey(1)

                    # Logging
                    if ENABLE_PROFILING and total_steps % 10 == 0:
                        total_time = time.time() - step_start
                        fps = 1.0 / total_time if total_time > 0 else 0
                        logger.info(
                            f"Step {total_steps}: "
                            f"cam={t_camera*1000:.0f}ms | "
                            f"obs={t_obs_build*1000:.0f}ms | "
                            f"pre={t_preprocess*1000:.0f}ms | "
                            f"inf={t_inference*1000:.0f}ms | "
                            f"post={t_postprocess*1000:.0f}ms | "
                            f"total={total_time*1000:.0f}ms ({fps:.1f} FPS) | "
                            f"xyz=[{action_smoothed[0]:+.2f},{action_smoothed[1]:+.2f},{action_smoothed[2]:+.2f}] "
                            f"rot=[{action_smoothed[3]:+.2f},{action_smoothed[4]:+.2f},{action_smoothed[5]:+.2f}]"
                        )
                    elif not ENABLE_PROFILING and total_steps % 50 == 0:
                        logger.info(f"Step {total_steps}: action={action_smoothed.round(3)}")

                    # Maintain control frequency
                    elapsed = time.time() - step_start
                    sleep_time = (1.0 / CONTROL_FREQUENCY) - elapsed
                    if sleep_time > 0:
                        time.sleep(sleep_time)
                    else:
                        # Only warn if significantly over target (>20% slower)
                        if elapsed > (1.0 / CONTROL_FREQUENCY) * 1.2 and total_steps % 50 == 0:
                            logger.warning(f"Step took {elapsed:.3f}s (target: {1.0/CONTROL_FREQUENCY:.3f}s)")

                episode_duration = time.time() - episode_start_time
                episode_count += 1
                logger.info(colored(f"Episode {episode_count} completed in {episode_duration:.1f}s", "green"))

            logger.info(colored(f"\nAll {MAX_EPISODES} episodes completed!", "green", attrs=["bold"]))

    except KeyboardInterrupt:
        logger.info(colored("\nCtrl+C detected, shutting down...", "yellow"))

    except Exception as e:
        logger.error(colored(f"Error occurred: {e}", "red"), exc_info=True)

    finally:
        logger.info("Cleaning up...")
        camera_manager.close()
        if SHOW_CAMERA_PREVIEW:
            cv2.destroyAllWindows()
        logger.info(colored("Shutdown complete", "green"))


if __name__ == "__main__":
    main()
