#!/usr/bin/env python
"""
Async VLA inference with OAK cameras and MECA500 robot control.

This script implements asynchronous inference following the LeRobot async_inference design:
1. Observation thread: Captures frames at high frequency (30Hz)
2. Inference thread: Generates action chunks asynchronously
3. Control thread: Executes actions at fixed frequency (15Hz)

Key improvements over synchronous version:
- Decoupled observation, inference, and control loops
- Action chunk-based execution (temporal consistency)
- Proper time synchronization with timestamps
- Latency tracking and compensation
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
import threading
from typing import Dict, List, Optional, Tuple
from collections import deque
from dataclasses import dataclass
from termcolor import colored
from queue import Queue, Empty

import mecademicpy.robot as mdr
import mecademicpy.robot_initializer as initializer
import mecademicpy.tools as tools

from lerobot.policies.factory import make_pre_post_processors
from lerobot.utils.utils import get_safe_torch_device, init_logging

# Add Train directory to path for normalization utils
train_dir = pathlib.Path(__file__).parent.parent / "Train"
sys.path.insert(0, str(train_dir))
from normalization_utils import Normalizer

init_logging()
logger = logging.getLogger(__name__)

# =========================
# Configuration
# =========================

# Robot Configuration
ROBOT_ADDRESS = "192.168.0.100"
ROBOT_TYPE = "meca500"

# Model Configuration
MODEL_ID = "lerobot/smolvla_base"
CHECKPOINT_PATH = "/home/irom/NAS/VLA/Insertion_VLAv4/Inference/checkpoints/checkpoints/checkpoint_latest.pt"

# Target Configuration
TARGET_COLOR = "red point"

# Camera Configuration
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 30
CAMERA_MANUAL_FOCUS = 101

# Async Control Configuration
OBSERVATION_FPS = 30  # Hz - how often to capture observations
CONTROL_FREQUENCY = 15  # Hz - how often to send commands to robot
INFERENCE_TRIGGER_FPS = 10  # Hz - how often to trigger new inference (can be lower than control)

# Action chunk configuration
# If policy outputs chunks, we'll use them for temporal consistency
USE_ACTION_CHUNKS = True  # Use all actions in chunk vs just first action
ACTION_CHUNK_OVERLAP = 0.5  # How much to overlap action chunks (0.5 = 50% overlap)

# Queue sizes
OBS_QUEUE_SIZE = 2  # Small queue - only keep latest observations
ACTION_QUEUE_SIZE = 10  # Larger queue - buffer action chunks

# Timeouts
OBS_QUEUE_TIMEOUT = 0.5  # seconds
ACTION_QUEUE_TIMEOUT = 0.1  # seconds

# Action Scaling
ACTION_SCALE_XYZ = 1.0
ACTION_SCALE_ROT = 1.0

# Action Smoothing
ENABLE_EMA_FILTER = True
EMA_ALPHA = 0.2

# Robot Motion Parameters
ROBOT_BLENDING = 100
ROBOT_CART_LIN_VEL = 100
ROBOT_JOINT_VEL = 1
ENABLE_ACC_CTRL = True

# Timeouts
MOVE_TIMEOUT = 180
IDLE_TIMEOUT = 30
ERROR_RECOVERY_DELAY = 0.5

# Visualization
SHOW_CAMERA_PREVIEW = True
ENABLE_PROFILING = True

# Episode Configuration
MAX_EPISODES = 100
MAX_STEPS_PER_EPISODE = 1000


@dataclass
class TimedObservation:
    """Observation with timestamp."""
    timestamp: float
    timestep: int
    frames: Dict[str, np.ndarray]
    robot_state: np.ndarray


@dataclass
class TimedAction:
    """Action with timestamp."""
    timestamp: float
    timestep: int
    action: np.ndarray
    chunk_id: int  # Which inference generated this action
    chunk_index: int  # Index within the chunk


class ObservationQueue:
    """Thread-safe queue for observations with filtering."""

    def __init__(self, maxsize: int = 2):
        self.queue = Queue(maxsize=maxsize)
        self.last_timestep = -1
        self.lock = threading.Lock()

    def put(self, obs: TimedObservation, skip_duplicates: bool = True):
        """Add observation to queue, optionally filtering duplicates."""
        with self.lock:
            if skip_duplicates and obs.timestep <= self.last_timestep:
                logger.debug(f"Skipping duplicate observation #{obs.timestep}")
                return False

            # If queue is full, remove oldest
            if self.queue.full():
                try:
                    old_obs = self.queue.get_nowait()
                    logger.debug(f"Queue full, removed observation #{old_obs.timestep}")
                except Empty:
                    pass

            self.queue.put(obs)
            self.last_timestep = obs.timestep
            return True

    def get(self, timeout: float = None) -> Optional[TimedObservation]:
        """Get observation from queue."""
        try:
            return self.queue.get(timeout=timeout)
        except Empty:
            return None

    def clear(self):
        """Clear the queue."""
        with self.lock:
            while not self.queue.empty():
                try:
                    self.queue.get_nowait()
                except Empty:
                    break
            self.last_timestep = -1


class ActionBuffer:
    """Buffer for action chunks with time-based lookup."""

    def __init__(self, max_chunks: int = 10):
        self.buffer = deque(maxlen=max_chunks)
        self.lock = threading.Lock()
        self.current_chunk_id = 0

    def add_chunk(self, actions: List[TimedAction]):
        """Add action chunk to buffer."""
        with self.lock:
            self.buffer.append(actions)
            logger.debug(f"Added action chunk {actions[0].chunk_id} with {len(actions)} actions")

    def get_action_at_time(self, target_time: float) -> Optional[TimedAction]:
        """Get action closest to target time."""
        with self.lock:
            if not self.buffer:
                return None

            # Flatten all actions from all chunks
            all_actions = []
            for chunk in self.buffer:
                all_actions.extend(chunk)

            if not all_actions:
                return None

            # Find action closest to target time (but not in the future)
            valid_actions = [a for a in all_actions if a.timestamp <= target_time]

            if not valid_actions:
                # All actions are in the future, use earliest one
                return min(all_actions, key=lambda a: a.timestamp)

            # Use action closest to target time
            return min(valid_actions, key=lambda a: abs(a.timestamp - target_time))

    def get_next_action_id(self) -> int:
        """Get next chunk ID."""
        with self.lock:
            chunk_id = self.current_chunk_id
            self.current_chunk_id += 1
            return chunk_id

    def clear(self):
        """Clear buffer."""
        with self.lock:
            self.buffer.clear()
            self.current_chunk_id = 0


class ActionSmoother:
    """EMA-based action smoothing."""

    def __init__(self, action_dim: int = 6, alpha: float = 0.2):
        self.action_dim = action_dim
        self.alpha = alpha
        self.ema_state = None

    def reset(self):
        """Reset filter state."""
        self.ema_state = None

    def smooth(self, action: np.ndarray) -> np.ndarray:
        """Apply EMA smoothing."""
        if self.ema_state is None:
            self.ema_state = action.copy()
            return action

        self.ema_state = self.alpha * action + (1.0 - self.alpha) * self.ema_state
        return self.ema_state.copy()


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
        """Initialize all available OAK cameras."""
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

        # Wait for cameras to be ready
        logger.info("Waiting for all cameras to start streaming...")
        max_wait_iterations = 50
        for iteration in range(max_wait_iterations):
            frames = self.get_frames()
            if len(frames) == len(self.queues):
                logger.info(f"All {len(self.queues)} cameras ready and streaming")
                break
            time.sleep(0.1)
        else:
            logger.warning(f"Timeout waiting for all cameras. Got {len(frames)}/{len(self.queues)} cameras")

        # Flush queues
        logger.debug("Flushing camera queues...")
        for q in self.queues:
            while q.tryGet() is not None:
                pass
        time.sleep(0.15)

        return len(self.queues)

    def get_frames(self) -> Dict[str, np.ndarray]:
        """Get current frames from all cameras."""
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
    """Manages MECA500 robot connection and control."""

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
        """Initialize and configure the robot."""
        self.logger.info('Activating and homing robot...')
        self._verify_robot_model()

        self.robot.SetJointVel(3)
        initializer.reset_sim_mode(self.robot)
        initializer.reset_motion_queue(self.robot, activate_home=True)
        initializer.reset_vacuum_module(self.robot)
        self.robot.WaitHomed()

        self.robot.SetCartLinVel(ROBOT_CART_LIN_VEL)
        self.robot.SetJointVel(ROBOT_JOINT_VEL)
        self.robot.SetBlending(ROBOT_BLENDING)
        self.robot.SetAutoConf(True)

        if ENABLE_ACC_CTRL:
            try:
                self.robot.SetCartAcc(50)
                self.robot.SetJointAcc(50)
                self.logger.info("Acceleration control enabled (50% of max)")
            except AttributeError:
                self.logger.warning("SetAccCtrl not available in this robot firmware")

        self.robot.WaitIdle(IDLE_TIMEOUT)

        self.logger.info('Moving to initial home position [0, -20, 20, 0, 30, 60]...')
        self.robot.MoveJoints(0, -20, 20, 0, 30, 60)
        self.robot.WaitIdle(MOVE_TIMEOUT)
        self.logger.info('Robot at home position')
        self.logger.info('Robot setup complete')

    def check_and_recover_error(self) -> bool:
        """Check for robot errors and attempt recovery."""
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
        """Get current robot state as 6-DOF pose."""
        try:
            pose = self.robot.GetPose()
            return np.array(pose, dtype=np.float32)
        except Exception as e:
            self.logger.warning(f'Failed to get robot state: {e}')
            return np.zeros(6, dtype=np.float32)

    def move_EE_single(self, action: np.ndarray):
        """Send single action to robot (non-blocking)."""
        try:
            self.check_and_recover_error()
            self._verify_robot_model()

            current_pose = self.robot.GetPose()
            target_pose = [current_pose[i] + action[i] for i in range(6)]

            self.robot.MovePose(*target_pose)

        except mdr.MecademicException as e:
            self.logger.error(f'Mecademic error sending action: {e}')
        except Exception as e:
            self.logger.error(f'Failed to send action: {e}')


def load_trained_checkpoint(checkpoint_path: str, device: torch.device):
    """Load trained policy checkpoint."""
    logger.info(f"Loading trained checkpoint from: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    policy_state_dict = checkpoint.get("policy_state_dict")
    if policy_state_dict is None:
        raise ValueError("Checkpoint does not contain 'policy_state_dict'")

    train_config = checkpoint.get("config", {})
    policy_cfg = train_config.get("policy", {})

    logger.info(f"Checkpoint info:")
    logger.info(f"  Step: {checkpoint.get('step', 'unknown')}")
    logger.info(f"  Epoch: {checkpoint.get('epoch', 'unknown')}")

    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    pretrained_model_id = policy_cfg.get("pretrained_model_id", MODEL_ID)

    logger.info(f"Loading base policy from: {pretrained_model_id}")
    policy = SmolVLAPolicy.from_pretrained(pretrained_model_id)

    logger.info("Updating policy config from training config...")
    policy.config.n_obs_steps = policy_cfg.get("n_obs_steps", 1)
    policy.config.chunk_size = policy_cfg.get("chunk_size", 1)
    policy.config.n_action_steps = policy_cfg.get("n_action_steps", 1)

    logger.info(f"  n_obs_steps: {policy.config.n_obs_steps}")
    logger.info(f"  chunk_size: {policy.config.chunk_size}")
    logger.info(f"  n_action_steps: {policy.config.n_action_steps}")

    # Handle DataParallel
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

    missing_keys, unexpected_keys = policy.load_state_dict(policy_state_dict, strict=False)

    if missing_keys:
        logger.warning(f"Missing keys: {missing_keys[:5]}...")
    if unexpected_keys:
        logger.warning(f"Unexpected keys: {unexpected_keys[:5]}...")

    logger.info(colored("Trained checkpoint loaded successfully!", "green"))

    # Load normalization stats
    normalizer = None
    if "normalization_stats" in checkpoint:
        logger.info("Loading normalization statistics from checkpoint...")
        stats = checkpoint["normalization_stats"]
        normalizer = Normalizer(stats).to(device)
        logger.info("Normalizer created successfully")
    else:
        logger.warning("No normalization stats found in checkpoint")

    policy.to(device)
    policy.eval()

    return policy, normalizer


def create_observation(
    frames: Dict[str, np.ndarray],
    robot_state: np.ndarray,
    num_cameras: int,
    device: torch.device,
    instruction: str,
    normalizer: Optional[Normalizer] = None
) -> Dict:
    """Create observation dictionary for VLA policy."""
    observation = {}

    # Process camera frames
    frame_list = []
    cam_keys = []
    for i in range(1, num_cameras + 1):
        cam_key = f"camera{i}"
        if cam_key in frames:
            frame_list.append(frames[cam_key])
            cam_keys.append(cam_key)

    if len(frame_list) == num_cameras:
        resized_frames = []
        for frame in frame_list:
            resized = cv2.resize(frame, (512, 512), interpolation=cv2.INTER_LINEAR)
            resized_frames.append(resized)

        stacked_frames = np.stack(resized_frames, axis=0)
        frame_tensor = torch.from_numpy(stacked_frames).float() / 255.0
        frame_tensor = frame_tensor.permute(0, 3, 1, 2).contiguous()
        frame_tensor = frame_tensor.to(device, non_blocking=True)

        for i, cam_key in enumerate(cam_keys):
            observation[f"observation.images.{cam_key}"] = frame_tensor[i:i+1]

    # Add robot state
    state_tensor = torch.from_numpy(robot_state).float().to(device).unsqueeze(0)

    if normalizer is not None:
        state_tensor = normalizer.normalize(state_tensor, 'observation.state')

    observation["observation.state"] = state_tensor
    observation["task"] = instruction
    observation["robot_type"] = ROBOT_TYPE

    return observation


class AsyncVLAController:
    """Async VLA controller with separate observation, inference, and control threads."""

    def __init__(
        self,
        policy,
        normalizer: Optional[Normalizer],
        preprocessor,
        postprocessor,
        camera_manager: OAKCameraManager,
        robot_manager: RobotManager,
        num_cameras: int,
        device: torch.device,
        instruction: str
    ):
        self.policy = policy
        self.normalizer = normalizer
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor
        self.camera_manager = camera_manager
        self.robot_manager = robot_manager
        self.num_cameras = num_cameras
        self.device = device
        self.instruction = instruction

        # Queues and buffers
        self.obs_queue = ObservationQueue(maxsize=OBS_QUEUE_SIZE)
        self.action_buffer = ActionBuffer(max_chunks=ACTION_QUEUE_SIZE)

        # Action smoother
        self.action_smoother = ActionSmoother(action_dim=6, alpha=EMA_ALPHA)

        # Thread control
        self.shutdown_event = threading.Event()
        self.threads = []

        # Timesteps
        self.obs_timestep = 0
        self.control_timestep = 0

        # Timing stats
        self.stats = {
            'obs_fps': [],
            'inference_time': [],
            'control_fps': []
        }

    def start(self):
        """Start all threads."""
        logger.info("Starting async VLA controller threads...")

        # Start observation thread
        obs_thread = threading.Thread(target=self._observation_loop, name="ObservationThread", daemon=True)
        obs_thread.start()
        self.threads.append(obs_thread)

        # Start inference thread
        inf_thread = threading.Thread(target=self._inference_loop, name="InferenceThread", daemon=True)
        inf_thread.start()
        self.threads.append(inf_thread)

        # Start control thread
        ctrl_thread = threading.Thread(target=self._control_loop, name="ControlThread", daemon=True)
        ctrl_thread.start()
        self.threads.append(ctrl_thread)

        logger.info("All threads started")

    def stop(self):
        """Stop all threads."""
        logger.info("Stopping async VLA controller...")
        self.shutdown_event.set()

        for thread in self.threads:
            thread.join(timeout=2.0)

        logger.info("All threads stopped")

    def reset(self):
        """Reset for new episode."""
        self.obs_queue.clear()
        self.action_buffer.clear()
        self.action_smoother.reset()
        self.policy.reset()
        self.obs_timestep = 0
        self.control_timestep = 0

    def _observation_loop(self):
        """Observation capture thread - runs at OBSERVATION_FPS."""
        logger.info(f"Observation thread started (target FPS: {OBSERVATION_FPS})")
        dt = 1.0 / OBSERVATION_FPS

        while not self.shutdown_event.is_set():
            loop_start = time.time()

            # Capture frames
            frames = self.camera_manager.get_frames()

            if len(frames) == self.num_cameras:
                # Get robot state
                robot_state = self.robot_manager.get_state()

                # Create timed observation
                obs = TimedObservation(
                    timestamp=time.time(),
                    timestep=self.obs_timestep,
                    frames=frames,
                    robot_state=robot_state
                )

                # Add to queue (will skip if duplicate or queue full)
                self.obs_queue.put(obs)
                self.obs_timestep += 1

            # Maintain loop rate
            elapsed = time.time() - loop_start
            sleep_time = dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    def _inference_loop(self):
        """Inference thread - generates action chunks asynchronously."""
        logger.info(f"Inference thread started (trigger FPS: {INFERENCE_TRIGGER_FPS})")
        dt = 1.0 / INFERENCE_TRIGGER_FPS

        while not self.shutdown_event.is_set():
            loop_start = time.time()

            # Get observation from queue
            obs = self.obs_queue.get(timeout=OBS_QUEUE_TIMEOUT)

            if obs is not None:
                inference_start = time.time()

                # Create observation dict
                observation = create_observation(
                    obs.frames,
                    obs.robot_state,
                    self.num_cameras,
                    self.device,
                    self.instruction,
                    self.normalizer
                )

                # Preprocess
                preprocessed_obs = self.preprocessor(observation)

                # Inference
                with torch.inference_mode():
                    action = self.policy.select_action(preprocessed_obs)

                # Postprocess
                action = self.postprocessor(action)

                # Extract action tensor
                if isinstance(action, dict) and "action" in action:
                    action_tensor = action["action"]
                elif isinstance(action, torch.Tensor):
                    action_tensor = action
                else:
                    logger.error(f"Unexpected action format: {type(action)}")
                    continue

                # Unnormalize if needed
                if self.normalizer is not None:
                    if action_tensor.ndim == 3:
                        batch_size, chunk_size, action_dim = action_tensor.shape
                        action_flat = action_tensor.reshape(-1, action_dim)
                        action_unnorm = self.normalizer.unnormalize(action_flat, 'action')
                        action_tensor = action_unnorm.reshape(batch_size, chunk_size, action_dim)
                    elif action_tensor.ndim == 2:
                        action_tensor = self.normalizer.unnormalize(action_tensor, 'action')
                    else:
                        action_tensor = self.normalizer.unnormalize(action_tensor, 'action')

                # Convert to numpy
                action_np = action_tensor.cpu().numpy()

                # Handle action shape
                if action_np.ndim == 3:
                    action_chunk = action_np[0]  # (chunk_size, 6)
                elif action_np.ndim == 2:
                    if action_np.shape[0] == 1:
                        action_chunk = action_np  # (1, 6)
                    else:
                        action_chunk = action_np  # (chunk_size, 6)
                else:
                    action_chunk = action_np.reshape(1, -1)  # (6,) -> (1, 6)

                # Create timed actions with timestamps
                chunk_id = self.action_buffer.get_next_action_id()
                timed_actions = []

                base_time = obs.timestamp
                action_dt = 1.0 / CONTROL_FREQUENCY  # Time between actions

                for i, act in enumerate(action_chunk):
                    # Scale actions
                    act_scaled = act.copy()
                    act_scaled[:3] *= ACTION_SCALE_XYZ
                    act_scaled[3:] *= ACTION_SCALE_ROT

                    timed_action = TimedAction(
                        timestamp=base_time + i * action_dt,
                        timestep=obs.timestep + i,
                        action=act_scaled,
                        chunk_id=chunk_id,
                        chunk_index=i
                    )
                    timed_actions.append(timed_action)

                # Add to action buffer
                self.action_buffer.add_chunk(timed_actions)

                inference_time = time.time() - inference_start
                self.stats['inference_time'].append(inference_time)

                logger.info(
                    f"Inference complete: obs #{obs.timestep} -> "
                    f"{len(timed_actions)} actions, {inference_time*1000:.1f}ms"
                )

            # Maintain loop rate
            elapsed = time.time() - loop_start
            sleep_time = dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    def _control_loop(self):
        """Control thread - sends actions to robot at CONTROL_FREQUENCY."""
        logger.info(f"Control thread started (frequency: {CONTROL_FREQUENCY} Hz)")
        dt = 1.0 / CONTROL_FREQUENCY

        last_action_chunk_id = -1

        while not self.shutdown_event.is_set():
            loop_start = time.time()
            current_time = time.time()

            # Get action at current time
            timed_action = self.action_buffer.get_action_at_time(current_time)

            if timed_action is not None:
                action = timed_action.action

                # Apply EMA smoothing
                if ENABLE_EMA_FILTER:
                    action = self.action_smoother.smooth(action)

                # Send to robot
                self.robot_manager.move_EE_single(action)

                # Log when we start using a new action chunk
                if timed_action.chunk_id != last_action_chunk_id:
                    logger.info(
                        f"Using action chunk {timed_action.chunk_id} "
                        f"(obs #{timed_action.timestep - timed_action.chunk_index})"
                    )
                    last_action_chunk_id = timed_action.chunk_id

                if self.control_timestep % 50 == 0:
                    latency = (current_time - timed_action.timestamp) * 1000
                    logger.debug(
                        f"Control step {self.control_timestep}: "
                        f"chunk {timed_action.chunk_id}[{timed_action.chunk_index}], "
                        f"latency={latency:.1f}ms, "
                        f"action={action.round(3)}"
                    )

                self.control_timestep += 1
            else:
                if self.control_timestep % 50 == 0:
                    logger.warning(f"No action available at t={current_time:.3f}")

            # Maintain loop rate
            elapsed = time.time() - loop_start
            sleep_time = dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)


def main():
    logger.info(colored("Starting Async LeRobot to MECA500 integration", "green", attrs=["bold"]))

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
            logger.info(f"Loading trained checkpoint: {CHECKPOINT_PATH}")
            policy, normalizer = load_trained_checkpoint(CHECKPOINT_PATH, device)

            # Generate task instruction
            instruction = f"Insert needle into {TARGET_COLOR}"
            logger.info(f"Task instruction: {instruction}")

            # Create preprocessors and postprocessors
            logger.info("Creating preprocessors and postprocessors")
            preprocessor, postprocessor = make_pre_post_processors(
                policy.config,
                MODEL_ID,
                preprocessor_overrides={"device_processor": {"device": device.type}}
            )

            # Create async controller
            controller = AsyncVLAController(
                policy=policy,
                normalizer=normalizer,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                camera_manager=camera_manager,
                robot_manager=robot_manager,
                num_cameras=num_cameras,
                device=device,
                instruction=instruction
            )

            logger.info(colored("Setup complete!", "green"))
            logger.info(f"Target: {TARGET_COLOR}")
            logger.info(f"Observation FPS: {OBSERVATION_FPS}")
            logger.info(f"Inference trigger FPS: {INFERENCE_TRIGGER_FPS}")
            logger.info(f"Control frequency: {CONTROL_FREQUENCY} Hz")
            logger.info(f"Action chunks: {'Enabled' if USE_ACTION_CHUNKS else 'Disabled'}")
            logger.info(f"EMA smoothing: {'Enabled' if ENABLE_EMA_FILTER else 'Disabled'} (alpha={EMA_ALPHA})")
            logger.info("Press Ctrl+C to stop")

            # Start controller
            controller.start()

            # Warmup
            logger.info("Warming up (5 seconds)...")
            time.sleep(5)

            episode_count = 0

            while episode_count < MAX_EPISODES:
                logger.info(colored(f"\n=== Episode {episode_count + 1}/{MAX_EPISODES} ===", "cyan"))

                # Reset for new episode
                controller.reset()

                episode_start_time = time.time()

                # Run episode
                for step in range(MAX_STEPS_PER_EPISODE):
                    time.sleep(1.0)  # Just wait - threads are running in background

                    if step % 10 == 0:
                        logger.info(f"Episode {episode_count + 1}, step {step}/{MAX_STEPS_PER_EPISODE}")

                episode_duration = time.time() - episode_start_time
                episode_count += 1
                logger.info(colored(f"Episode {episode_count} completed in {episode_duration:.1f}s", "green"))

            logger.info(colored(f"\nAll {MAX_EPISODES} episodes completed!", "green", attrs=["bold"]))

            # Stop controller
            controller.stop()

    except KeyboardInterrupt:
        logger.info(colored("\nCtrl+C detected, shutting down...", "yellow"))

    except Exception as e:
        logger.error(colored(f"Error occurred: {e}", "red"), exc_info=True)

    finally:
        logger.info("Cleaning up...")
        camera_manager.close()
        logger.info(colored("Shutdown complete", "green"))


if __name__ == "__main__":
    main()
