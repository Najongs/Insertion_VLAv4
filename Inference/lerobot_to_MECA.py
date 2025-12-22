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
MODEL_ID = "lerobot/smolvla_base"  # Base model architecture
CHECKPOINT_PATH = "/home/najo/NAS/VLA/Insertion_VLAv4/Train/outputs/train/smolvla_vla_insertion_multigpu/checkpoints/checkpoint_step_0016000.pt"

# Target Configuration (choose one: Blue point, Green point, Red point, White point, Yellow point)
TARGET_COLOR = "Red point"  # Change this to target different colored insertion points

# Camera Configuration
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 30
CAMERA_MANUAL_FOCUS = 105

# Control Loop Configuration
CONTROL_FREQUENCY = 5  # Hz
MAX_EPISODES = 100
MAX_STEPS_PER_EPISODE = 1000

# Action Scaling (safety limits)
ACTION_SCALE_XYZ = 0.5  # mm per step
ACTION_SCALE_ROT = 1.0  # degrees per step

# Robot Motion Timeouts (seconds)
MOVE_TIMEOUT = 180
IDLE_TIMEOUT = 30
ERROR_RECOVERY_DELAY = 0.5

# Performance Profiling
ENABLE_PROFILING = False


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
        self.robot.SetCartLinVel(100)
        self.robot.SetJointVel(1)
        self.robot.SetBlending(50)
        self.robot.SetAutoConf(True)  # Auto-select best configuration

        self.robot.WaitIdle(IDLE_TIMEOUT)

        # Move to home position
        self.logger.info('Moving to initial home position [0, 0, 0, 0, 0, 0]...')
        self.move_angle_points([[0, 0, 0, 0, 0, 0]])
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
        task_name: Target name (e.g., "Red point", "Blue point")

    Returns:
        Formatted instruction string
    """
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
        # Convert to tensor and normalize: (N, H, W, C) -> (N, C, H, W)
        stacked_frames = np.stack(frame_list, axis=0)
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

            # Load trained checkpoint instead of base model
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

            logger.info(colored("Setup complete!", "green"))
            logger.info(f"Target: {TARGET_COLOR}")
            logger.info(f"Robot type: {ROBOT_TYPE}")
            logger.info(f"Control frequency: {CONTROL_FREQUENCY} Hz")
            logger.info(f"Action scale: XYZ={ACTION_SCALE_XYZ}mm, Rot={ACTION_SCALE_ROT}deg")

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

            while episode_count < MAX_EPISODES:
                logger.info(colored(f"\n=== Episode {episode_count + 1}/{MAX_EPISODES} ===", "cyan"))

                # Reset policy for new episode
                policy.reset()

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

                    if action_np.ndim == 2:
                        action_np = action_np[0]

                    action_scaled = action_np.copy()
                    action_scaled[:3] *= ACTION_SCALE_XYZ
                    action_scaled[3:] *= ACTION_SCALE_ROT

                    # Execute action
                    robot_manager.move_EE_single(action_scaled)
                    total_steps += 1

                    # Logging
                    if ENABLE_PROFILING and total_steps % 10 == 0:
                        total_time = time.time() - step_start
                        logger.info(
                            f"Step {total_steps}: "
                            f"camera={t_camera*1000:.1f}ms, "
                            f"obs_build={t_obs_build*1000:.1f}ms, "
                            f"preprocess={t_preprocess*1000:.1f}ms, "
                            f"inference={t_inference*1000:.1f}ms, "
                            f"postprocess={t_postprocess*1000:.1f}ms, "
                            f"total={total_time*1000:.1f}ms"
                        )
                    elif not ENABLE_PROFILING and total_steps % 50 == 0:
                        logger.info(f"Step {total_steps}: action={action_scaled.round(3)}")

                    # Maintain control frequency
                    elapsed = time.time() - step_start
                    sleep_time = (1.0 / CONTROL_FREQUENCY) - elapsed
                    if sleep_time > 0:
                        time.sleep(sleep_time)
                    else:
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
        logger.info(colored("Shutdown complete", "green"))


if __name__ == "__main__":
    main()
