#!/usr/bin/env python
"""
Real-time VLA Data Collection with 6-DoF Gamepad & Multi-Camera View
- Control: Xbox/PS Controller (Full 6-Axis + D-Pad Support)
- Logging: HDF5 (All Cameras)
- Features: Auto-Recovery, Multi-View, HOME Button, D-Pad Control
"""
import os
import sys
import time
import threading
import logging
import pathlib
import h5py
import numpy as np
from datetime import datetime
from termcolor import colored
import contextlib
import cv2
import queue

import pygame
import depthai as dai
import mecademicpy.robot as mdr
import mecademicpy.robot_initializer as initializer
from lerobot.utils.utils import init_logging

# --- Configuration ---
ROBOT_ADDRESS = "192.168.0.100"
DATASET_DIR = "collected_data"

CONTROL_FREQUENCY = 15
# 속도 설정
SCALE_POS = 0.8   # 스틱 이동 속도 (빠름)
SCALE_HAT = 0.3   # 방향키 이동 속도 (정밀 조작용)
SCALE_Z   = 0.5   # 삽입 속도
SCALE_ROT = 0.3   # 회전 속도
DEADZONE = 0.2    # 노이즈 제거

# 초기 위치 (Safe Start Pose)
HOME_JOINTS = (0, -20, 20, 0, 30, 60)

init_logging()
logger = logging.getLogger(__name__)

# ============================================================
# 1️⃣ Global Clock
# ============================================================
class GlobalClock(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.timestamp = round(time.time(), 3)
        self.running = True
        self.lock = threading.Lock()
    def now(self):
        with self.lock: return self.timestamp
    def run(self):
        while self.running:
            with self.lock: self.timestamp = round(time.time(), 3)
            time.sleep(0.005)
    def stop(self): self.running = False

# ============================================================
# 2️⃣ RtSampler (Real-time Robot State Sampler)
# ============================================================
class RtSampler(threading.Thread):
    def __init__(self, robot, clock, rate_hz=100):
        super().__init__(daemon=True)
        self.robot = robot
        self.dt = 1.0 / float(rate_hz)
        self.clock = clock
        self.stop_evt = threading.Event()
        self.lock = threading.Lock()
        self.latest_q = np.zeros(6)
        self.latest_p = np.zeros(6)
    def stop(self): self.stop_evt.set()
    def get_latest_data(self):
        with self.lock: return self.latest_q.copy(), self.latest_p.copy()
    def run(self):
        logger.info("🤖 Starting robot state sampler...")

        # 초기 로봇 상태 읽기
        initial_success = False
        for _ in range(10):  # 최대 10번 시도
            try:
                q = list(self.robot.GetJoints())
                p = list(self.robot.GetPose())
                if q and len(q) >= 6 and p and len(p) >= 6:
                    with self.lock:
                        self.latest_q = np.array(q[:6])
                        self.latest_p = np.array(p[:6])
                    logger.info(f"✅ Initial robot state acquired: q={self.latest_q.round(2)}")
                    initial_success = True
                    break
            except Exception as e:
                logger.debug(f"Initial read attempt failed: {e}")
            time.sleep(0.1)

        if not initial_success:
            logger.warning("⚠️ Failed to get initial robot state! Data will be all zeros until first successful read.")

        next_t = time.time()
        success_count = 0
        fail_count = 0

        while not self.stop_evt.is_set():
            ts_now = self.clock.now()
            q, p = None, None

            # 1. 관절 각도 (Joint Angles) 가져오기
            try:
                q = list(self.robot.GetJoints())
            except Exception as e:
                if fail_count == 0:  # 첫 실패만 로그
                    logger.debug(f"GetJoints failed: {e}")

            # 2. 말단 자세 (Cartesian Pose) 가져오기
            try:
                p = list(self.robot.GetPose())
            except Exception as e:
                if fail_count == 0:  # 첫 실패만 로그
                    logger.debug(f"GetPose failed: {e}")

            # 데이터가 유효하면 메모리에 업데이트
            if q and len(q) >= 6 and p and len(p) >= 6:
                with self.lock:
                    self.latest_q = np.array(q[:6])
                    self.latest_p = np.array(p[:6])
                success_count += 1
            else:
                fail_count += 1
                if fail_count % 100 == 1:  # 100번마다 한번씩만 경고
                    logger.warning(f"⚠️ Invalid robot data (fail count: {fail_count})")

            next_t += self.dt
            if next_t - time.time() > 0: time.sleep(next_t - time.time())

        logger.info(f"📊 RtSampler stats: {success_count} success, {fail_count} failures")

# ============================================================
# 3️⃣ Gamepad Controller (Added D-Pad Support & Multi-Mode)
# ============================================================
class GamepadController:
    def __init__(self):
        pygame.init()
        pygame.joystick.init()
        self.joystick = None
        self.control_mode = 1  # 컨트롤 모드: 1, 2, 3
        self.mode_switch_cooldown = 0  # 모드 전환 쿨다운
        self.smoothing_enabled = False  # 가속도 모드 (X 버튼으로 토글)
        self.smoothing_cooldown = 0  # 토글 쿨다운
        self.current_action = np.zeros(6)  # 현재 속도 (가속도 모드용)
        self.acceleration_rate = 0.2   # 가속률 (빠른 가속)
        self.deceleration_rate = 0.7   # 감속률 (빠른 멈춤)
        if pygame.joystick.get_count() > 0:
            self.joystick = pygame.joystick.Joystick(0)
            self.joystick.init()
            logger.info(f"🎮 Connected: {self.joystick.get_name()}")
            logger.info(f"🔧 Control Mode: {self.control_mode} (Press BACK/SELECT to change)")
        else:
            logger.error("❌ No Gamepad found!")

    def get_action(self):
        if not self.joystick: return np.zeros(6), False, False, False
        pygame.event.pump()

        # --- Mode Switching (BACK/SELECT button = button 6) ---
        btn_mode_switch = self.joystick.get_button(6) if self.joystick.get_numbuttons() > 6 else False
        if btn_mode_switch and self.mode_switch_cooldown == 0:
            self.control_mode = (self.control_mode % 3) + 1
            logger.info(colored(f"🔧 Switched to Control Mode {self.control_mode}", "yellow"))
            self.mode_switch_cooldown = 10  # 10 프레임 쿨다운
        if self.mode_switch_cooldown > 0:
            self.mode_switch_cooldown -= 1

        # --- Smoothing Toggle (X button = button 2) ---
        btn_smoothing = self.joystick.get_button(2) if self.joystick.get_numbuttons() > 2 else False
        if btn_smoothing and self.smoothing_cooldown == 0:
            self.smoothing_enabled = not self.smoothing_enabled
            status = "ON" if self.smoothing_enabled else "OFF"
            logger.info(colored(f"🌊 Smoothing Mode: {status}", "cyan"))
            self.smoothing_cooldown = 10  # 10 프레임 쿨다운
        if self.smoothing_cooldown > 0:
            self.smoothing_cooldown -= 1

        # --- 1. Analog Stick Inputs (Raw) ---
        # Left Stick: Move
        y_stick_raw = self.joystick.get_axis(1)
        x_stick_raw = -self.joystick.get_axis(0)

        # Right Stick
        rs_x_raw = self.joystick.get_axis(3)  # 좌우
        rs_y_raw = -self.joystick.get_axis(4) # 상하

        # Apply deadzone
        y_stick = y_stick_raw if abs(y_stick_raw) > DEADZONE else 0.0
        x_stick = x_stick_raw if abs(x_stick_raw) > DEADZONE else 0.0
        rs_x = rs_x_raw if abs(rs_x_raw) > DEADZONE else 0.0
        rs_y = rs_y_raw if abs(rs_y_raw) > DEADZONE else 0.0

        # [NEW] Axis-lock for precise cardinal movement (Left Stick only)
        # 한 축이 다른 축보다 1.5배 이상 크면, 작은 축은 0으로 (대각선 방지)
        if abs(y_stick) > abs(x_stick) * 1.5:
            x_stick = 0  # 상하 움직임만
        elif abs(x_stick) > abs(y_stick) * 1.5:
            y_stick = 0  # 좌우 움직임만

        # Apply scaling to movement
        y_stick *= SCALE_POS
        x_stick *= SCALE_POS

        # Triggers: Z-Axis (or rotation depending on mode)
        lt = (self.joystick.get_axis(2) + 1) / 2
        rt = (self.joystick.get_axis(5) + 1) / 2

        # Bumpers
        lb = self.joystick.get_button(4)
        rb = self.joystick.get_button(5)

        # --- 2. D-Pad (Hat) Inputs ---
        hat_x, hat_y = self.joystick.get_hat(0)
        y_hat = -hat_x * SCALE_HAT   # D-pad LEFT/RIGHT → Y축 (좌우)
        x_hat = hat_y * SCALE_HAT    # D-pad UP/DOWN → X축 (전후)

        # --- 3. Combine Movement ---
        y = y_stick + y_hat
        x = x_stick + x_hat

        # --- 4. Rotation Mapping (Mode-dependent) ---
        if self.control_mode == 1:
            # Mode 1: 기본 (Original)
            # RS: Pitch/Roll, Bumper: Yaw, Trigger: Z
            rx = rs_y * SCALE_ROT  # Pitch (상하)
            ry = rs_x * SCALE_ROT  # Roll (좌우)
            rz = (rb - lb) * SCALE_ROT * 1.5  # Yaw
            z = (rt - lt) * SCALE_Z

        elif self.control_mode == 2:
            # Mode 2: 바늘 삽입 최적화
            # RS: Pitch/Yaw, Bumper: Roll, Trigger: Z
            rx = rs_y * SCALE_ROT  # Pitch (상하)
            rz = rs_x * SCALE_ROT * 1.5  # Yaw (좌우) - 더 민감하게
            ry = (rb - lb) * SCALE_ROT  # Roll
            z = (rt - lt) * SCALE_Z

        else:  # self.control_mode == 3
            # Mode 3: 트리거 회전
            # RS 상하: Pitch only, Trigger: Yaw, Bumper: Roll
            rx = rs_y * SCALE_ROT  # Pitch (상하)
            ry = (rb - lb) * SCALE_ROT  # Roll
            rz = (rt - lt) * SCALE_ROT * 2.0  # Yaw (트리거)
            z = 0  # Z축은 다른 방법으로 제어해야 함 (이 모드의 단점)

        target_action = np.array([y, x, z, rx, ry, rz])

        # 가속도 모드가 활성화되어 있으면 smoothing 적용
        if self.smoothing_enabled:
            if np.linalg.norm(target_action) < 0.01:
                # 조이스틱을 거의 안 움직이면 → 빠르게 감속
                self.current_action += (target_action - self.current_action) * self.deceleration_rate
            else:
                # 조이스틱을 움직이면 → 천천히 가속
                self.current_action += (target_action - self.current_action) * self.acceleration_rate
            action = np.where(np.abs(self.current_action) < 0.001, 0, self.current_action)
        else:
            # 즉시 반응 모드
            action = target_action
            self.current_action = target_action  # 모드 전환 시를 위해 동기화

        # Buttons
        btn_rec = self.joystick.get_button(0) # A
        btn_disc = self.joystick.get_button(1) # B
        btn_home = self.joystick.get_button(3) # Y
        btn_exit = self.joystick.get_button(7) if self.joystick.get_numbuttons() > 7 else False # START

        return action, btn_rec, btn_disc, btn_home, btn_exit

# ============================================================
# 4️⃣ Camera & Recorder
# ============================================================
class OAKCameraManager:
    def __init__(self, width=640, height=480, fps=30):
        self.width, self.height = width, height
        self.stack = contextlib.ExitStack()
        self.queues = []
    def initialize_cameras(self):
        infos = dai.Device.getAllAvailableDevices()
        if not infos: raise RuntimeError("No OAK devices")
        for info in infos:
            p = dai.Pipeline()
            c = p.create(dai.node.ColorCamera)
            c.setBoardSocket(dai.CameraBoardSocket.CAM_A)
            c.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
            c.setPreviewSize(self.width, self.height)
            c.setInterleaved(False)

            # 카메라 ID가 19로 시작하면 수동 초점 설정
            camera_id = info.getMxId()
            if camera_id.startswith("19"):
                c.initialControl.setManualFocus(105)
                logger.info(f"📷 Camera {camera_id}: Manual focus set to 105")
            else:
                logger.info(f"📷 Camera {camera_id}: Auto focus")

            out = p.create(dai.node.XLinkOut)
            out.setStreamName("rgb")
            c.preview.link(out.input)
            d = self.stack.enter_context(dai.Device(p, info, dai.UsbSpeed.SUPER))
            self.queues.append(d.getOutputQueue("rgb", 4, False))
        return len(self.queues)
    def get_frames(self):
        frames = {}
        for i, q in enumerate(self.queues):
            f = q.tryGet()
            if f: frames[f"camera{i+1}"] = f.getCvFrame()
        return frames
    def close(self): self.stack.close()

class VLARecorder:
    def __init__(self, output_dir, clock):
        self.out = pathlib.Path(output_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        self.clock = clock
        self.buffer = []
        self.recording = False
        self.is_saving = False # 현재 저장 중인지 확인하는 플래그

    def start(self):
        if self.is_saving:
            logger.warning("⚠️ Still saving previous episode! Wait a moment.")
            return
        self.buffer = []
        self.recording = True
        logger.info("🎥 STARTED Recording")

    def add(self, frames, q, p, action):
        if not self.recording: return
        # 메모리 절약을 위해 여기서 변환하지 않고 저장할 때 변환할 수도 있지만,
        # CPU 부하 분산을 위해 수집 시 변환 유지 (단, 메모리 넉넉한 경우)
        rgb = {k: cv2.cvtColor(v, cv2.COLOR_BGR2RGB) for k, v in frames.items()}
        self.buffer.append({"ts": self.clock.now(), "imgs": rgb, "q": q, "p": p, "act": action})

    def save_async(self):
        """백그라운드 스레드에서 저장을 수행 (메인 루프 멈춤 방지)"""
        if not self.buffer: return
        
        # 1. 현재 버퍼를 임시 변수에 넘기고, 메인 버퍼는 즉시 비움 (다음 녹화 준비)
        buffer_snapshot = self.buffer
        self.buffer = []
        self.recording = False
        self.is_saving = True # 저장 시작 플래그 ON

        # 2. 저장 작업 정의 (별도 스레드에서 실행될 함수)
        def worker(data, filename):
            try:
                start_time = time.time()
                logger.info(f"💾 Saving {len(data)} steps to disk... (Background)")
                
                with h5py.File(filename, 'w') as f:
                    obs = f.create_group("observations")
                    img_grp = obs.create_group("images")
                    
                    # 첫 번째 프레임으로 키 확인
                    first = data[0]["imgs"]
                    
                    # 이미지 저장 (압축 방식 변경: gzip -> lzf)
                    # lzf는 압축률은 조금 낮지만 속도가 매우 빠름
                    for k in first.keys():
                        img_stack = np.stack([x["imgs"][k] for x in data])
                        img_grp.create_dataset(k, data=img_stack, compression="lzf")
                    
                    # 나머지 데이터 저장
                    obs.create_dataset("qpos", data=np.stack([x["q"] for x in data]))
                    obs.create_dataset("ee_pose", data=np.stack([x["p"] for x in data]))
                    f.create_dataset("action", data=np.stack([x["act"] for x in data]))
                    f.create_dataset("timestamp", data=np.stack([x["ts"] for x in data]))
                
                duration = time.time() - start_time
                logger.info(colored(f"✅ Save Complete: {filename} ({duration:.1f}s)", "green"))
            
            except Exception as e:
                logger.error(f"❌ Save Failed: {e}")
            
            finally:
                self.is_saving = False # 저장 완료 플래그 OFF

        # 3. 파일명 생성 및 스레드 시작
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = self.out / f"episode_{timestamp}.h5"
        
        t = threading.Thread(target=worker, args=(buffer_snapshot, fname))
        t.start()
        
        logger.info("⏳ Saving started in background... You can move the robot.")

    def discard(self):
        self.buffer = []
        self.recording = False
        logger.warning("🗑️ DISCARDED")

# ============================================================
# 5️⃣ Main
# ============================================================
def main():
    clock = GlobalClock()
    clock.start()
    gp = GamepadController()
    if not gp.joystick: return
    cam = OAKCameraManager()
    rec = VLARecorder(DATASET_DIR, clock)

    try:
        robot = mdr.Robot()
        logger.info(f"🔌 Connecting to robot at {ROBOT_ADDRESS}...")
        robot.Connect(address=ROBOT_ADDRESS)

        if not robot.IsConnected():
            logger.error(f"❌ Failed to connect to robot at {ROBOT_ADDRESS}")
            return

        logger.info("✅ Connected! Activating and homing...")
        robot.ActivateAndHome()
        robot.SetRealTimeMonitoring(1)

        logger.info("🏠 Moving to start pose...")
        robot.MoveJoints(*HOME_JOINTS)
        robot.WaitIdle()
        logger.info("✅ Ready!")

        sampler = RtSampler(robot, clock, 100)
        sampler.start()

        logger.info("📷 Initializing cameras...")
        num_cameras = cam.initialize_cameras()
        logger.info(f"✅ {num_cameras} camera(s) initialized")

        logger.info(colored("\n=== CONTROLS ===", "cyan"))
        logger.info(" [LS / D-Pad] Move X/Y (Axis-locked for precise movement)")
        logger.info(" [X]          Toggle Smoothing Mode (Acceleration ON/OFF)")
        logger.info(" [BACK/SELECT] Switch Control Mode (1/2/3)")
        logger.info(" [A/B/Y]      Rec / Discard / Home")
        logger.info(" [START]      Exit Program")
        logger.info(colored("\n=== CONTROL MODES ===", "cyan"))
        logger.info(" Mode 1 (Default):  RS=Pitch/Roll, LB/RB=Yaw, LT/RT=Z")
        logger.info(" Mode 2 (Needle):   RS=Pitch/Yaw,  LB/RB=Roll, LT/RT=Z")
        logger.info(" Mode 3 (Trigger):  RS=Pitch only, LB/RB=Roll, LT/RT=Yaw")

        while True:
            t0 = time.time()
            
            # 0. Safety Check
            try:
                if robot.GetStatusRobot().error_status:
                    logger.warning("⚠️ Error! Auto-Reset...")
                    robot.ResetError()
                    time.sleep(0.1); robot.ResumeMotion(); time.sleep(0.5)
                    continue
            except Exception as e:
                logger.debug(f"Safety check failed: {e}")

            # 1. Inputs
            frames = cam.get_frames()
            q, p = sampler.get_latest_data()
            act_raw, rec_btn, disc_btn, home_btn, exit_btn = gp.get_action()

            # --- EXIT BUTTON ---
            if exit_btn:
                logger.info(colored("🛑 EXIT button pressed. Shutting down...", "red"))
                break

            # --- HOME BUTTON ---
            if home_btn:
                logger.info(colored("🏠 GOING HOME...", "yellow"))
                try:
                    if robot.GetStatusRobot().error_status:
                        robot.ResetError(); robot.ResumeMotion()
                    robot.MoveJoints(*HOME_JOINTS)
                    robot.WaitIdle()
                except Exception as e: logger.error(f"Home Failed: {e}")
                continue 

            # 2. Move Robot
            if np.any(np.abs(act_raw) > 0.001):
                try:
                    robot.MoveLinRelTrf(*[float(x) for x in act_raw])
                except Exception as e:
                    logger.debug(f"Move command failed: {e}")

            # 3. Recorder
            if rec_btn:
                if not rec.recording: 
                    rec.start()
                    time.sleep(0.5)
                else: 
                    # [수정됨] 비동기 저장 호출
                    rec.save_async() 
                    time.sleep(0.5)
                    
            if disc_btn and rec.recording: rec.discard(); time.sleep(0.5)
            
            rec.add(frames, q, p, act_raw)

            # 4. GUI
            if frames:
                sorted_keys = sorted(frames.keys())
                img_list = []
                for key in sorted_keys:
                    img = frames[key].copy()
                    cv2.putText(img, key, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                    img_list.append(img)

                # Combine all camera views horizontally
                if img_list:
                    try:
                        # Ensure all images have the same height for hstack
                        if len(img_list) > 1:
                            heights = [img.shape[0] for img in img_list]
                            if len(set(heights)) > 1:
                                # Resize all to same height if different
                                target_h = min(heights)
                                resized = []
                                for img in img_list:
                                    if img.shape[0] != target_h:
                                        aspect = img.shape[1] / img.shape[0]
                                        target_w = int(target_h * aspect)
                                        resized.append(cv2.resize(img, (target_w, target_h)))
                                    else:
                                        resized.append(img)
                                combined_view = np.hstack(resized)
                            else:
                                combined_view = np.hstack(img_list)
                        else:
                            combined_view = img_list[0]

                        # Status text
                        if rec.recording:
                            txt = f"REC ({len(rec.buffer)})"
                            col = (0, 0, 255) # 빨강
                        elif rec.is_saving:
                            txt = "SAVING..." # 저장 중일 때 표시
                            col = (0, 255, 255) # 노랑
                        else:
                            txt = "IDLE"
                            col = (0, 255, 0) # 초록

                        cv2.putText(combined_view, txt, (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.2, col, 3)

                        # Control mode display
                        mode_names = {1: "Mode1: Default", 2: "Mode2: Needle", 3: "Mode3: Trigger"}
                        mode_txt = mode_names.get(gp.control_mode, "Unknown")
                        cv2.putText(combined_view, mode_txt, (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                        # Smoothing mode display
                        smooth_txt = "Smoothing: ON" if gp.smoothing_enabled else "Smoothing: OFF"
                        smooth_col = (0, 255, 255) if gp.smoothing_enabled else (128, 128, 128)
                        cv2.putText(combined_view, smooth_txt, (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.6, smooth_col, 2)

                        cv2.imshow("Multi-View Dashboard", combined_view)
                    except Exception as e:
                        logger.error(f"GUI display error: {e}")

            if cv2.waitKey(1) == ord('q'): break
            
            el = time.time() - t0
            if el < 1/CONTROL_FREQUENCY: time.sleep(1/CONTROL_FREQUENCY - el)

    except KeyboardInterrupt: logger.info("Stopped.")
    except Exception as e: logger.error(e, exc_info=True)
    finally:
        if 'sampler' in locals(): sampler.stop(); sampler.join()
        if 'robot' in locals() and robot.IsConnected(): robot.DeactivateRobot(); robot.Disconnect()
        cam.close(); clock.stop(); cv2.destroyAllWindows()

if __name__ == "__main__":
    main()