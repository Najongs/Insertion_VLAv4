import mujoco
import numpy as np
import cv2
import time
import os
import h5py
import datetime
import threading
import pathlib
from collections import deque

# Progress Bar를 위한 라이브러리 (없으면 print로 대체)
try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, total=None: x

# === Configuration ===
MODEL_PATH = "meca_scene22.xml"
SAVE_DIR = "collected_data_sim"
MAX_EPISODES = 1   # 수집할 총 에피소드 수
CONTROL_FREQ = 20   # 제어 주기 (Hz)
SAVE_FPS = 15       # 데이터 저장 프레임 (Hz)

# Image Settings
IMG_WIDTH = 640
IMG_HEIGHT = 480

# Task Parameters
TARGET_INSERTION_DEPTH = 0.0275
TRAJ_DURATION = 15.0
COAXIAL_TOLERANCE = 50e-6

# === Recorder Class (Headless) ===
class SimRecorder:
    def __init__(self, output_dir):
        self.out = pathlib.Path(output_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        self.buffer = []
        self.recording = False
        self.is_saving = False

    def start(self):
        if self.is_saving: return
        self.buffer = []
        self.recording = True

    def add(self, frames, qpos, ee_pose, action, timestamp):
        if not self.recording: return
        self.buffer.append({
            "ts": timestamp,
            "imgs": frames,
            "q": qpos,
            "p": ee_pose,
            "act": action
        })

    def save_async(self):
        if not self.buffer: return
        data_snapshot = self.buffer
        self.buffer = []
        self.recording = False
        self.is_saving = True

        def worker(data, filename):
            try:
                with h5py.File(filename, 'w') as f:
                    obs = f.create_group("observations")
                    img_grp = obs.create_group("images")

                    q_data = np.array([x['q'] for x in data], dtype=np.float32)
                    p_data = np.array([x['p'] for x in data], dtype=np.float32)
                    act_data = np.array([x['act'] for x in data], dtype=np.float32)
                    ts_data = np.array([x['ts'] for x in data], dtype=np.float32)

                    obs.create_dataset("qpos", data=q_data, compression="gzip")
                    obs.create_dataset("ee_pose", data=p_data, compression="gzip")
                    f.create_dataset("action", data=act_data, compression="gzip")
                    f.create_dataset("timestamp", data=ts_data, compression="gzip")

                    first_imgs = data[0]["imgs"]
                    for cam_name in first_imgs.keys():
                        jpeg_list = []
                        for step in data:
                            img = step["imgs"][cam_name]
                            success, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 95])
                            if success: jpeg_list.append(buf.flatten())
                            else: jpeg_list.append(np.zeros(1, dtype=np.uint8))

                        dt = h5py.special_dtype(vlen=np.dtype('uint8'))
                        dset = img_grp.create_dataset(cam_name, (len(jpeg_list),), dtype=dt)
                        for i, code in enumerate(jpeg_list): dset[i] = code

                # print(f"[Background] Saved: {filename.name}") # 너무 시끄러우면 주석 처리

            except Exception as e:
                print(f"❌ Save Failed: {e}")
            finally:
                self.is_saving = False

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = self.out / f"episode_{timestamp}.h5"
        t = threading.Thread(target=worker, args=(data_snapshot, fname))
        t.start()

    def discard(self):
        self.buffer = []
        self.recording = False

# === Helper Functions ===
def smooth_step(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3 - 2 * t)

def randomize_phantom_pos(model, data, phantom_id, rot_id):
    # ID가 유효하지 않으면 리턴
    if phantom_id == -1 or rot_id == -1: 
        return
    offset_x = np.random.uniform(-0.1, 0.1)
    offset_y = np.random.uniform(-0.05, 0.03)
    offset_z = 0.0 
    new_pos = np.array([offset_x, offset_y, offset_z])
    model.body_pos[phantom_id] = new_pos
    
    # 2. 회전 랜덤화 (Rotating Body)
    random_angle_deg = np.random.uniform(-30, 30)
    new_quat = np.zeros(4)
    mujoco.mju_euler2Quat(new_quat, [0, 0, np.deg2rad(random_angle_deg)], "xyz")
    model.body_quat[rot_id] = new_quat
    # print(f">>> Randomize: Pos=({offset_x:.2f}, {offset_y:.2f}), Angle={random_angle_deg:.1f} deg")
    mujoco.mj_forward(model, data)

# === Main Script (Headless) ===
def main():
    print(f"🔄 Loading Model: {MODEL_PATH}")
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)
    
    # Offscreen Renderer
    renderer = mujoco.Renderer(model, height=IMG_HEIGHT, width=IMG_WIDTH)
    
    # IDs setup
    try:
        tip_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "needle_tip")
        back_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "needle_back")
        target_entry_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "trocar_target")
        target_depth_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "trocar_depth")
        phantom_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "phantom_assembly") # 이동용
        rotating_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "rotating_assembly")   # 회전용)
        n_motors = model.nu
        dof = model.nv
    except:
        print("⚠️ Warning: Some IDs not found.")
        phantom_body_id = -1

    recorder = SimRecorder(SAVE_DIR)
    
    # [Viewer 버전과 동일한 파라미터]
    home_pose = np.array([0.5, 0.0, 0.0, 0.0, -0.5, 0.0])
    damping = 1e-3
    current_speed = 0.5 

    print(f"🚀 Starting Headless Collection for {MAX_EPISODES} Episodes...")
    print(f"📁 Output: {os.path.abspath(SAVE_DIR)}\n")

    pbar = tqdm(total=MAX_EPISODES, desc="Collecting", unit="ep")

    episode_count = 0
    while episode_count < MAX_EPISODES:
        # --- Reset ---
        mujoco.mj_resetData(model, data)
        data.qpos[:6] = home_pose
        randomize_phantom_pos(model, data, phantom_body_id, rotating_id)
        mujoco.mj_forward(model, data)
        
        # State Init
        task_state = 1
        traj_start_time = data.time
        insertion_started = False
        accumulated_depth = 0.0
        align_timer = 0
        traj_initialized = False # 추가
        
        # Targets
        p_entry = data.site_xpos[target_entry_id].copy()
        p_depth = data.site_xpos[target_depth_id].copy()
        start_tip = data.site_xpos[tip_id].copy()
        start_back = data.site_xpos[back_id].copy()
        
        needle_len = np.linalg.norm(start_tip - start_back)
        
        recorder.start()
        step_count = 0
        success = False
        
        # --- Step Loop (Headless with Viewer Logic) ---
        while True:
            t_curr = data.time
            curr_tip = data.site_xpos[tip_id].copy()
            curr_back = data.site_xpos[back_id].copy()
            
            # --- 1. Trajectory Planning (Viewer 버전 로직) ---
            if task_state == 1: # Align
                if not traj_initialized:
                    traj_start_time = t_curr
                    start_tip_pos = curr_tip.copy()
                    start_back_pos = curr_back.copy()
                    traj_initialized = True

                elapsed = t_curr - traj_start_time
                progress = smooth_step(elapsed / TRAJ_DURATION)
                
                # 목표 지점 계산
                axis_vec = p_depth - p_entry
                axis_dir = axis_vec / (np.linalg.norm(axis_vec) + 1e-10)
                
                goal_tip = p_entry - (axis_dir * 0.0001)
                goal_back = goal_tip - (axis_dir * needle_len)
                
                target_tip_pos = (1 - progress) * start_tip_pos + progress * goal_tip
                target_back_pos = (1 - progress) * start_back_pos + progress * goal_back
                
                if progress >= 1.0:
                    dist_error = np.linalg.norm(curr_tip - goal_tip)
                    if dist_error < 0.002: align_timer += 1
                    else: align_timer = 0
                    
                    if align_timer > 20:
                        task_state = 2
                        insertion_started = False

            elif task_state == 2: # Insert
                if not insertion_started:
                    # [중요] 현재 위치 기준 삽입 (튀는 현상 방지)
                    phase3_base_tip = curr_tip.copy()
                    insertion_started = True
                    accumulated_depth = 0.0
                
                step_z = 0.0000025
                accumulated_depth += step_z
                
                axis_vec = p_depth - p_entry
                axis_dir = axis_vec / (np.linalg.norm(axis_vec) + 1e-10)
                
                target_tip_pos = phase3_base_tip + (axis_dir * accumulated_depth)
                target_back_pos = target_tip_pos - (axis_dir * needle_len)
                
                if accumulated_depth >= TARGET_INSERTION_DEPTH:
                    task_state = 3
                    hold_start_time = data.time

            elif task_state == 3:
                # 1. 위치 고정 (State 2의 마지막 목표 위치를 계속 유지)
                # (axis_dir 등이 루프 안에서 계산되고 있으므로 그대로 사용 가능)
                axis_vec = p_depth - p_entry
                axis_dir = axis_vec / (np.linalg.norm(axis_vec) + 1e-10)
                
                # 목표를 "현재 위치"가 아니라 "최종 삽입 깊이"로 고정해야 흔들리지 않습니다.
                target_tip_pos = phase3_base_tip + (axis_dir * TARGET_INSERTION_DEPTH)
                target_back_pos = target_tip_pos - (axis_dir * needle_len)
                
                # 2. 시간 체크 (1초 대기)
                if data.time - hold_start_time >= 1.0:
                    success = True
                    break

            # --- 2. Stacked IK Solver (Viewer 버전 로직) ---
            # Position Errors
            err_tip = target_tip_pos - curr_tip
            err_back = target_back_pos - curr_back
            
            # Roll Correction
            tip_rot_mat = data.site_xmat[tip_id].reshape(3, 3)
            current_side_vec = tip_rot_mat @ np.array([1, 0, 0])
            needle_axis_curr = (curr_tip - curr_back) / (np.linalg.norm(curr_tip - curr_back) + 1e-10)
            target_side_vec = np.cross(needle_axis_curr, np.array([0, 0, 1]))
            if np.linalg.norm(target_side_vec) < 1e-3: target_side_vec = np.array([1, 0, 0])
            else: target_side_vec /= np.linalg.norm(target_side_vec)
            err_roll = np.cross(current_side_vec, target_side_vec)
            
            # Jacobian
            jac_tip_full = np.zeros((6, dof))
            jac_back = np.zeros((3, dof))
            mujoco.mj_jacSite(model, data, jac_tip_full[:3], jac_tip_full[3:], tip_id)
            mujoco.mj_jacSite(model, data, jac_back, None, back_id)
            
            # Stacked Calculation
            J_p1 = jac_tip_full[:3, :n_motors] # Tip Pos
            e_p1 = err_tip * 50.0   # Tip Error Gain
            
            if np.linalg.norm(e_p1) > 1.0: e_p1 = e_p1 / np.linalg.norm(e_p1) * 1.0
                
            J_p1_pinv = np.linalg.pinv(J_p1, rcond=1e-4)
            dq_p1 = J_p1_pinv @ e_p1
            P_null_1 = np.eye(n_motors) - (J_p1_pinv @ J_p1)
            
            J_p2 = jac_back[:, :n_motors] # Back Pos
            e_p2 = err_back * 50.0
            J_p2_proj = J_p2 @ P_null_1
            J_p2_pinv = np.linalg.pinv(J_p2_proj, rcond=1e-4)
            dq_p2 = J_p2_pinv @ (e_p2 - J_p2 @ dq_p1)
            P_null_2 = P_null_1 - (J_p2_pinv @ J_p2_proj)
            
            J_p3 = jac_tip_full[3:, :n_motors] # Roll
            e_p3 = err_roll * 10.0
            J_p3_proj = J_p3 @ P_null_2
            J_p3_pinv = np.linalg.pinv(J_p3_proj, rcond=1e-4)
            dq_p3 = J_p3_pinv @ (e_p3 - J_p3 @ (dq_p1 + dq_p2))
            
            dq = dq_p1 + dq_p2 + dq_p3
            
            # Apply Control
            current_action = data.qpos[:n_motors] + dq * current_speed
            data.ctrl[:n_motors] = current_action
            
            # --- 3. Physics Step ---
            mujoco.mj_step(model, data)
            step_count += 1
            
            # --- 4. Data Save (15Hz) ---
            if step_count % 67 == 0: 
                frames = {}
                for cam_name in ["side_camera", "tool_camera", "top_camera"]:
                    renderer.update_scene(data, camera=cam_name)
                    rgb = renderer.render()
                    frames[cam_name] = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                
                recorder.add(
                    frames=frames,
                    qpos=data.qpos[:n_motors].copy(),
                    ee_pose=data.site_xpos[tip_id].copy(),
                    action=current_action.copy(),
                    timestamp=data.time
                )

            # Timeout check (40초)
            if data.time - traj_start_time > 40.0:
                break

        # Episode End
        if success:
            recorder.save_async()
            episode_count += 1
            pbar.update(1)
        else:
            recorder.discard()

    pbar.close()
    print("\n✅ All Collections Finished!")

if __name__ == "__main__":
    main()