import mujoco
import mujoco.viewer
import numpy as np
import cv2
import time

# 1. Load Model
model_path = "meca_scene22.xml"
print(f"Loading Model: {model_path}")
model = mujoco.MjModel.from_xml_path(model_path)
data = mujoco.MjData(model)
renderer = mujoco.Renderer(model, height=480, width=640)

# 2. Get IDs
try:
    tip_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "needle_tip")
    back_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "needle_back")
    target_entry_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "trocar_target")
    target_depth_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "trocar_depth")
    viz_tip_tgt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "viz_target_tip")
    phantom_pos_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "phantom_assembly") # 이동용
    rotating_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "rotating_assembly")   # 회전용
    dof = model.nv
except Exception as e:
    print(f"[Error] XML ID Load Failed: {e}")
    raise e

# === Control Parameters ===
damping = 1e-3
current_speed = 0.5

TARGET_DISTANCE_FROM_ENTRY = 0.0001
TARGET_INSERTION_DEPTH = 0.02
COAXIAL_TOLERANCE = 50e-6

# State Variables
task_state = 1
align_timer = 0
insertion_started = False
accumulated_depth = 0.0
phase3_base_tip = np.zeros(3)

# === Trajectory Variables (부드러운 움직임을 위한 변수) ===
traj_initialized = False
traj_start_time = 0.0
TRAJ_DURATION = 5.0  # 정렬까지 걸리는 시간 (초) - 숫자가 클수록 더 천천히 움직임
start_tip_pos = np.zeros(3)
start_back_pos = np.zeros(3)

def randomize_phantom_pos(model, data):
    # 1. 위치 이동 (Translation) -> phantom_assembly에 적용
    offset_x = np.random.uniform(-0.1, 0.1)
    offset_y = np.random.uniform(-0.05, 0.03)
    offset_z = 0.0 
    new_pos = np.array([offset_x, offset_y, offset_z])
    model.body_pos[phantom_pos_id] = new_pos
    
    # 2. 회전 (Rotation) -> rotating_assembly에 적용
    # Z축 기준 랜덤 회전 (-30도 ~ 30도)
    random_angle_deg = np.random.uniform(-30, 30)
    new_quat = np.zeros(4)
    mujoco.mju_euler2Quat(new_quat, [0, 0, np.deg2rad(random_angle_deg)], "xyz")
    model.body_quat[rotating_id] = new_quat
    print(f">>> Randomize: Pos=({offset_x:.2f}, {offset_y:.2f}), Angle={random_angle_deg:.1f} deg")
    mujoco.mj_forward(model, data)
    

# === Helper: SmoothStep Function ===
# 0에서 1로 변할 때 부드러운 S자 곡선을 그리며 변환 (Ease-In, Ease-Out 효과)
def smooth_step(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3 - 2 * t)

randomize_phantom_pos(model, data)

print("Started: Smooth Trajectory Mode")

home_pose = np.array([0.5, 0.0, 0.0, 0.0, -0.5, 0.0])

with mujoco.viewer.launch_passive(model, data) as viewer:
    data.qpos[:6] = home_pose  # 1. 관절 값 대입
    mujoco.mj_forward(model, data)
    needle_len = np.linalg.norm(data.site_xpos[tip_id] - data.site_xpos[back_id])

    step_count = 0
    
    # Init Viz
    target_tip_pos = data.site_xpos[tip_id].copy()
    
    while viewer.is_running():
        step_start = time.time()
        curr_time = data.time
        
        # Current States
        p_entry = data.site_xpos[target_entry_id].copy()
        p_depth = data.site_xpos[target_depth_id].copy()
        curr_tip = data.site_xpos[tip_id].copy()
        curr_back = data.site_xpos[back_id].copy()

        # Target Vectors
        axis_vec = p_depth - p_entry
        axis_dir = axis_vec / (np.linalg.norm(axis_vec) + 1e-10)
        
        # === State Machine ===
        if task_state == 1:
            # [Phase 1: Smooth Approach & Align]
            status_color = (255, 0, 255)
            
            # 1. 초기화: 움직임 시작 전, 현재 위치(Start)를 딱 한 번 저장
            if not traj_initialized:
                traj_start_time = curr_time
                start_tip_pos = curr_tip.copy()
                start_back_pos = curr_back.copy()
                traj_initialized = True
                print(">>> Trajectory Generated: Moving smoothly to target.")

            # 2. 진행률(Progress) 계산 (0.0 ~ 1.0)
            elapsed_t = curr_time - traj_start_time
            raw_progress = elapsed_t / TRAJ_DURATION
            alpha = smooth_step(raw_progress) # 부드러운 가속/감속 적용
            
            # 3. 최종 목표 위치 계산 (Goal)
            goal_tip = p_entry - (axis_dir * TARGET_DISTANCE_FROM_ENTRY)
            goal_back = goal_tip - (axis_dir * needle_len)
            
            # 4. 보간(Interpolation): 현재 목표점(Moving Target) 계산
            # Start에서 Goal까지 alpha만큼 이동한 지점
            target_tip_pos = (1 - alpha) * start_tip_pos + alpha * goal_tip
            target_back_pos = (1 - alpha) * start_back_pos + alpha * goal_back
            
            msg = f"Aligning... {alpha*100:.1f}%"

            # 5. 도착 확인 및 정밀 정렬 체크
            if raw_progress >= 1.0:
                # 보간이 끝났어도 미세한 오차가 있을 수 있으므로 체크
                dist_error = np.linalg.norm(curr_tip - goal_tip)
                
                # 동축 오차 확인
                vec_entry_to_tip = curr_tip - p_entry
                proj_point = p_entry + (np.dot(vec_entry_to_tip, axis_dir) * axis_dir)
                dist_coaxial = np.linalg.norm(curr_tip - proj_point)
                
                if dist_error < 0.002 and dist_coaxial < COAXIAL_TOLERANCE:
                    align_timer += 1
                    msg = "Holding Alignment..."
                else:
                    align_timer = 0
                
                if align_timer > 20:
                    task_state = 2
                    insertion_started = False
                    print(">>> Alignment Complete. Starting Insertion.")

        elif task_state == 2:
            # [Phase 2: Insertion]
            msg = f"Inserting ({accumulated_depth*1000:.1f}mm)"
            status_color = (0, 255, 0)
            
            if not insertion_started:
                phase3_base_tip = p_entry - (axis_dir * TARGET_DISTANCE_FROM_ENTRY)
                insertion_started = True
                accumulated_depth = 0.0

            step_z = 0.000025
            accumulated_depth += step_z
            
            target_tip_pos = phase3_base_tip + (axis_dir * accumulated_depth)
            target_back_pos = target_tip_pos - (axis_dir * needle_len)
            
            current_speed = 0.5

            if accumulated_depth >= TARGET_INSERTION_DEPTH:
                task_state = 3
                print(">>> Finished.")

        elif task_state == 3:
            msg = "FINISHED"
            status_color = (0, 0, 255)
            current_speed = 0.5

        # === Stacked IK Solver ===
        data.site_xpos[viz_tip_tgt_id] = target_tip_pos
        
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
        J_p1 = jac_tip_full[:3] # Tip Pos
        e_p1 = err_tip * 50.0   # Tip Error Gain
        
        # [중요] Error Clamping: 급격한 움직임 방지를 위한 안전장치
        # 타겟이 멀리 있어도 속도를 강제로 제한함
        if np.linalg.norm(e_p1) > 1.0: e_p1 = e_p1 / np.linalg.norm(e_p1) * 1.0
            
        J_p1_pinv = np.linalg.pinv(J_p1, rcond=1e-4)
        dq_p1 = J_p1_pinv @ e_p1
        P_null_1 = np.eye(dof) - (J_p1_pinv @ J_p1)
        
        J_p2 = jac_back # Back Pos
        e_p2 = err_back * 50.0
        J_p2_proj = J_p2 @ P_null_1
        J_p2_pinv = np.linalg.pinv(J_p2_proj, rcond=1e-4)
        dq_p2 = J_p2_pinv @ (e_p2 - J_p2 @ dq_p1)
        P_null_2 = P_null_1 - (J_p2_pinv @ J_p2_proj)
        
        J_p3 = jac_tip_full[3:] # Roll
        e_p3 = err_roll * 10.0
        J_p3_proj = J_p3 @ P_null_2
        J_p3_pinv = np.linalg.pinv(J_p3_proj, rcond=1e-4)
        dq_p3 = J_p3_pinv @ (e_p3 - J_p3 @ (dq_p1 + dq_p2))
        
        dq = dq_p1 + dq_p2 + dq_p3
        
        data.ctrl[:] = data.qpos[:dof] + (dq * current_speed)
        mujoco.mj_step(model, data)
        step_count += 1
        
        if step_count % 67 == 0:
            frames = []
            for cam in ["side_camera", "tool_camera", "top_camera"]:
                renderer.update_scene(data, camera=cam)
                frames.append(cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR))
            combined = np.hstack(frames)
            cv2.putText(combined, msg, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
            cv2.imshow("Smooth Trajectory", combined)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'): break
            elif key == ord('r'):
                mujoco.mj_resetData(model, data)
                data.qpos[:6] = home_pose
                randomize_phantom_pos(model, data)
                mujoco.mj_forward(model, data)
                task_state = 1
                traj_initialized = False # 궤적 초기화 플래그 리셋
                insertion_started = False
                accumulated_depth = 0.0

        viewer.sync()
        elapsed = time.time() - step_start
        if elapsed < model.opt.timestep:
            time.sleep(model.opt.timestep - elapsed)

cv2.destroyAllWindows()