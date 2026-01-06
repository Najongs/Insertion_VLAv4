import h5py
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import sys

def save_sensor_plots(force_data, aline_data, output_path):
    """센서 데이터(Force + A-line) 시각화를 이미지 파일로 저장"""
    print(f"📊 센서 데이터 그래프를 생성합니다: {output_path}")
    print(f"  Force shape: {force_data.shape}, A-line shape: {aline_data.shape}")

    fig, axes = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [1, 2]})

    # 1. Force 데이터 (시계열)
    ax1 = axes[0]
    ax1.plot(force_data, linewidth=1.5, color='red')
    ax1.set_title('Force Sensor Data (FPI)', fontsize=12, weight='bold')
    ax1.set_xlabel('Time Steps')
    ax1.set_ylabel('Force (N)')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(0, color='black', linewidth=0.5, linestyle='--')

    # 통계 정보 추가
    force_stats = f"Min: {force_data.min():.3f}, Max: {force_data.max():.3f}, Mean: {force_data.mean():.3f}, Std: {force_data.std():.3f}"
    ax1.text(0.02, 0.98, force_stats, transform=ax1.transAxes,
             fontsize=9, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # 2. A-line 데이터 (M-mode 이미지)
    ax2 = axes[1]
    # Transpose: (시간, 1025) → (1025, 시간) for correct display
    im = ax2.imshow(aline_data.T, aspect='auto', cmap='gray', interpolation='nearest')
    ax2.set_title('OCT A-scan Data (M-mode Image)', fontsize=12, weight='bold')
    ax2.set_xlabel('Time Steps')
    ax2.set_ylabel('Depth Pixels (A-scan)')
    cbar = fig.colorbar(im, ax=ax2, label='Intensity')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ 센서 그래프가 저장되었습니다: {output_path}")


def save_plots(actions, qpos, ee_pose, output_path):
    """데이터 그래프를 이미지 파일로 저장"""
    print(f"📊 그래프를 생성하여 저장합니다: {output_path}")

    # 데이터 검증
    qpos_valid = np.any(np.abs(qpos) > 0.001)
    ee_pose_valid = np.any(np.abs(ee_pose) > 0.001)

    if not qpos_valid:
        print("⚠️  WARNING: qpos (joint positions) is all zeros! Robot data may not have been recorded.")
    if not ee_pose_valid:
        print("⚠️  WARNING: ee_pose (end-effector pose) is all zeros! Robot data may not have been recorded.")

    # 통계 출력
    print(f"\n📈 Data Statistics:")
    print(f"  Actions:  min={actions.min():.3f}, max={actions.max():.3f}, mean={actions.mean():.3f}")
    print(f"  Qpos:     min={qpos.min():.3f}, max={qpos.max():.3f}, std={qpos.std():.3f}")
    print(f"  EE Pose:  min={ee_pose.min():.3f}, max={ee_pose.max():.3f}, std={ee_pose.std():.3f}")
    print(f"  EE Position (XYZ):    min={ee_pose[:, :3].min():.3f}, max={ee_pose[:, :3].max():.3f}")
    print(f"  EE Orientation (RPY): min={ee_pose[:, 3:].min():.3f}, max={ee_pose[:, 3:].max():.3f}")

    fig, axes = plt.subplots(4, 1, figsize=(14, 16))

    # 1. Action (명령) - 6축 모두 확인
    ax1 = axes[0]
    title_suffix = " [OK]" if np.any(np.abs(actions) > 0.001) else " [No data]"
    ax1.set_title("1. Joystick Actions (Command)" + title_suffix, fontsize=12, weight='bold')
    ax1.plot(actions[:, 0], label='X (Vel)', linestyle='--', alpha=0.6, linewidth=1.5)
    ax1.plot(actions[:, 1], label='Y (Vel)', linestyle='--', alpha=0.6, linewidth=1.5)
    ax1.plot(actions[:, 2], label='Z (Vel)', linestyle='--', alpha=0.6, linewidth=1.5)
    ax1.plot(actions[:, 3], label='Rx (Pitch)', linewidth=2)
    ax1.plot(actions[:, 4], label='Ry (Roll)', linewidth=2)
    ax1.plot(actions[:, 5], label='Rz (Yaw)', linewidth=2)
    ax1.legend(loc='upper right', ncol=3, fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylabel("Command Value")

    # 2. Joint Positions (관절) - 6개 관절 확인
    ax2 = axes[1]
    title_suffix = " [OK]" if qpos_valid else " [ERROR - All zeros]"
    ax2.set_title("2. Joint Positions (J1 ~ J6)" + title_suffix, fontsize=12, weight='bold')
    colors = ['r', 'g', 'b', 'c', 'm', 'y']
    for i in range(6):
        ax2.plot(qpos[:, i], label=f'Joint {i+1}', color=colors[i], linewidth=1.5)
    ax2.legend(loc='upper right', ncol=3, fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylabel("Degrees")

    # 3. End-Effector Position (위치: X, Y, Z)
    ax3 = axes[2]
    title_suffix = " [OK]" if ee_pose_valid else " [ERROR - All zeros]"
    ax3.set_title("3. End-Effector Position (X, Y, Z)" + title_suffix, fontsize=12, weight='bold')
    ax3.plot(ee_pose[:, 0], label='X (mm)', color='r', linewidth=2)
    ax3.plot(ee_pose[:, 1], label='Y (mm)', color='g', linewidth=2)
    ax3.plot(ee_pose[:, 2], label='Z (mm)', color='b', linewidth=2)
    ax3.legend(loc='upper right', fontsize=9)
    ax3.grid(True, alpha=0.3)
    ax3.set_ylabel("Position (mm)")

    # 4. End-Effector Orientation (회전: Rx, Ry, Rz)
    ax4 = axes[3]
    title_suffix = " [OK]" if ee_pose_valid else " [ERROR - All zeros]"
    ax4.set_title("4. End-Effector Orientation (Rx, Ry, Rz)" + title_suffix, fontsize=12, weight='bold')
    ax4.plot(ee_pose[:, 3], label='Rx (Alpha)', color='c', linewidth=2)
    ax4.plot(ee_pose[:, 4], label='Ry (Beta)', color='m', linewidth=2)
    ax4.plot(ee_pose[:, 5], label='Rz (Gamma)', color='y', linewidth=2)
    ax4.legend(loc='upper right', fontsize=9)
    ax4.grid(True, alpha=0.3)
    ax4.set_ylabel("Orientation (degrees)")
    ax4.set_xlabel("Time Steps")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ 그래프가 저장되었습니다: {output_path}")


def save_video_and_plots(file_path, output_dir=None):
    """데이터셋의 영상과 그래프를 파일로 저장"""
    print(f"📂 Loading Dataset: {file_path}")

    # 출력 디렉토리 설정
    if output_dir is None:
        output_dir = Path(file_path).parent / "saved_outputs"
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(exist_ok=True)

    # 파일명 생성 (타임스탬프 포함)
    dataset_name = Path(file_path).stem
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = f"{dataset_name}_{timestamp}"

    video_path = output_dir / f"{base_name}_video.mp4"
    plot_path = output_dir / f"{base_name}_plots.png"
    sensor_plot_path = output_dir / f"{base_name}_sensor.png"

    try:
        with h5py.File(file_path, 'r') as f:
            # --- 1. 데이터 로드 ---
            images_grp = f['observations/images']
            qpos = f['observations/qpos'][:]
            actions = f['action'][:]

            # EE Pose가 있는지 확인하고 로드
            if 'observations/ee_pose' in f:
                ee_pose = f['observations/ee_pose'][:]
            else:
                ee_pose = np.zeros_like(qpos)

            # 센서 데이터 로드 (있는 경우)
            has_sensor = 'observations/sensor' in f
            if has_sensor:
                sensor_grp = f['observations/sensor']
                force_data = sensor_grp['force'][:]
                aline_data = sensor_grp['aline'][:]
                print(f"✅ 센서 데이터 발견: Force {force_data.shape}, A-line {aline_data.shape}")
            else:
                force_data = None
                aline_data = None
                print(f"⚠️ 센서 데이터 없음")

            cam_keys = sorted(list(images_grp.keys()))
            total_steps = len(actions)
            print(f"📷 Cameras: {cam_keys}")
            print(f"⏱️ Total Steps: {total_steps}")

            # 영상 데이터 메모리 로드
            print("📥 Loading video frames to memory...")
            video_streams = {}
            for cam in cam_keys:
                video_streams[cam] = images_grp[cam][:]

            # --- 2. 비디오 저장 ---
            print(f"\n🎬 비디오를 저장합니다: {video_path}")

            # 첫 프레임으로 비디오 설정 초기화
            first_frames = []
            for cam in cam_keys:
                frame = cv2.cvtColor(video_streams[cam][0], cv2.COLOR_RGB2BGR)
                first_frames.append(frame)

            combined_sample = np.hstack(first_frames)
            h, w, _ = combined_sample.shape
            info_board_height = 100
            total_height = h + info_board_height

            # 해상도 조정 (너무 크면 줄이기)
            scale = 1.0
            if w > 1920:
                scale = 0.6
                w = int(w * scale)
                total_height = int(total_height * scale)

            # VideoWriter 초기화
            # FPS 설정: Robot_action.py의 CONTROL_FREQUENCY와 동일하게 설정
            fps = 15  # 데이터 수집 FPS와 동일 (Robot_action.py:32)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(video_path), fourcc, fps, (w, total_height))
            print(f"  FPS 설정: {fps}")

            # 모든 프레임 처리
            for i in range(total_steps):
                if i % 50 == 0:
                    print(f"  Progress: {i}/{total_steps} frames")

                frames = []
                for cam in cam_keys:
                    frame = cv2.cvtColor(video_streams[cam][i], cv2.COLOR_RGB2BGR)
                    cv2.putText(frame, f"{cam}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                    frames.append(frame)

                # 화면 병합
                combined_img = np.hstack(frames)

                # 정보창 (Info Board)
                h_orig, w_orig, _ = combined_img.shape
                info_board = np.zeros((info_board_height, w_orig, 3), dtype=np.uint8)

                # 현재 상태 텍스트
                curr_act = actions[i]
                curr_q = qpos[i]

                txt1 = f"Step: {i}/{total_steps}"
                txt2 = f"Act(Move): {curr_act[:3].round(2)} | Act(Rot): {curr_act[3:].round(2)}"
                txt3 = f"Joints: {curr_q.round(1)}"

                cv2.putText(info_board, txt1, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(info_board, txt2, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(info_board, txt3, (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

                display = np.vstack([combined_img, info_board])

                # 해상도 조정
                if scale < 1.0:
                    display = cv2.resize(display, (w, total_height))

                out.write(display)

            out.release()
            print(f"✅ 비디오가 저장되었습니다: {video_path}")

            # --- 3. 그래프 저장 ---
            save_plots(actions, qpos, ee_pose, plot_path)

            # --- 4. 센서 데이터 그래프 저장 (있는 경우) ---
            if has_sensor and force_data is not None and aline_data is not None:
                save_sensor_plots(force_data, aline_data, sensor_plot_path)

            print(f"\n🎉 모든 파일이 저장되었습니다!")
            print(f"  📁 출력 디렉토리: {output_dir}")
            print(f"  🎬 비디오: {video_path.name}")
            print(f"  📊 그래프: {plot_path.name}")
            if has_sensor:
                print(f"  📊 센서 그래프: {sensor_plot_path.name}")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


def visualize_sensor_data(npz_file):
    """
    .npz 파일에 저장된 센서 데이터를 시각화합니다.

    - force 데이터는 선 그래프로 표시합니다.
    - aline (fpi) 데이터는 2D 이미지로 표시합니다.
    """
    try:
        # 데이터 로드
        data_npz = np.load(npz_file)
    except FileNotFoundError:
        print(f"오류: 파일을 찾을 수 없습니다: {npz_file}")
        return

    # 'forces'와 'alines' 키 확인
    if 'forces' not in data_npz or 'alines' not in data_npz:
        print(f"오류: '{npz_file}' 파일에 'forces' 또는 'alines' 키가 없습니다.")
        print(f"사용 가능한 키: {list(data_npz.keys())}")
        return

    force_data = data_npz['forces']
    aline_data = data_npz['alines']
    print(f"Force 데이터를 로드했습니다. 형태: {force_data.shape}")
    print(f"Aline 데이터를 로드했습니다. 형태: {aline_data.shape}")

    # 시각화
    fig, axs = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [1, 3]})
    
    # 파일명을 제목에 추가
    base_filename = os.path.basename(npz_file)
    fig.suptitle(f'Sensor Data Visualization\n({base_filename})', fontsize=16)

    # Force 데이터 플롯
    axs[0].plot(force_data)
    axs[0].set_title('Force Data')
    axs[0].set_xlabel('Time step')
    axs[0].set_ylabel('Force')
    axs[0].grid(True)

    # Aline (FPI) 데이터 플롯 (이미지)
    im = axs[1].imshow(aline_data.T, aspect='auto', cmap='gray', interpolation='none')
    axs[1].set_title('Aline (FPI) Data')
    axs[1].set_xlabel('Time step')
    axs[1].set_ylabel('FPI sensor index')
    fig.colorbar(im, ax=axs[1], label='Sensor Reading')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # 이미지 파일로 저장
    output_filename = 'sensor_data_visualization.png'
    plt.savefig(output_filename)
    print(f"시각화 결과를 '{output_filename}' 파일로 저장했습니다.")
    plt.close()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        npz_file_path = sys.argv[1]
        visualize_sensor_data(npz_file_path)
    else:
        print("사용법: python visualize_sensor.py <.npz 파일 경로>")



if __name__ == "__main__":
    # 가장 최근 파일 자동 로드
    dataset_dir = Path("./collected_data")
    files = sorted(dataset_dir.glob("*.h5"), key=lambda f: f.stat().st_mtime, reverse=True)

    if not files:
        print("⚠️ No dataset files (.h5) found!")
    else:
        print(f"📌 처리할 파일: {files[0].name}")
        save_video_and_plots(files[0])
