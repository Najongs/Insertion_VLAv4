import h5py
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def plot_detailed_data(actions, qpos, ee_pose):
    """데이터 재생이 끝나면 실행되는 상세 그래프 함수"""
    print("📊 상세 데이터 그래프를 생성합니다...")

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
    # 이동(XYZ)은 점선으로 흐리게
    ax1.plot(actions[:, 0], label='X (Vel)', linestyle='--', alpha=0.6, linewidth=1.5)
    ax1.plot(actions[:, 1], label='Y (Vel)', linestyle='--', alpha=0.6, linewidth=1.5)
    ax1.plot(actions[:, 2], label='Z (Vel)', linestyle='--', alpha=0.6, linewidth=1.5)
    # 회전(RxRyRz)은 실선으로 진하게 (중점 확인)
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
    plt.show()

def play_and_inspect(file_path):
    print(f"📂 Loading Dataset: {file_path}")
    
    try:
        with h5py.File(file_path, 'r') as f:
            # --- 1. 데이터 로드 ---
            images_grp = f['observations/images']
            qpos = f['observations/qpos'][:]
            actions = f['action'][:]
            # EE Pose가 있는지 확인하고 로드 (없으면 0으로 채움)
            if 'observations/ee_pose' in f:
                ee_pose = f['observations/ee_pose'][:]
            else:
                ee_pose = np.zeros_like(qpos)

            cam_keys = sorted(list(images_grp.keys()))
            total_steps = len(actions)
            print(f"📷 Cameras: {cam_keys}")
            print(f"⏱️ Total Steps: {total_steps}")

            # 영상 데이터 메모리 로드
            # ✨ 자동 형식 감지 (JPEG / GZIP 모두 지원)
            video_streams = {}

            # 첫 번째 카메라로 형식 확인
            first_cam = cam_keys[0]
            first_dataset = images_grp[first_cam]

            # 실제 데이터를 읽어서 형식 감지 (가장 확실한 방법)
            is_jpeg_format = False
            try:
                # 첫 번째 프레임 샘플 읽기
                sample = first_dataset[0]

                if isinstance(sample, np.ndarray):
                    # shape가 (H, W, 3)이면 GZIP (이미 디코딩된 이미지)
                    if len(sample.shape) == 3 and sample.shape[2] == 3:
                        is_jpeg_format = False
                        print(f"   💡 Sample shape: {sample.shape} → GZIP format detected")
                    # shape가 (N,)이고 dtype이 uint8이면 JPEG (압축된 바이너리)
                    elif len(sample.shape) == 1 and sample.dtype == np.uint8:
                        is_jpeg_format = True
                        print(f"   💡 Sample shape: {sample.shape} → JPEG format detected")
                    else:
                        # 기타 경우: dataset shape로 판단
                        is_jpeg_format = len(first_dataset.shape) != 4
                else:
                    # numpy array가 아니면 JPEG일 가능성
                    is_jpeg_format = True
            except Exception as e:
                print(f"   ⚠️ Format detection failed: {e}, falling back to shape check")
                # Fallback: dataset shape로 판단
                is_jpeg_format = len(first_dataset.shape) != 4

            if is_jpeg_format:
                print("🔄 Detected JPEG compression format - Decoding frames...")
                for cam in cam_keys:
                    dataset = images_grp[cam]
                    decoded_frames = []
                    decode_failures = 0

                    # 프레임 수 확인
                    num_frames = len(dataset)

                    for idx in range(num_frames):
                        try:
                            # HDF5 vlen dtype 읽기 (각 요소 개별 접근)
                            jpeg_buf = dataset[idx]

                            # 디버깅: 첫 프레임 정보 출력
                            if idx == 0 and decode_failures == 0:
                                print(f"   🔍 Debug {cam} frame 0: type={type(jpeg_buf)}, ", end="")
                                if isinstance(jpeg_buf, np.ndarray):
                                    print(f"shape={jpeg_buf.shape}, dtype={jpeg_buf.dtype}")
                                else:
                                    print(f"len={len(jpeg_buf) if hasattr(jpeg_buf, '__len__') else 'N/A'}")

                            # 데이터 타입 확인 및 변환
                            if isinstance(jpeg_buf, np.ndarray):
                                # 이미 numpy array인 경우 - flatten 및 uint8 변환
                                jpeg_array = jpeg_buf.flatten().astype(np.uint8)
                            elif isinstance(jpeg_buf, (bytes, bytearray)):
                                # bytes인 경우 변환
                                jpeg_array = np.frombuffer(jpeg_buf, dtype=np.uint8)
                            else:
                                # 기타 타입 (리스트 등)
                                jpeg_array = np.array(jpeg_buf, dtype=np.uint8).flatten()

                            # 빈 버퍼 체크
                            if jpeg_array.size == 0:
                                decode_failures += 1
                                decoded_frames.append(np.zeros((480, 640, 3), dtype=np.uint8))
                                continue

                            # OpenCV가 요구하는 형식 확인: 1D contiguous uint8 array
                            if not jpeg_array.flags['C_CONTIGUOUS']:
                                jpeg_array = np.ascontiguousarray(jpeg_array)

                            # JPEG 디코딩: bytes → BGR → RGB
                            img_bgr = cv2.imdecode(jpeg_array, cv2.IMREAD_COLOR)

                            if img_bgr is not None and img_bgr.size > 0:
                                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                                decoded_frames.append(img_rgb)
                            else:
                                decode_failures += 1
                                # 빈 프레임 추가 (에러 방지)
                                decoded_frames.append(np.zeros((480, 640, 3), dtype=np.uint8))

                        except Exception as e:
                            decode_failures += 1
                            if idx < 5 or decode_failures < 5:  # 처음 몇 개만 상세 로그
                                print(f"   ⚠️ Error decoding {cam} frame {idx}: {e}")
                            decoded_frames.append(np.zeros((480, 640, 3), dtype=np.uint8))

                    video_streams[cam] = np.array(decoded_frames)
                    if decode_failures > 0:
                        print(f"   ⚠️ {cam}: {decode_failures}/{len(decoded_frames)} frames failed decoding (replaced with black)")
                    else:
                        print(f"   ✓ {cam}: {len(decoded_frames)} frames decoded")
            else:
                print("🔄 Detected GZIP compression format - Loading frames...")
                for cam in cam_keys:
                    # GZIP: 이미 디코딩된 RGB 배열
                    video_streams[cam] = images_grp[cam][:]
                    print(f"   ✓ {cam}: {len(video_streams[cam])} frames loaded") 

            # --- 2. 비디오 재생 (OpenCV) ---
            print("\n=== ⏯️ Playback Controls ===")
            print(" [SPACE] Pause/Resume")
            print(" [Q]     Stop Video & Show Graphs")
            
            paused = False
            i = 0
            
            while i < total_steps:
                if not paused:
                    frames = []
                    for cam in cam_keys:
                        # RGB -> BGR
                        frame = cv2.cvtColor(video_streams[cam][i], cv2.COLOR_RGB2BGR)
                        cv2.putText(frame, f"{cam}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                        frames.append(frame)
                    
                    # 화면 병합
                    combined_img = np.hstack(frames)
                    
                    # 정보창 (Info Board)
                    h, w, _ = combined_img.shape
                    info_board = np.zeros((100, w, 3), dtype=np.uint8)
                    
                    # 현재 상태 텍스트
                    curr_act = actions[i]
                    curr_q = qpos[i]
                    
                    # Action의 XYZ와 회전값(RxRyRz)을 나눠서 표시
                    txt1 = f"Step: {i}/{total_steps}"
                    txt2 = f"Act(Move): {curr_act[:3].round(2)} | Act(Rot): {curr_act[3:].round(2)}"
                    txt3 = f"Joints: {curr_q.round(1)}"
                    
                    cv2.putText(info_board, txt1, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.putText(info_board, txt2, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(info_board, txt3, (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
                    
                    display = np.vstack([combined_img, info_board])
                    
                    # 해상도가 너무 크면 줄이기
                    if display.shape[1] > 1920:
                        display = cv2.resize(display, (0, 0), fx=0.6, fy=0.6)

                    cv2.imshow("Dataset Verification Player", display)
                    i += 1

                key = cv2.waitKey(33) # 30 FPS
                if key == ord('q'):
                    break
                elif key == ord(' '):
                    paused = not paused

            cv2.destroyAllWindows()
            print("🎬 Playback finished.")

            # --- 3. 그래프 분석 (Matplotlib) ---
            plot_detailed_data(actions, qpos, ee_pose)

    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    # 가장 최근 파일 자동 로드
    dataset_dir = Path("./collected_data")
    files = sorted(dataset_dir.glob("*.h5"), key=lambda f: f.stat().st_mtime, reverse=True)
    
    if not files:
        print("⚠️ No dataset files (.h5) found!")
    else:
        play_and_inspect(files[0])