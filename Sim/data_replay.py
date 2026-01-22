import h5py
import cv2
import numpy as np
import os

# ==============================================================
# [설정] 확인하고 싶은 HDF5 파일 경로를 여기에 넣으세요
# (경로에 한글이 있으면 에러가 날 수 있으니 주의하세요)
# ==============================================================
FILE_PATH = r"C:\Users\DGIST\Desktop\fixed_mujoco-3.4.0-windows-x86_64\Meca500_urdf\meshes\collected_data_sim\episode_20260121_170218.h5"

def print_structure(name, obj):
    """HDF5 내부 구조를 출력하는 헬퍼 함수"""
    if isinstance(obj, h5py.Group):
        print(f"📁 [Group] {name}")
    elif isinstance(obj, h5py.Dataset):
        print(f"   💾 [Dataset] {name} | Shape: {obj.shape} | Type: {obj.dtype}")

def main():
    if not os.path.exists(FILE_PATH):
        print(f"❌ 파일을 찾을 수 없습니다: {FILE_PATH}")
        return

    print(f"OPENING: {FILE_PATH}")
    
    with h5py.File(FILE_PATH, 'r') as f:
        # 1. 내부 구조 출력 (디버깅용)
        print("\n=== HDF5 File Structure ===")
        f.visititems(print_structure)
        print("===========================\n")

        # 2. 데이터 로드
        # (학습 때는 필요한 만큼만 로드하지만, 뷰어니까 메모리에 다 올립니다)
        qpos_data = f['observations/qpos'][:]
        action_data = f['action'][:]
        ee_data = f['observations/ee_pose'][:]
        
        # 이미지 키 찾기 (side, top, tool 등)
        img_grp = f['observations/images']
        cam_keys = list(img_grp.keys())
        cam_keys.sort() # 순서 고정
        
        total_steps = len(qpos_data)
        print(f"🎬 Total Steps: {total_steps}")
        print(f"📷 Found Cameras: {cam_keys}")

        # 3. 뷰어 루프
        cv2.namedWindow("Dataset Viewer", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Dataset Viewer", 1200, 600)
        
        # 트랙바(슬라이더) 생성
        def nothing(x): pass
        cv2.createTrackbar("Step", "Dataset Viewer", 0, total_steps - 1, nothing)
        
        is_playing = True
        current_step = 0
        
        while True:
            # 트랙바 위치 업데이트 (재생 중일 때)
            if is_playing:
                current_step += 1
                if current_step >= total_steps:
                    current_step = 0 # 무한 반복
                cv2.setTrackbarPos("Step", "Dataset Viewer", current_step)
            else:
                # 일시정지 중이면 트랙바 위치를 가져옴
                current_step = cv2.getTrackbarPos("Step", "Dataset Viewer")

            # --- 이미지 디코딩 및 합치기 ---
            frames = []
            for k in cam_keys:
                # HDF5에서 바이너리 데이터 가져오기 (uint8 array)
                binary_data = img_grp[k][current_step]
                
                # [중요] JPEG Decoding
                # 저장할 때 cv2.imencode를 썼으므로, 읽을 땐 cv2.imdecode를 씁니다.
                img = cv2.imdecode(binary_data, cv2.IMREAD_COLOR)
                
                if img is None:
                    # 데이터가 깨졌거나 비어있을 경우 검은 화면
                    img = np.zeros((240, 320, 3), dtype=np.uint8)
                    cv2.putText(img, "Error", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                # 카메라 이름 표시
                cv2.putText(img, k, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                frames.append(img)
            
            # 가로로 이어 붙이기
            combined_img = np.hstack(frames)

            # --- 정보 오버레이 (HUD) ---
            # 하단에 검은 바 추가해서 텍스트 표시
            h, w, _ = combined_img.shape
            info_panel = np.zeros((150, w, 3), dtype=np.uint8)
            
            # 현재 상태 정보 가져오기
            cur_q = qpos_data[current_step]
            cur_act = action_data[current_step]
            cur_ee = ee_data[current_step]
            
            # 텍스트 출력
            green = (0, 255, 0)
            white = (255, 255, 255)
            font = cv2.FONT_HERSHEY_SIMPLEX
            
            cv2.putText(info_panel, f"Step: {current_step}/{total_steps} ({'PLAY' if is_playing else 'PAUSE'})", 
                        (20, 30), font, 0.7, green, 2)
            
            # 소수점 3자리까지 예쁘게 포맷팅
            q_str = "Qpos: " + " ".join([f"{x: .3f}" for x in cur_q])
            a_str = "Act : " + " ".join([f"{x: .3f}" for x in cur_act])
            ee_str = f"EE  : X={cur_ee[0]:.4f} Y={cur_ee[1]:.4f} Z={cur_ee[2]:.4f}"
            
            cv2.putText(info_panel, q_str, (20, 60), font, 0.5, white, 1)
            cv2.putText(info_panel, a_str, (20, 90), font, 0.5, (100, 255, 255), 1) # Action은 노란색
            cv2.putText(info_panel, ee_str, (20, 120), font, 0.5, white, 1)

            # 이미지 + 정보창 합치기 (세로)
            final_view = np.vstack([combined_img, info_panel])
            
            cv2.imshow("Dataset Viewer", final_view)
            
            # --- 키 입력 처리 ---
            key = cv2.waitKey(33) & 0xFF # 약 30FPS 속도
            
            if key == ord('q'): # 종료
                break
            elif key == 32: # Spacebar: 일시정지/재생 토글
                is_playing = not is_playing
            elif key == ord('a'): # A: 이전 프레임 (일시정지 상태에서)
                is_playing = False
                cv2.setTrackbarPos("Step", "Dataset Viewer", max(0, current_step - 1))
            elif key == ord('d'): # D: 다음 프레임 (일시정지 상태에서)
                is_playing = False
                cv2.setTrackbarPos("Step", "Dataset Viewer", min(total_steps - 1, current_step + 1))

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()