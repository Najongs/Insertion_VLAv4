# VLA Dataset for Insertion Task

통합 데이터셋 모듈 - 로봇 insertion 작업을 위한 PyTorch Dataset

## 📁 데이터 구조

```
/home/najo/NAS/VLA/dataset/
├── New_dataset2/
│   ├── Green_point/
│   │   ├── data_collection_20251108_053848/
│   │   │   ├── metadata.json          # 메타데이터
│   │   │   ├── robot_states.npz       # 로봇 상태 (joints + poses)
│   │   │   ├── sensor_data_*.npz      # 센서 데이터 (OCT alines + forces)
│   │   │   ├── View1/*.jpg            # 카메라 뷰 1
│   │   │   ├── View2/*.jpg            # 카메라 뷰 2
│   │   │   └── ...
│   │   └── ...
│   ├── Red_point/
│   └── Blue_point/
├── New_dataset3/
├── New_dataset4/
├── New_dataset5/
└── New_dataset6/
```

## 🚀 사용법

### 기본 사용 예제

```python
from vla_dataset import VLADataset, create_dataloader

# 단일 에피소드 로드
dataset = VLADataset(
    data_dir="/home/najo/NAS/VLA/dataset/New_dataset2/Green_point/data_collection_20251108_053848",
    horizon=8,
    sensor_window_size=65,
    robot_window_size=100,
    action_expert_hz=10,
)

print(f"Dataset size: {len(dataset)}")

# 샘플 가져오기
sample = dataset[0]
print(f"Images: {len(sample['images'])} views")
print(f"Sensor data: {sample['sensor_data'].shape}")  # (65, 1026)
print(f"Robot states: {sample['robot_states'].shape}")  # (100, 12)
print(f"Actions: {sample['actions'].shape}")  # (8, 7)
```

### DataLoader 생성

```python
# 여러 태스크의 에피소드들을 자동으로 로드
dataloader = create_dataloader(
    dataset_paths=[
        "/home/najo/NAS/VLA/dataset/New_dataset2/Green_point",
        "/home/najo/NAS/VLA/dataset/New_dataset2/Red_point",
        "/home/najo/NAS/VLA/dataset/New_dataset2/Blue_point",
    ],
    batch_size=4,
    num_workers=4,
    shuffle=True,
    horizon=8,
    sensor_window_size=65,
    robot_window_size=100,
    action_expert_hz=10,
)

# 학습 루프
for batch in dataloader:
    instructions = batch['instruction']  # List[str]
    images = batch['images']  # List[List[str]]
    sensor_data = batch['sensor_data']  # (B, T_sensor, 1026)
    robot_states = batch['robot_states']  # (B, T_robot, 12)
    actions = batch['actions']  # (B, horizon, 7)

    # 모델 학습...
```

## 📊 데이터 형식

### Sample 구조

```python
{
    'instruction': str,              # 태스크 instruction
    'images': List[str],            # 이미지 경로 리스트 (각 view마다)
    'sensor_data': Tensor,          # Shape: (sensor_window_size, 1026)
                                    #   1026 = 1025 (OCT alines) + 1 (force)
    'robot_states': Tensor,         # Shape: (robot_window_size, 12)
                                    #   12 = 6 (joints) + 6 (poses)
    'actions': Tensor,              # Shape: (horizon, 7)
                                    #   7 = 3 (delta_xyz) + 3 (delta_rotation) + 1 (gripper)
    'has_sensor': bool,             # 센서 데이터 유효 여부
    'has_robot_states': bool,       # 로봇 상태 유효 여부
    'episode_id': str,              # 에피소드 ID
    'timestamp': float,             # 타임스탬프
}
```

### Batch 구조

```python
{
    'instruction': List[str],                    # (B,)
    'images': List[List[str]],                   # (B, num_views)
    'sensor_data': Tensor,                       # (B, sensor_window_size, 1026)
    'robot_states': Tensor,                      # (B, robot_window_size, 12)
    'actions': Tensor,                           # (B, horizon, 7)
    'has_sensor_mask': BoolTensor,              # (B,)
    'has_robot_states_mask': BoolTensor,        # (B,)
    'episode_ids': List[str],                    # (B,)
    'timestamps': List[float],                   # (B,)
}
```

## ⚙️ 파라미터 설명

### VLADataset

- `data_dir`: 에피소드 디렉토리 경로 (metadata.json 포함)
- `horizon`: Action prediction horizon (default: 8)
  - 한 번에 예측할 미래 action의 개수
- `sensor_window_size`: 센서 히스토리 윈도우 크기 (default: 65)
  - 과거 65개의 센서 데이터 사용 (trailing window)
- `robot_window_size`: 로봇 상태 히스토리 윈도우 크기 (default: 100)
  - 과거 100개의 로봇 상태 사용 (trailing window)
- `action_expert_hz`: Action frequency in Hz (default: 10)
  - 로봇은 100Hz로 움직이지만, action은 10Hz로 생성
  - action_interval = robot_hz / action_expert_hz = 10

### create_dataloader

- `dataset_paths`: 태스크 디렉토리 또는 에피소드 디렉토리 경로 리스트
- `batch_size`: 배치 크기 (default: 4)
- `num_workers`: DataLoader worker 수 (default: 4)
- `shuffle`: 데이터 섞기 여부 (default: True)

## 🔧 주요 특징

1. **메모리 최적화**
   - `mmap_mode='r'`을 사용한 lazy loading
   - 실제 필요할 때만 데이터를 메모리에 로드

2. **Trailing Window**
   - 센서 데이터와 로봇 상태는 **과거 데이터만** 사용
   - 실제 inference 상황과 동일하게 미래 데이터를 사용하지 않음

3. **Delta Action**
   - Action은 absolute pose가 아닌 **delta pose**로 계산
   - Translation: `end_pose[:3] - start_pose[:3]`
   - Rotation: rotation vector로 표현된 delta rotation

4. **Terminal Action**
   - 에피소드 끝 5개 action은 정지 신호 (모두 0)

5. **Multi-view Support**
   - View1 ~ View5까지 최대 5개의 카메라 뷰 지원

## 📦 의존성

```bash
pip install -r requirements.txt
```

필요한 패키지:
- torch >= 2.0.0
- numpy >= 1.24.0
- scipy >= 1.10.0
- pandas >= 2.0.0
- Pillow >= 9.0.0

## 🤖 SmolVLA 학습

### 학습 준비

VLA 데이터셋으로 SmolVLA 모델을 학습하기 위한 파일들이 준비되어 있습니다:

1. **lerobot_adapter.py** - VLA 데이터셋을 LeRobot 형식으로 변환
2. **train_config.yaml** - SmolVLA 학습 설정
3. **train_smolvla.py** - 학습 스크립트

### 빠른 시작

```bash
cd /home/najo/NAS/VLA/Insertion_VLAv4/Train

# 1. 데이터셋 어댑터 테스트
python lerobot_adapter.py

# 2. 학습 시작 (기본 설정)
python train_smolvla.py --config train_config.yaml

# 3. 커스텀 설정으로 학습
python train_smolvla.py \
  --config train_config.yaml \
  --batch_size 4 \
  --steps 10000 \
  --lr 5e-5
```

### 학습 설정 수정

`train_config.yaml` 파일에서 주요 설정을 수정할 수 있습니다:

```yaml
# 데이터셋 에피소드 추가/제거
dataset:
  episode_dirs:
    - "New_dataset2/Green_point/data_collection_20251108_053719"
    - "New_dataset2/Green_point/data_collection_20251108_053848"
    # ... 더 많은 에피소드 추가

# 학습 파라미터 조정
training:
  batch_size: 8          # GPU 메모리에 맞게 조정
  steps: 20000           # 학습 스텝 수
  log_freq: 100          # 로그 출력 빈도
  save_freq: 2000        # 체크포인트 저장 빈도

# 최적화 설정
optimizer:
  lr: 1e-4               # 학습률
```

### 학습 모니터링

학습 중 다음 정보가 출력됩니다:
- Loss 값
- Learning rate
- 학습 시간

체크포인트는 `outputs/train/smolvla_vla_insertion/checkpoints/`에 저장됩니다.

### 학습된 모델 사용

학습이 완료되면 `outputs/train/smolvla_vla_insertion/final_model/`에 최종 모델이 저장됩니다.

이 모델을 inference 코드에서 사용:

```python
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

# 학습된 모델 로드
policy = SmolVLAPolicy.from_pretrained(
    "outputs/train/smolvla_vla_insertion/final_model"
)
policy.eval()

# 추론 실행
action = policy.select_action(observation)
```

## 🧪 테스트

```bash
cd /home/najo/NAS/VLA/Insertion_VLAv4/Train

# VLA 데이터셋 테스트
python vla_dataset.py

# LeRobot 어댑터 테스트
python lerobot_adapter.py
```

예상 출력:
```
🧪 Testing VLA Dataset...
✅ Dataset created: 710 samples
✅ All tests passed!
```

## 📝 참고사항

### Action 계산

Action은 다음과 같이 계산됩니다:

```python
# 10Hz action from 100Hz robot states
action_interval = 10  # 100Hz / 10Hz

for i in range(horizon):
    start_idx = (action_step + i) * action_interval
    end_idx = start_idx + action_interval

    # Delta translation
    delta_xyz = poses[end_idx][:3] - poses[start_idx][:3]

    # Delta rotation (rotation vector)
    r_start = Rotation.from_euler("xyz", poses[start_idx][3:], degrees=True)
    r_end = Rotation.from_euler("xyz", poses[end_idx][3:], degrees=True)
    delta_rotation = (r_end * r_start.inv()).as_rotvec()

    # Combine: [dx, dy, dz, drx, dry, drz, gripper]
    action = [*delta_xyz, *delta_rotation, 1.0]
```

### Sensor Data

센서 데이터는 OCT A-line과 Force 센서를 결합:

```python
# Shape: (sensor_window_size, 1026)
#   - alines: (sensor_window_size, 1025)
#   - forces: (sensor_window_size, 1)
sensor_data = np.concatenate([alines, forces[:, None]], axis=1)
```

센서는 650Hz로 샘플링되며, 로봇 100Hz에 대응하여 동기화:
```python
sensor_ratio = 650 / 100 = 6.5
sensor_idx = robot_idx * 6.5
```
