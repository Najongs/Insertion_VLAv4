# 데이터셋 업로드 가이드

VLA Insertion 데이터셋을 Hugging Face Hub에 업로드하는 방법입니다.

## 📋 목차

1. [Quick Start](#quick-start)
2. [Blue Point 에피소드 10개 업로드](#blue-point-에피소드-10개-업로드)
3. [전체 색상 데이터셋 업로드](#전체-색상-데이터셋-업로드)
4. [개별 에피소드 선택 업로드](#개별-에피소드-선택-업로드)
5. [업로드된 데이터셋 사용](#업로드된-데이터셋-사용)

---

## Quick Start

### 사전 준비

```bash
# 1. 필요한 패키지 설치
pip install datasets huggingface_hub pillow

# 2. Hugging Face 토큰 설정
export HF_TOKEN="hf_xxxxxxxxxxxxxxxxxxxxxx"
```

---

## Blue Point 에피소드 10개 업로드

가장 간단한 예제입니다. Blue_point 디렉토리의 처음 10개 에피소드를 업로드합니다.

### 1. 스크립트 수정

`upload_blue_dataset.sh` 파일 열기:

```bash
nano upload_blue_dataset.sh
```

Repository ID 수정:

```bash
REPO_ID="Najongs/vla-insertion-blue-point"  # 본인의 username으로 변경
```

### 2. 실행

```bash
cd /home/najo/NAS/VLA/Insertion_VLAv4/sub_tasks
bash upload_blue_dataset.sh
```

### 3. 결과 확인

업로드 완료 후 다음에서 확인:
```
https://huggingface.co/datasets/Najongs/vla-insertion-blue-point
```

---

## 전체 색상 데이터셋 업로드

5가지 색상(Blue, Green, Red, White, Yellow)의 데이터셋을 한 번에 업로드합니다.

### 1. 스크립트 수정

`upload_all_datasets.sh` 파일 열기:

```bash
nano upload_all_datasets.sh
```

Username 수정:

```bash
USERNAME="Najongs"  # 본인의 username으로 변경
```

### 2. 실행

```bash
cd /home/najo/NAS/VLA/Insertion_VLAv4/sub_tasks
bash upload_all_datasets.sh
```

### 3. 생성되는 데이터셋

다음 5개의 데이터셋이 생성됩니다:

- `username/vla-insertion-blue_point`
- `username/vla-insertion-green_point`
- `username/vla-insertion-red_point`
- `username/vla-insertion-white_point`
- `username/vla-insertion-yellow_point`

각 데이터셋에는 최대 10개의 에피소드가 포함됩니다.

---

## 개별 에피소드 선택 업로드

특정 에피소드들만 선택해서 업로드할 수 있습니다.

### 방법 1: 디렉토리 지정 (자동 선택)

```bash
python upload_dataset.py \
    --episode_dir /home/najo/NAS/VLA/dataset/New_dataset2/Blue_point \
    --max_episodes 10 \
    --repo_id "username/vla-insertion-blue" \
    --dataset_name "VLA Insertion - Blue Point" \
    --private
```

### 방법 2: 에피소드 직접 지정

```bash
python upload_dataset.py \
    --episode_dirs \
        /home/najo/NAS/VLA/dataset/New_dataset2/Blue_point/data_collection_20251108_055533 \
        /home/najo/NAS/VLA/dataset/New_dataset2/Blue_point/data_collection_20251108_055647 \
        /home/najo/NAS/VLA/dataset/New_dataset2/Blue_point/data_collection_20251108_055800 \
    --repo_id "username/vla-insertion-blue-selected" \
    --dataset_name "VLA Insertion - Blue Point (Selected)" \
    --private
```

### 테스트 (업로드 없이 준비만)

```bash
python upload_dataset.py \
    --episode_dir /home/najo/NAS/VLA/dataset/New_dataset2/Blue_point \
    --max_episodes 3 \
    --repo_id "username/test-dataset" \
    --no_upload
```

이렇게 하면 `outputs/dataset_upload/` 디렉토리에 데이터셋이 준비되며, 확인 후 수동 업로드 가능합니다.

---

## 고급 사용법

### 에피소드 개수 조정

```bash
# 5개만 업로드
python upload_dataset.py \
    --episode_dir /path/to/episodes \
    --max_episodes 5 \
    --repo_id "username/dataset"

# 모든 에피소드 업로드 (max_episodes 미지정)
python upload_dataset.py \
    --episode_dir /path/to/episodes \
    --repo_id "username/dataset"
```

### 공개/비공개 설정

```bash
# 비공개 데이터셋
python upload_dataset.py ... --private

# 공개 데이터셋 (--private 플래그 제거)
python upload_dataset.py ...
```

### 여러 색상 혼합 업로드

```bash
python upload_dataset.py \
    --episode_dirs \
        /path/to/Blue_point/episode1 \
        /path/to/Blue_point/episode2 \
        /path/to/Green_point/episode1 \
        /path/to/Red_point/episode1 \
    --repo_id "username/vla-insertion-mixed" \
    --dataset_name "VLA Insertion - Mixed Colors"
```

---

## 업로드된 데이터셋 사용

### Python에서 사용

```python
from datasets import load_dataset

# 데이터셋 로드
dataset = load_dataset("username/vla-insertion-blue-point")

# 데이터셋 정보 확인
print(dataset)
print(f"Total frames: {len(dataset['train'])}")

# 첫 번째 샘플 확인
sample = dataset["train"][0]
print(f"Episode: {sample['episode_id']}")
print(f"Frame: {sample['frame_index']}")
print(f"Pose: {sample['end_effector_pose']}")
print(f"Image: {sample['image_View1']}")  # PIL Image

# 이미지 표시
sample['image_View1'].show()
```

### 데이터 로더 생성

```python
from torch.utils.data import DataLoader
from datasets import load_dataset

dataset = load_dataset("username/vla-insertion-blue-point", split="train")

# Convert to PyTorch format
dataset.set_format(type="torch", columns=["joint_positions", "end_effector_pose"])

# Create DataLoader
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

for batch in dataloader:
    poses = batch["end_effector_pose"]
    # Training code...
```

### 필터링

```python
# 특정 에피소드만 선택
episode_1 = dataset["train"].filter(
    lambda x: x["episode_id"] == "data_collection_20251108_055533"
)

# 특정 프레임 범위만 선택
first_100_frames = dataset["train"].select(range(100))
```

---

## 데이터셋 구조

업로드된 데이터셋은 다음 정보를 포함합니다:

```python
{
    "episode_id": str,              # 에피소드 식별자
    "frame_index": int,             # 프레임 번호
    "timestamp": float,             # 시간 (초)
    "joint_positions": [float] * 6, # 로봇 관절 각도
    "end_effector_pose": [float] * 6,  # 엔드이펙터 위치 [x,y,z,a,b,r]
    "image_View1": PIL.Image,       # 카메라 1 이미지
    "image_View2": PIL.Image,       # 카메라 2 이미지
    "image_View3": PIL.Image,       # 카메라 3 이미지
    "image_View4": PIL.Image,       # 카메라 4 이미지
    "image_View5": PIL.Image,       # 카메라 5 이미지
    "sensor_alines": [float] * 1025,  # OCT 센서 데이터 (선택)
    "sensor_force": float,          # 힘 측정값 (선택)
}
```

---

## 문제 해결

### Q: "No episodes found" 오류

**원인:** 지정한 디렉토리에 `metadata.json` 파일이 없음

**해결:**
```bash
# 디렉토리 구조 확인
ls -la /home/najo/NAS/VLA/dataset/New_dataset2/Blue_point/

# metadata.json이 각 에피소드 디렉토리에 있는지 확인
ls -la /home/najo/NAS/VLA/dataset/New_dataset2/Blue_point/data_collection_*/
```

### Q: 업로드가 느려요

**원인:** 이미지 파일이 많아서 시간이 걸립니다

**해결:**
- 인내심을 갖고 기다리기
- 에피소드 개수를 줄여서 테스트
- `--max_episodes` 옵션 사용

### Q: 메모리 부족 오류

**원인:** 한 번에 너무 많은 데이터를 로드

**해결:**
```bash
# 에피소드를 나눠서 업로드
python upload_dataset.py --episode_dir /path --max_episodes 5 --repo_id "user/dataset-part1"
python upload_dataset.py --episode_dir /path --max_episodes 5 --repo_id "user/dataset-part2"
```

### Q: Token 오류

**원인:** HF_TOKEN이 설정되지 않았거나 권한이 없음

**해결:**
```bash
# 토큰 확인
echo $HF_TOKEN

# 토큰 재설정
export HF_TOKEN="hf_your_new_token"

# 또는 huggingface-cli로 로그인
huggingface-cli login
```

---

## 데이터셋 정보

### 파일 크기 예상

- **에피소드당**: ~500MB - 2GB (이미지 개수에 따라)
- **10 에피소드**: ~5GB - 20GB
- **50 에피소드 (전체)**: ~25GB - 100GB

### 업로드 시간 예상

- **인터넷 속도 100Mbps**: 10 에피소드 약 10-30분
- **인터넷 속도 1Gbps**: 10 에피소드 약 1-5분

---

## 참고 자료

- [Hugging Face Datasets 문서](https://huggingface.co/docs/datasets/)
- [데이터셋 카드 작성 가이드](https://huggingface.co/docs/hub/datasets-cards)
- [데이터셋 업로드 튜토리얼](https://huggingface.co/docs/datasets/upload_dataset)

---

## 다음 단계

데이터셋 업로드 후:

1. README.md 확인 및 수정
2. 데이터셋 카드 커스터마이징
3. 예제 노트북 추가
4. Community tab에서 사용자와 소통

---

더 자세한 내용은 [README.md](README.md)를 참조하세요.
