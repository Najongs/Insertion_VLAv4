# Hugging Face Hub 통합 도구

학습된 SmolVLA 모델과 VLA 데이터셋을 Hugging Face Hub에 업로드/다운로드하는 통합 도구입니다.

## 📋 목차

- [파일 구성](#파일-구성)
- [모델 업로드](#모델-업로드)
- [모델 다운로드](#모델-다운로드)
- [데이터셋 업로드](#데이터셋-업로드)
- [Quick Start 예제](#quick-start-예제)

## 파일 구성

```
sub_tasks/
# 모델 관련
├── upload_to_huggingface.py    # 모델 업로드 스크립트
├── upload_model.sh              # 모델 업로드 간편 실행
├── download_model.py            # 모델 다운로드 스크립트
├── download_model.sh            # 모델 다운로드 간편 실행

# 데이터셋 관련
├── upload_dataset.py            # 데이터셋 업로드 스크립트
├── upload_blue_dataset.sh       # Blue Point 업로드 예제
├── upload_all_datasets.sh       # 전체 색상 데이터셋 업로드

# 문서
├── README.md                    # 이 문서 (전체 개요)
├── QUICKSTART.md               # 5분 시작 가이드
├── MODEL_DOWNLOAD_GUIDE.md     # 모델 다운로드 상세 가이드
├── DATASET_UPLOAD_GUIDE.md     # 데이터셋 업로드 상세 가이드
└── requirements.txt            # 필요 패키지
```

---

## 사전 준비

### 1. Hugging Face 계정 및 토큰

1. [Hugging Face](https://huggingface.co/) 계정 생성
2. [Settings → Access Tokens](https://huggingface.co/settings/tokens)에서 **Write** 권한이 있는 토큰 생성
3. 토큰을 환경 변수로 설정:

```bash
export HF_TOKEN="hf_xxxxxxxxxxxxxxxxxxxxxx"
```

또는 `.bashrc` / `.zshrc`에 추가:

```bash
echo 'export HF_TOKEN="hf_xxxxxxxxxxxxxxxxxxxxxx"' >> ~/.bashrc
source ~/.bashrc
```

### 2. 필요한 패키지 설치

```bash
# 모델 업로드/다운로드
pip install huggingface_hub torch pyyaml

# 데이터셋 업로드 (추가)
pip install datasets pillow

# 또는 한 번에
pip install -r requirements.txt
```

---

## 모델 업로드

학습된 SmolVLA 모델을 Hugging Face Hub에 업로드합니다.

### 방법 1: Bash 스크립트 사용 (간편)

1. `upload_model.sh` 파일을 편집하여 repository ID 수정:

```bash
# upload_model.sh 파일에서
REPO_ID="username/smolvla-insertion-vla"  # 본인의 Hugging Face username으로 변경
```

2. 스크립트 실행:

```bash
cd /home/najo/NAS/VLA/Insertion_VLAv4/sub_tasks
bash upload_model.sh
```

### 방법 2: Python 스크립트 직접 사용

```bash
cd /home/najo/NAS/VLA/Insertion_VLAv4/sub_tasks

export PYTHONPATH=/home/najo/NAS/VLA/Insertion_VLAv4/lerobot/src:$PYTHONPATH

python upload_to_huggingface.py \
    --checkpoint /home/najo/NAS/VLA/Insertion_VLAv4/Train/outputs/train/smolvla_vla_insertion_multigpu/checkpoints/checkpoint_step_0016000.pt \
    --repo_id "username/smolvla-insertion-vla" \
    --output_dir outputs/hf_upload \
    --private
```

## 주요 옵션

### 필수 옵션

- `--checkpoint`: 업로드할 checkpoint 파일 경로
- `--repo_id`: Hugging Face repository ID (예: `"username/model-name"`)

### 선택 옵션

- `--output_dir`: 업로드 전 파일을 준비할 로컬 디렉토리 (기본값: `outputs/hf_upload`)
- `--private`: 비공개 repository로 생성 (플래그만 추가)
- `--token`: Hugging Face API 토큰 (환경 변수 `HF_TOKEN` 우선)
- `--no_upload`: 파일만 준비하고 업로드하지 않음 (테스트용)

## 예제

### 1. 비공개 모델 업로드

```bash
python upload_to_huggingface.py \
    --checkpoint checkpoint_step_0016000.pt \
    --repo_id "myusername/smolvla-insertion" \
    --private
```

### 2. 공개 모델 업로드

```bash
python upload_to_huggingface.py \
    --checkpoint checkpoint_step_0016000.pt \
    --repo_id "myusername/smolvla-insertion"
```

### 3. 파일만 준비 (업로드 안 함)

```bash
python upload_to_huggingface.py \
    --checkpoint checkpoint_step_0016000.pt \
    --repo_id "myusername/smolvla-insertion" \
    --no_upload
```

이렇게 하면 `outputs/hf_upload/` 디렉토리에 파일이 준비되며, 확인 후 수동으로 업로드 가능합니다.

### 4. 다른 checkpoint 업로드

```bash
python upload_to_huggingface.py \
    --checkpoint /path/to/checkpoint_step_0032000.pt \
    --repo_id "myusername/smolvla-insertion-step32k" \
    --output_dir outputs/hf_upload_32k
```

## 업로드되는 파일

업로드 시 다음 파일들이 자동으로 생성됩니다:

```
repository/
├── README.md                   # 모델 카드 (자동 생성)
├── config.json                 # Hugging Face 설정
├── training_config.yaml        # 학습 설정 (참고용)
├── model.safetensors          # 모델 가중치 (또는 pytorch_model.bin)
└── config.yaml                # 모델 아키텍처 설정
```

### README.md (Model Card)

자동 생성되는 모델 카드에는 다음 정보가 포함됩니다:

- 모델 설명 및 용도
- 학습 데이터셋 정보
- 학습 설정 (steps, epochs 등)
- 사용 예제 코드
- 라이선스 및 citation 정보

## 업로드 후 모델 사용

Hugging Face에 업로드된 모델은 다음과 같이 사용할 수 있습니다:

```python
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

# 업로드된 모델 로드
policy = SmolVLAPolicy.from_pretrained("username/smolvla-insertion-vla")
policy.eval()

# Inference
with torch.no_grad():
    action = policy.select_action(observation)
```

## 문제 해결

### 1. 토큰 인증 오류

```
Error: Invalid or missing Hugging Face token
```

**해결 방법:**
- `HF_TOKEN` 환경 변수가 올바르게 설정되었는지 확인
- 토큰이 **Write** 권한이 있는지 확인
- `huggingface-cli login` 실행하여 로그인

### 2. Repository 이름 오류

```
Error: Repository name invalid
```

**해결 방법:**
- Repository ID는 `"username/model-name"` 형식이어야 함
- 소문자와 하이픈(-), 언더스코어(_)만 사용
- 예: `"john-doe/smolvla-insertion"`

### 3. 파일 크기 제한

Hugging Face는 파일 크기에 제한이 있습니다:
- 단일 파일: 최대 50GB (LFS 사용 시)
- 일반 파일: 최대 10MB (LFS 없이)

**해결 방법:**
- Git LFS가 자동으로 처리됨
- 큰 파일은 자동으로 LFS로 업로드됨

### 4. 네트워크 오류

```
Error: Connection timeout
```

**해결 방법:**
- 인터넷 연결 확인
- 방화벽 설정 확인
- 재시도

## 고급 사용법

### 다른 checkpoint와 비교를 위해 여러 버전 업로드

```bash
# Step 16000
python upload_to_huggingface.py \
    --checkpoint checkpoint_step_0016000.pt \
    --repo_id "username/smolvla-insertion" \
    --output_dir outputs/hf_upload_16k

# Step 32000
python upload_to_huggingface.py \
    --checkpoint checkpoint_step_0032000.pt \
    --repo_id "username/smolvla-insertion-32k" \
    --output_dir outputs/hf_upload_32k
```

### Model Card 수동 편집

업로드 전에 모델 카드를 확인하고 수정하려면:

```bash
# 파일만 준비
python upload_to_huggingface.py \
    --checkpoint checkpoint.pt \
    --repo_id "username/model-name" \
    --no_upload

# README.md 편집
nano outputs/hf_upload/README.md

# 수동 업로드
huggingface-cli upload username/model-name outputs/hf_upload
```

---

## 모델 다운로드

Hugging Face Hub에서 학습된 모델을 다운로드합니다.

### Quick Start

```bash
# 스크립트 수정
nano download_model.sh

# REPO_ID 변경:
REPO_ID="Najongs/smolvla-insertion-vla"

# 실행
bash download_model.sh
```

### Python으로 직접 다운로드

```bash
python download_model.py \
    --repo_id "Najongs/smolvla-insertion-vla" \
    --output_dir "downloads/model" \
    --save_checkpoint
```

### Python 코드에서 직접 사용

```python
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

# Hub에서 직접 로드
policy = SmolVLAPolicy.from_pretrained("Najongs/smolvla-insertion-vla")
policy.eval()

# Inference
action = policy.select_action(observation)
```

**더 자세한 내용은 [MODEL_DOWNLOAD_GUIDE.md](MODEL_DOWNLOAD_GUIDE.md)를 참조하세요.**

---

## 데이터셋 업로드

VLA Insertion 데이터셋을 Hugging Face Hub에 업로드합니다.

### Blue Point 에피소드 10개 업로드 (예제)

```bash
# 스크립트 수정
nano upload_blue_dataset.sh

# REPO_ID 변경:
REPO_ID="Najongs/vla-insertion-blue-point"

# 실행
bash upload_blue_dataset.sh
```

이렇게 하면 `/home/najo/NAS/VLA/dataset/New_dataset2/Blue_point` 디렉토리의 처음 10개 에피소드가 업로드됩니다.

### 전체 색상 데이터셋 업로드

5가지 색상(Blue, Green, Red, White, Yellow)을 한 번에 업로드:

```bash
# 스크립트 수정
nano upload_all_datasets.sh

# USERNAME 변경:
USERNAME="Najongs"

# 실행
bash upload_all_datasets.sh
```

### Python으로 개별 업로드

```bash
# 특정 디렉토리의 에피소드 업로드
python upload_dataset.py \
    --episode_dir /home/najo/NAS/VLA/dataset/New_dataset2/Blue_point \
    --max_episodes 10 \
    --repo_id "username/vla-insertion-blue" \
    --dataset_name "VLA Insertion - Blue Point" \
    --private

# 특정 에피소드 선택 업로드
python upload_dataset.py \
    --episode_dirs \
        /path/to/episode1 \
        /path/to/episode2 \
        /path/to/episode3 \
    --repo_id "username/dataset" \
    --dataset_name "My Dataset"
```

### 업로드된 데이터셋 사용

```python
from datasets import load_dataset

# 데이터셋 로드
dataset = load_dataset("username/vla-insertion-blue-point")

# 첫 번째 샘플
sample = dataset["train"][0]
print(f"Episode: {sample['episode_id']}")
print(f"Pose: {sample['end_effector_pose']}")
sample['image_View1'].show()  # 이미지 표시
```

**더 자세한 내용은 [DATASET_UPLOAD_GUIDE.md](DATASET_UPLOAD_GUIDE.md)를 참조하세요.**

---

## Quick Start 예제

### 1. 모델 업로드 → 다운로드

```bash
# 1. 모델 업로드
bash upload_model.sh

# 2. 다른 곳에서 다운로드
bash download_model.sh
```

### 2. 데이터셋 업로드 → 학습

```bash
# 1. 데이터셋 업로드
bash upload_blue_dataset.sh

# 2. Python에서 사용
python train.py --dataset_id "username/vla-insertion-blue-point"
```

### 3. 전체 파이프라인

```bash
# 1. 학습
python train.py

# 2. 모델 업로드
bash upload_model.sh

# 3. 데이터셋 업로드
bash upload_all_datasets.sh

# 4. 다른 환경에서 다운로드
bash download_model.sh

# 5. Inference
python inference.py --model downloads/model
```

---

## 참고 자료

- [Hugging Face Hub 문서](https://huggingface.co/docs/hub/index)
- [Hugging Face Datasets 문서](https://huggingface.co/docs/datasets/)
- [LeRobot 문서](https://github.com/huggingface/lerobot)
- [SmolVLA 모델](https://huggingface.co/lerobot/smolvla_base)

## 라이선스

이 코드는 Apache 2.0 라이선스를 따릅니다.

## 문의

문제가 있으면 이슈를 생성하거나 담당자에게 문의하세요.
