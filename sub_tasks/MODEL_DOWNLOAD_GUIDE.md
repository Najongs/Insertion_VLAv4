# 모델 다운로드 가이드

Hugging Face Hub에서 학습된 SmolVLA 모델을 다운로드하는 방법입니다.

## 📋 목차

1. [Quick Start](#quick-start)
2. [Python에서 직접 사용](#python에서-직접-사용)
3. [Checkpoint 형식으로 다운로드](#checkpoint-형식으로-다운로드)
4. [다운로드 옵션](#다운로드-옵션)

---

## Quick Start

### 방법 1: Bash 스크립트 사용 (가장 간편)

```bash
# 1. 스크립트 수정
nano download_model.sh

# REPO_ID를 다운로드할 모델로 변경:
REPO_ID="Najongs/smolvla-insertion-vla"

# 2. 실행
bash download_model.sh
```

모델이 `downloads/model/` 디렉토리에 다운로드됩니다.

### 방법 2: Python 스크립트 직접 사용

```bash
python download_model.py \
    --repo_id "Najongs/smolvla-insertion-vla" \
    --output_dir "downloads/my_model" \
    --save_checkpoint
```

---

## Python에서 직접 사용

가장 간단한 방법은 Python 코드에서 직접 로드하는 것입니다:

```python
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
import torch

# Hugging Face Hub에서 모델 로드
model_id = "Najongs/smolvla-insertion-vla"
policy = SmolVLAPolicy.from_pretrained(model_id)
policy.eval()

# GPU로 이동
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
policy.to(device)

# Inference
with torch.no_grad():
    action = policy.select_action(observation)

print(f"Predicted action: {action}")
```

이 방법은 모델을 자동으로 캐시에 다운로드합니다 (보통 `~/.cache/huggingface/hub/`).

---

## Checkpoint 형식으로 다운로드

Training 스크립트와 호환되는 checkpoint 형식(.pt)으로도 다운로드할 수 있습니다.

### 방법 1: 스크립트 사용

```bash
python download_model.py \
    --repo_id "username/model-name" \
    --output_dir "downloads/model" \
    --save_checkpoint \
    --checkpoint_path "checkpoints/downloaded_model.pt"
```

### 방법 2: Python 코드

```python
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
import torch

# 모델 로드
policy = SmolVLAPolicy.from_pretrained("username/model-name")

# Checkpoint로 저장
checkpoint = {
    "policy_state_dict": policy.state_dict(),
    "config": {
        "policy": {
            "pretrained_model_id": "username/model-name",
            "n_obs_steps": policy.config.n_obs_steps,
            "chunk_size": policy.config.chunk_size,
            "n_action_steps": policy.config.n_action_steps,
        }
    },
    "step": getattr(policy.config, "training_step", 0),
    "epoch": getattr(policy.config, "training_epoch", 0),
}

torch.save(checkpoint, "downloaded_model.pt")
print("Checkpoint saved!")
```

---

## 다운로드 옵션

### 기본 다운로드

```bash
python download_model.py \
    --repo_id "username/model-name"
```

다운로드 위치: `downloads/model/`

### 출력 디렉토리 지정

```bash
python download_model.py \
    --repo_id "username/model-name" \
    --output_dir "my_models/smolvla_v1"
```

### Checkpoint 형식으로 저장

```bash
python download_model.py \
    --repo_id "username/model-name" \
    --save_checkpoint
```

모델과 checkpoint 모두 저장됨:
- `downloads/model/` - Hugging Face 형식
- `downloads/model/checkpoint.pt` - PyTorch checkpoint

### 특정 버전/브랜치 다운로드

```bash
python download_model.py \
    --repo_id "username/model-name" \
    --revision "main"  # 또는 특정 commit hash
```

### 비공개 모델 다운로드

```bash
# 환경 변수로 토큰 설정
export HF_TOKEN="hf_your_token_here"

python download_model.py \
    --repo_id "username/private-model" \
    --token "$HF_TOKEN"
```

또는 토큰을 직접 전달:

```bash
python download_model.py \
    --repo_id "username/private-model" \
    --token "hf_your_token_here"
```

---

## 모델 사용 예제

### 1. Inference 스크립트에서 사용

```python
# inference_script.py
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
import torch

# 다운로드한 로컬 모델 사용
model_path = "downloads/model"
policy = SmolVLAPolicy.from_pretrained(model_path)
policy.eval()

# 또는 Hub에서 직접 로드
policy = SmolVLAPolicy.from_pretrained("username/model-name")
policy.eval()

# Inference
device = torch.device("cuda")
policy.to(device)

observation = {
    "observation.images.camera1": image_tensor,
    "observation.state": state_tensor,
    "task": "Insert needle into Red point",
    "robot_type": "meca500",
}

with torch.no_grad():
    action = policy.select_action(observation)
```

### 2. Fine-tuning을 위한 Checkpoint 로드

```python
import torch
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

# Checkpoint 로드
checkpoint = torch.load("downloads/model/checkpoint.pt")
policy_state_dict = checkpoint["policy_state_dict"]
config = checkpoint["config"]["policy"]

# 모델 생성 및 가중치 로드
policy = SmolVLAPolicy.from_pretrained(config["pretrained_model_id"])
policy.load_state_dict(policy_state_dict, strict=False)

# Fine-tuning 시작
optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-5)
# ... training code
```

### 3. 모델 비교

여러 버전의 모델을 다운로드하여 비교:

```bash
# 버전 1 다운로드
python download_model.py \
    --repo_id "username/model-v1" \
    --output_dir "models/v1" \
    --save_checkpoint

# 버전 2 다운로드
python download_model.py \
    --repo_id "username/model-v2" \
    --output_dir "models/v2" \
    --save_checkpoint

# 평가 스크립트에서 비교
python compare_models.py \
    --models models/v1 models/v2 \
    --dataset eval_data
```

---

## 다운로드한 모델 정보 확인

```python
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

# 모델 로드
policy = SmolVLAPolicy.from_pretrained("downloads/model")

# Config 정보
print(f"Observation steps: {policy.config.n_obs_steps}")
print(f"Chunk size: {policy.config.chunk_size}")
print(f"Action steps: {policy.config.n_action_steps}")

# 모델 크기
total_params = sum(p.numel() for p in policy.parameters())
print(f"Total parameters: {total_params:,}")

# Training 정보 (있는 경우)
if hasattr(policy.config, "training_step"):
    print(f"Training step: {policy.config.training_step}")
if hasattr(policy.config, "training_epoch"):
    print(f"Training epoch: {policy.config.training_epoch}")
```

---

## 캐시 관리

Hugging Face Hub는 모델을 자동으로 캐시합니다.

### 캐시 위치 확인

```bash
echo $HF_HOME
# 기본값: ~/.cache/huggingface/
```

### 캐시 삭제

```bash
# 특정 모델 캐시 삭제
rm -rf ~/.cache/huggingface/hub/models--username--model-name

# 전체 캐시 삭제 (주의!)
rm -rf ~/.cache/huggingface/hub/
```

### 캐시 디렉토리 변경

```bash
export HF_HOME="/path/to/custom/cache"
python download_model.py --repo_id "username/model"
```

---

## 문제 해결

### Q: "Repository not found" 오류

**원인:** Repository가 존재하지 않거나 비공개

**해결:**
1. Repository ID가 정확한지 확인
2. 비공개 모델인 경우 토큰 설정:
   ```bash
   export HF_TOKEN="hf_your_token"
   ```

### Q: 다운로드가 중단됨

**원인:** 네트워크 문제

**해결:**
- 재시도 (캐시가 있어서 이어서 다운로드됨)
- 안정적인 네트워크 사용

### Q: "Out of disk space" 오류

**원인:** 디스크 공간 부족

**해결:**
- 디스크 공간 확인: `df -h`
- 불필요한 캐시 삭제
- 다른 디스크로 캐시 위치 변경

### Q: Git LFS 오류

**원인:** Git LFS가 설치되지 않음

**해결:**
```bash
# Ubuntu/Debian
sudo apt-get install git-lfs

# macOS
brew install git-lfs

# 설치 후
git lfs install
```

---

## 오프라인 사용

모델을 다운로드한 후 오프라인에서 사용:

```python
# 1. 온라인에서 모델 다운로드
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
policy = SmolVLAPolicy.from_pretrained("username/model")
policy.save_pretrained("my_offline_model")

# 2. 오프라인에서 로컬 모델 사용
policy = SmolVLAPolicy.from_pretrained("my_offline_model", local_files_only=True)
```

---

## 모델 공유

다운로드한 모델을 다른 사람과 공유:

### 방법 1: 로컬 파일 공유

```bash
# 모델 디렉토리 압축
tar -czf smolvla_model.tar.gz downloads/model/

# 전송 후 압축 해제
tar -xzf smolvla_model.tar.gz

# 사용
python -c "
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
policy = SmolVLAPolicy.from_pretrained('downloads/model')
"
```

### 방법 2: 재업로드

```bash
# 다운로드한 모델을 다른 Repository에 업로드
python upload_to_huggingface.py \
    --checkpoint downloads/model/checkpoint.pt \
    --repo_id "new-username/model-copy"
```

---

## 참고 자료

- [Hugging Face Hub 문서](https://huggingface.co/docs/hub/)
- [Transformers 모델 로딩](https://huggingface.co/docs/transformers/main/model_sharing)
- [LeRobot 문서](https://github.com/huggingface/lerobot)

---

## 다음 단계

모델 다운로드 후:

1. Inference 스크립트에 통합
2. 로봇 제어 시스템과 연결
3. Fine-tuning 진행
4. 성능 평가

더 자세한 내용은 [README.md](README.md)를 참조하세요.
