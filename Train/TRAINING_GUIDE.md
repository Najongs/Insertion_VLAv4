# SmolVLA Training Guide - New Dataset

새로운 HDF5 데이터셋으로 SmolVLA 모델을 학습하는 가이드입니다.

## 📁 파일 구조

```
Train/
├── train_smolvla_new_dataset.py      # 학습 메인 스크립트
├── hdf5_lerobot_adapter.py           # HDF5 데이터 어댑터
├── train_config_new_dataset.yaml     # 학습 설정 파일
├── train_single_gpu.sh               # 단일 GPU 실행 스크립트
├── train_multi_gpu.sh                # 멀티 GPU 실행 스크립트
├── train_background.sh               # 백그라운드 실행 스크립트
├── stop_training.sh                  # 학습 중단 스크립트
└── TRAINING_GUIDE.md                 # 이 파일
```

## 🚀 빠른 시작

### 1. 단일 GPU 학습

```bash
cd /home/irom/NAS/VLA/Insertion_VLAv4/Train
bash train_single_gpu.sh
```

### 2. 멀티 GPU 학습 (현재 시스템: GPU 1개)

```bash
# 모든 GPU 사용
bash train_multi_gpu.sh

# 특정 GPU만 사용
CUDA_VISIBLE_DEVICES=0,1 bash train_multi_gpu.sh
```

### 3. 백그라운드 학습 (장시간 학습)

```bash
# 단일 GPU
bash train_background.sh single

# 멀티 GPU
bash train_background.sh multi

# 로그 확인
tail -f outputs/train/smolvla_new_dataset/logs/train_*.log
```

### 4. 학습 중단

```bash
bash stop_training.sh
```

## ⚙️ 설정 파일 (train_config_new_dataset.yaml)

### 주요 설정 항목

```yaml
# 데이터셋
dataset:
  root_dir: "/home/irom/NAS/VLA/Insertion_VLAv4/New_dataset/collected_data"
  horizon: 1                    # 액션 예측 호라이즌
  use_ee_pose: true            # EE pose 사용 (6차원)
  use_qpos: false              # Joint position 사용 안 함

# 정책 모델
policy:
  pretrained_model_path: "/home/irom/NAS/VLA/Insertion_VLAv4/sub_tasks/downloads/model"
  freeze_vision_encoder: true   # 비전 인코더 동결 (빠른 학습)
  train_expert_only: true       # Expert만 학습 (VLM 동결)
  use_multi_gpu: false          # 멀티 GPU 사용 (쉘 스크립트에서 자동 설정)

# 학습
training:
  steps: 50000                  # 학습 스텝
  batch_size: 8                 # 배치 크기
  log_freq: 100                 # 로그 출력 빈도
  save_freq: 2000               # 체크포인트 저장 빈도

# 옵티마이저
optimizer:
  lr: 1e-4                      # 학습률
```

## 💻 멀티 GPU 설정

### DataParallel 사용

현재 코드는 PyTorch의 `DataParallel`을 사용합니다:

```python
# train_smolvla_new_dataset.py에서 자동으로 처리됨
if use_multi_gpu and torch.cuda.device_count() > 1:
    policy = nn.DataParallel(policy)
```

### 효과적인 배치 크기

- **단일 GPU**: 배치 크기 = 설정 값 (예: 8)
- **멀티 GPU**: 효과적인 배치 크기 = 설정 값 × GPU 개수
  - 예: 배치 크기 8, GPU 2개 → 효과적인 배치 크기 16

### GPU 메모리 고려사항

현재 시스템: RTX 5080 (16GB VRAM)

| 배치 크기 | 메모리 사용량 (예상) | 권장 |
|----------|-------------------|------|
| 4        | ~8GB             | ✅ 안전 |
| 8        | ~12GB            | ✅ 권장 |
| 16       | ~20GB            | ❌ OOM 위험 |

## 📊 학습 모니터링

### 1. 로그 확인

```bash
# 실시간 로그
tail -f outputs/train/smolvla_new_dataset/logs/train_*.log

# 최근 로그 확인
ls -lht outputs/train/smolvla_new_dataset/logs/
```

### 2. GPU 사용량 확인

```bash
# 1초마다 업데이트
watch -n 1 nvidia-smi

# GPU 메모리 사용량만 확인
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv
```

### 3. 학습 진행 확인

```bash
# 프로세스 확인
ps aux | grep train_smolvla

# PID 파일 확인
cat outputs/train/smolvla_new_dataset/train.pid
```

### 4. 체크포인트 확인

```bash
ls -lh outputs/train/smolvla_new_dataset/checkpoints/
```

## 🔧 고급 옵션

### 명령줄 옵션으로 설정 변경

```bash
# 배치 크기 변경
python train_smolvla_new_dataset.py --config train_config_new_dataset.yaml --batch_size 4

# 학습 스텝 변경
python train_smolvla_new_dataset.py --config train_config_new_dataset.yaml --steps 100000

# Learning rate 변경
python train_smolvla_new_dataset.py --config train_config_new_dataset.yaml --lr 5e-5

# 출력 디렉토리 변경
python train_smolvla_new_dataset.py --config train_config_new_dataset.yaml --output_dir outputs/my_training

# 여러 옵션 동시 사용
python train_smolvla_new_dataset.py \
    --config train_config_new_dataset.yaml \
    --batch_size 4 \
    --steps 100000 \
    --lr 5e-5
```

### 특정 GPU 선택

```bash
# GPU 0만 사용
CUDA_VISIBLE_DEVICES=0 bash train_single_gpu.sh

# GPU 1만 사용
CUDA_VISIBLE_DEVICES=1 bash train_single_gpu.sh

# GPU 0,1,2 사용
CUDA_VISIBLE_DEVICES=0,1,2 bash train_multi_gpu.sh
```

## 📈 학습 결과

### 체크포인트

```
outputs/train/smolvla_new_dataset/checkpoints/
├── checkpoint_step_0002000.pt
├── checkpoint_step_0004000.pt
├── checkpoint_step_0006000.pt
├── ...
└── checkpoint_latest.pt
```

### 최종 모델

```
outputs/train/smolvla_new_dataset/final_model/
├── config.json
├── pytorch_model.bin
└── ...
```

## 🐛 문제 해결

### OOM (Out of Memory) 에러

```bash
# 배치 크기 줄이기
python train_smolvla_new_dataset.py --config train_config_new_dataset.yaml --batch_size 4

# 또는 설정 파일 수정
vim train_config_new_dataset.yaml
# training.batch_size를 4로 변경
```

### 학습이 느린 경우

```yaml
# train_config_new_dataset.yaml 수정
training:
  num_workers: 8  # 데이터 로딩 워커 증가 (CPU 코어 수에 맞게 조정)
```

### 체크포인트에서 재개

```python
# 수동으로 체크포인트 로드 (train_smolvla_new_dataset.py 수정 필요)
checkpoint = torch.load("outputs/train/smolvla_new_dataset/checkpoints/checkpoint_latest.pt")
policy.load_state_dict(checkpoint["policy_state_dict"])
optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
start_step = checkpoint["step"]
```

## 📝 학습 팁

1. **처음에는 짧게 테스트**
   ```bash
   python train_smolvla_new_dataset.py --config train_config_new_dataset.yaml --steps 1000
   ```

2. **데이터 확인**
   ```bash
   python hdf5_lerobot_adapter.py
   ```

3. **GPU 메모리 모니터링**
   ```bash
   watch -n 1 nvidia-smi
   ```

4. **백그라운드 실행 추천** (장시간 학습)
   ```bash
   bash train_background.sh single
   ```

5. **정기적으로 체크포인트 확인**
   - 기본: 2000 스텝마다 저장
   - 필요시 `save_freq` 조정

## 📞 도움말

문제가 발생하면:

1. 로그 파일 확인: `outputs/train/smolvla_new_dataset/logs/`
2. GPU 상태 확인: `nvidia-smi`
3. 설정 파일 확인: `train_config_new_dataset.yaml`
4. 데이터 확인: `python hdf5_lerobot_adapter.py`

---

**Created**: 2025-12-23
**Dataset**: New HDF5 VLA Dataset (18 episodes)
**Model**: SmolVLA (downloaded from Hugging Face)
