# SmolVLA Simulation Training Guide

## Overview

이 가이드는 시뮬레이션 데이터로 SmolVLA 모델을 학습하는 방법을 설명합니다.

**시뮬레이션 데이터셋**: Eye_trocar_sim (352 episodes)
- 위치: `/home/najo/NAS/VLA/dataset/New_dataset/collected_data/Eye_trocar/Eye_trocar_sim`
- 데이터 수집: `Sim/Save_dataset_arg.py` 사용
- Domain Randomization (DR) 적용됨

## Files

### 학습 관련 파일
- `train_smolvla_sim.py` - 시뮬레이션 데이터 학습 스크립트
- `train_config_smolvla_sim.yaml` - 시뮬레이션 학습 설정
- `train_smolvla_sim.sh` - 학습 실행 스크립트

### 데이터 통계 계산
- `compute_dataset_stats_sim.sh` - 시뮬레이션 데이터 통계 계산 스크립트

## Quick Start

### 1. 데이터셋 통계 계산 (첫 실행 시 필수)

```bash
cd /home/najo/NAS/VLA/Insertion_VLAv4/Train
bash compute_dataset_stats_sim.sh
```

이 스크립트는:
- 352개 시뮬레이션 에피소드 분석
- Action/State의 평균, 표준편차, 최소값, 최대값 계산
- `dataset_stats_sim.yaml` 생성

### 2. 학습 시작

```bash
bash train_smolvla_sim.sh
```

이 스크립트는:
- 5개 GPU 사용 (DDP)
- `train_config_smolvla_sim.yaml` 설정 사용
- Validation split 자동 생성 (20 episodes)

## Training Configuration

### 주요 설정 (`train_config_smolvla_sim.yaml`)

```yaml
dataset:
  root_dir: "/home/najo/NAS/VLA/dataset/New_dataset/collected_data/Eye_trocar/Eye_trocar_sim"
  horizon: 50
  task_instruction: "Insert needle into eye trocar in simulation"

  # 데이터 증강: 비활성화 (DR이 이미 적용됨)
  augment: false

training:
  steps: 30000            # 학습 스텝
  batch_size: 16          # GPU당 배치 크기
  gradient_accumulation_steps: 1

validation:
  enable: true
  num_episodes: 20        # 검증용 에피소드 (352개 중)
  val_freq: 2000          # 검증 주기

optimizer:
  lr: 0.0001              # 학습률
  betas: [0.9, 0.95]

scheduler:
  num_warmup_steps: 1000
  num_decay_steps: 30000
```

## Simulation Data Details

### Domain Randomization (DR) 적용 항목

데이터 수집 시 이미 적용된 DR:
1. **배경 랜덤화**: random_backgrounds 폴더의 이미지 사용
2. **조명 랜덤화**: 조명 위치 ±0.5m 변동
3. **카메라 지터**: 위치 ±2mm, 각도 ±0.02 rad
4. **물체 색상 랜덤화**: RGB 색상 ±0.1 변동
5. **이미지 노이즈**: Gaussian noise (std=5)
6. **센서 노이즈**: qpos 노이즈 (std=0.001)
7. **동역학 랜덤화**: 관절 damping ±50% 변동

### 카메라

시뮬레이션 데이터:
- `side_camera` → LeRobot `camera1`
- `tool_camera` → LeRobot `camera2`
- `top_camera` → LeRobot `camera3`

### 상태/액션

- **State**: end-effector pose (6 dims) - x, y, z, rx, ry, rz
- **Action**: joint positions (6 dims) - q1, q2, q3, q4, q5, q6
- **Horizon**: 50 steps

## Expected Training Behavior

### 시뮬레이션 학습 특징

✅ **장점**:
- 빠른 수렴 (데이터 일관성 높음)
- 낮은 학습 loss
- Validation loss ≈ Training loss (overfitting 적음)

⚠️ **주의사항**:
- Sim2Real gap 존재 가능
- 실제 로봇 테스트 시 성능 차이 발생 가능
- 필요시 실제 데이터로 fine-tuning 권장

### 예상 Loss 범위

- **초기** (step 0-1000): loss > 1.0
- **중간** (step 1000-10000): loss 0.1-0.5
- **후반** (step 10000+): loss < 0.1
- **수렴** (step 20000+): loss < 0.05

## Outputs

학습 중 생성되는 파일들:

```
outputs/train/smolvla_needle_insertion_sim/
├── checkpoints/
│   ├── checkpoint_step_2000.pt
│   ├── checkpoint_step_4000.pt
│   ├── ...
│   ├── checkpoint_latest.pt      # 최종 체크포인트
│   └── final_hf/                 # HuggingFace 형식
├── loss_spikes.txt               # Loss spike 로그
├── loss_spikes.csv               # Loss spike CSV
└── validation_episodes.txt       # 검증 에피소드 정보
```

## Monitoring Training

### W&B (Weights & Biases)

설정에서 W&B 활성화:
```yaml
wandb:
  enable: true
  project: "smolvla-meca500-insertion-sim"
```

W&B에서 모니터링 가능한 메트릭:
- `train/loss`, `train/action_loss`, `train/temporal_loss`
- `val/loss`, `val/action_loss`
- `train/learning_rate`, `train/grad_norm`
- `system/vram_allocated_gb`, `system/ram_gb`
- `spike/loss`, `spike/ratio` (loss spike detection)

### Console Output

터미널에서 실시간 로그 확인:
```
[ 10.0%] Step 3000/30000 | Loss: 0.1234 (Action: 0.1200, Temporal: 0.0034) | ...
[VAL] Step 2000 | Loss: 0.1156 | Action Loss: 0.1120
⚠️  LOSS SPIKE at step 1234: 0.5678 (avg: 0.1234, 4.6x) | Episodes: [45, 67, 89]
```

## Resuming Training

체크포인트에서 재개:
```bash
python train_smolvla_sim.py \
    --config train_config_smolvla_sim.yaml \
    --resume outputs/train/smolvla_needle_insertion_sim/checkpoints/checkpoint_step_10000.pt
```

스케줄러 리셋 (warmup 다시 시작):
```bash
python train_smolvla_sim.py \
    --config train_config_smolvla_sim.yaml \
    --resume outputs/train/smolvla_needle_insertion_sim/checkpoints/checkpoint_step_10000.pt \
    --reset_scheduler
```

## Sim2Real Transfer Strategy

### 1단계: 시뮬레이션 학습 (현재)
```bash
bash train_smolvla_sim.sh
```
- 30,000 steps
- Validation: 20 episodes

### 2단계: 실제 로봇 평가
- 학습된 모델을 실제 로봇에 배포
- 성능 측정 (성공률, 정확도)

### 3단계: Fine-tuning (필요시)
만약 sim2real gap이 크다면:
```bash
python train_smolvla.py \
    --config train_config_smolvla_normalized.yaml \
    --resume outputs/train/smolvla_needle_insertion_sim/checkpoints/checkpoint_latest.pt \
    --steps 5000  # 적은 스텝으로 fine-tuning
    --lr 0.00001  # 낮은 학습률
```

## Troubleshooting

### 1. "No HDF5 files found" 에러
```bash
# 데이터셋 경로 확인
ls -la /home/najo/NAS/VLA/dataset/New_dataset/collected_data/Eye_trocar/Eye_trocar_sim/
```

### 2. "dataset_stats_sim.yaml not found" 경고
```bash
# 통계 파일 생성
bash compute_dataset_stats_sim.sh
```

### 3. CUDA Out of Memory
`train_config_smolvla_sim.yaml`에서 batch_size 줄이기:
```yaml
training:
  batch_size: 8  # 16 → 8로 감소
```

### 4. Loss가 수렴하지 않음
- Learning rate 조정: `optimizer.lr: 0.00005`
- Warmup steps 증가: `scheduler.num_warmup_steps: 2000`
- Temporal loss 가중치 조정: `training.lambda_temporal: 0.05`

## Comparison with Real Data Training

| 항목 | 실제 데이터 | 시뮬레이션 데이터 |
|------|------------|-----------------|
| 에피소드 수 | ~700 | 352 |
| 데이터 증강 | O (brightness, contrast, etc.) | X (DR 적용됨) |
| 학습 스텝 | 50,000 | 30,000 |
| 수렴 속도 | 느림 | 빠름 |
| Loss 범위 | 0.1-0.3 | 0.05-0.1 |
| 실제 성능 | 높음 | Sim2Real gap 존재 |

## Next Steps

1. ✅ 시뮬레이션 데이터 통계 계산
2. ✅ 시뮬레이션 학습 시작
3. ⏳ 학습 모니터링 (W&B)
4. ⏳ 체크포인트 평가 (실제 로봇)
5. ⏳ Fine-tuning (필요시)

## Questions?

- 학습 설정 변경: `train_config_smolvla_sim.yaml` 수정
- 데이터 경로 변경: yaml의 `dataset.root_dir` 수정
- GPU 수 변경: `train_smolvla_sim.sh`의 `NUM_GPUS` 수정
