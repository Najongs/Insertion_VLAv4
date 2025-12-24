# Training Script Changes - VLM Full Training

## 🎯 문제점
이전 학습에서는 **Expert만 학습**되었고 **VLM(Vision-Language Model)이 freeze**되어 있었습니다.
- 학습된 파라미터: 155/500 (31%)
- VLM이 "나사 구멍 vs 빨간 마커"를 구별하지 못함
- Pretrained VLM은 일반 이미지로만 학습되어 도메인 특화 feature를 학습하지 못함

## ✅ 해결 방법

### 1. **모든 파라미터 명시적 Unfreeze** (train_smolvla_new_dataset.py)
```python
# Policy 로드 후 모든 파라미터를 먼저 unfreeze
for param in policy.parameters():
    param.requires_grad = True
```

**변경 전:**
- Pretrained model 로드 시 일부 파라미터가 이미 freeze되어 있었음
- Freeze 설정이 제대로 적용되지 않음

**변경 후:**
- 모든 파라미터를 명시적으로 trainable로 설정
- Freeze 설정의 default를 `False`로 변경
- 학습 시작 전/후 trainable params 카운트 로그 추가

### 2. **Color Augmentation 제거** (train_config_new_dataset.yaml)
```yaml
augment_saturation: 0.0  # 비활성화 (was 0.15)
augment_hue: 0.0         # 비활성화 (was 0.03) - 색상 학습에 중요!
```

**이유:**
- Hue augmentation이 빨간색을 다른 색으로 변환
- VLM이 색상 기반 feature를 학습하지 못함
- "빨간 마커" vs "회색 나사 구멍" 구별 불가능

### 3. **학습 기간 증가**
```yaml
steps: 170830  # 10 epochs (이전: 85415 = 5 epochs)
```

**이유:**
- VLM fine-tuning은 Expert만 학습하는 것보다 오래 걸림
- Pretrained weights의 관성 극복 필요
- 새로운 도메인 학습 시간 확보

### 4. **Trainable Parameters Logging**
```python
logger.info(f"After unfreezing: {initial_trainable:,} / {initial_total:,} params trainable")
logger.info(f"After freeze settings: {final_trainable:,} / {initial_total:,} params trainable")
```

**효과:**
- 학습 시작 시 몇 개의 파라미터가 실제로 학습되는지 확인 가능
- 이전 학습에서는 이 정보가 없어서 문제를 발견하기 어려웠음

## 📊 예상 결과

### 이전 학습:
```
VLM (vision-language model): 345 params  ❌ FROZEN
LM Expert (action expert): 145 params    ✓ 학습됨
Action/State projection: 10 params       ✓ 학습됨
────────────────────────────────────────
Total trained: 155/500 params (31%)
```

### 새로운 학습:
```
VLM (vision-language model): 345 params  ✓ 학습됨
LM Expert (action expert): 145 params    ✓ 학습됨
Action/State projection: 10 params       ✓ 학습됨
────────────────────────────────────────
Total trained: 500/500 params (100%)
```

## 🚀 사용 방법

### Multi-GPU 학습:
```bash
bash train_multi_gpu.sh
```

### Single GPU 학습:
```bash
bash train_single_gpu.sh
```

### 학습 시작 시 확인사항:
1. 로그에서 "After unfreezing" 메시지 확인
2. Trainable params가 500개 근처인지 확인 (155가 아님!)
3. "VLM frozen" 메시지가 **출력되지 않아야** 함

## 📝 Config 요약

### Policy Settings:
```yaml
freeze_vision_encoder: false  # Vision encoder 학습
train_expert_only: false      # VLM 전체 학습
train_state_proj: false       # 모든 레이어 학습
```

### Augmentation Settings:
```yaml
augment_brightness: 0.10     # ±10% (reduced from 15%)
augment_contrast: 0.10       # ±10% (reduced from 15%)
augment_saturation: 0.0      # DISABLED (was 15%)
augment_hue: 0.0             # DISABLED (was 3%) - critical!
```

### Training Settings:
```yaml
steps: 170830                # 10 epochs
batch_size: 1                # MUST be 1
lr: 0.0001                   # Learning rate
```

## 🎓 공식 LeRobot 패턴 적용

이 수정사항은 공식 LeRobot 학습 스크립트의 패턴을 따릅니다:
1. **명시적 파라미터 관리**: 모든 파라미터를 먼저 unfreeze
2. **Logging**: Trainable params 카운트 로그
3. **정확한 freeze 제어**: Default를 False로 설정

## ⚠️ 주의사항

1. **학습 시간**: 10 epochs는 약 **48시간** 소요 (이전의 2배)
2. **메모리**: VLM 전체 학습으로 인해 메모리 사용량 증가 가능
3. **첫 학습 시**: 로그를 주의 깊게 확인하여 실제로 500개 파라미터가 학습되는지 확인

## 🔍 학습 진행 중 확인

```bash
# 로그에서 trainable params 확인
grep "trainable" outputs/train/smolvla_new_dataset_multigpu/logs/train_*.log

# 체크포인트에서 확인
python3 -c "
import torch
ckpt = torch.load('outputs/train/smolvla_new_dataset_multigpu/checkpoints/checkpoint_latest.pt', map_location='cpu')
optimizer_state = ckpt.get('optimizer_state_dict', {})
print(f'Parameters in optimizer: {len(optimizer_state.get(\"state\", {}))}')
"
```

기대 출력: **500개 근처** (155가 아님!)
