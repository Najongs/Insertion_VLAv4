# VLA SmolVLA Evaluation

이 디렉토리에는 학습된 SmolVLA 체크포인트를 평가하기 위한 코드가 포함되어 있습니다.

## 파일 구조

```
Eval/
├── evaluate_smolvla.py      # 메인 평가 스크립트
├── eval_config.yaml          # 평가 설정 파일
└── README.md                 # 이 파일
```

## 사용 방법

### 1. 기본 평가

학습된 체크포인트를 평가하려면:

```bash
cd /home/najo/NAS/VLA/Insertion_VLAv4/Eval

python evaluate_smolvla.py \
    --checkpoint /home/najo/NAS/VLA/Insertion_VLAv4/Train/outputs/train/smolvla_vla_insertion_multigpu/checkpoints/checkpoint_step_0016000.pt \
    --config eval_config.yaml \
    --output_dir outputs/eval
```

### 2. GPU 지정

특정 GPU를 사용하려면:

```bash
CUDA_VISIBLE_DEVICES=0 python evaluate_smolvla.py \
    --checkpoint ../Train/outputs/train/smolvla_vla_insertion_multigpu/checkpoints/checkpoint_step_0016000.pt \
    --config eval_config.yaml
```

### 3. CPU에서 평가

CPU에서 평가하려면:

```bash
python evaluate_smolvla.py \
    --checkpoint ../Train/outputs/train/smolvla_vla_insertion_multigpu/checkpoints/checkpoint_step_0016000.pt \
    --config eval_config.yaml \
    --device cpu
```

### 4. Prediction 저장

예측 결과를 저장하려면:

```bash
python evaluate_smolvla.py \
    --checkpoint ../Train/outputs/train/smolvla_vla_insertion_multigpu/checkpoints/checkpoint_step_0016000.pt \
    --config eval_config.yaml \
    --save_predictions \
    --output_dir outputs/eval_with_predictions
```

## 설정 파일 (eval_config.yaml)

### Dataset 설정

```yaml
dataset:
  root_dir: /home/najo/NAS/VLA/dataset
  episode_dirs: []  # 비어있으면 training episodes 사용
  horizon: 1
  sensor_window_size: 65
  robot_window_size: 100
  action_expert_hz: 10
  use_poses_only: true
```

### Evaluation 설정

```yaml
evaluation:
  batch_size: 8
  num_workers: 4
  save_predictions: false
  per_episode_metrics: true
```

## 평가 지표

스크립트는 다음 지표를 계산합니다:

1. **Loss**: 모델의 전체 손실값
2. **Action MSE**: 예측된 액션과 ground truth 간의 평균 제곱 오차
3. **Position MSE**: 위치 명령 (dx, dy, dz)의 MSE
4. **Rotation MSE**: 회전 명령 (drx, dry, drz)의 MSE
5. **Gripper Accuracy**: 그리퍼 상태 예측 정확도
6. **Per-dimension MSE**: 각 액션 차원별 MSE

## 출력

평가 완료 후 다음이 생성됩니다:

1. **콘솔 출력**: 실시간 평가 진행 상황 및 최종 결과
2. **YAML 파일**: 모든 지표가 포함된 결과 파일
   - 위치: `outputs/eval/eval_results_step_XXXXX.yaml`

### 결과 예시

```
================================================================================
EVALUATION RESULTS
================================================================================
loss                     :   0.123456 ± 0.012345
action_mse               :   0.001234 ± 0.000123
position_mse             :   0.000567 ± 0.000056
rotation_mse             :   0.000890 ± 0.000089
gripper_accuracy         :   0.987654 ± 0.012345

--------------------------------------------------------------------------------
PER-DIMENSION ACTION MSE:
--------------------------------------------------------------------------------
  dx        :   0.000123 ± 0.000012
  dy        :   0.000234 ± 0.000023
  dz        :   0.000345 ± 0.000034
  drx       :   0.000456 ± 0.000045
  dry       :   0.000567 ± 0.000056
  drz       :   0.000678 ± 0.000067
  gripper   :   0.000789 ± 0.000078
================================================================================
```

## 평가 데이터셋 선택

### 옵션 1: Training 데이터 사용 (기본값)

`eval_config.yaml`에서 `episode_dirs`를 비워두면 학습에 사용된 모든 에피소드로 평가합니다.

### 옵션 2: 특정 에피소드 선택

평가용 특정 에피소드를 선택하려면:

```yaml
dataset:
  episode_dirs:
    - New_dataset2/Blue_point/data_collection_20251108_055533
    - New_dataset2/Green_point/data_collection_20251108_053719
    - New_dataset3/Red_point/data_collection_20251110_065722
```

### 옵션 3: Validation Split 생성

학습 데이터의 일부를 validation set으로 분리하려면, training config에서 해당 에피소드를 제거하고 evaluation config에 추가하세요.

## 문제 해결

### 1. CUDA Out of Memory

배치 사이즈를 줄이세요:

```yaml
evaluation:
  batch_size: 4  # 또는 더 작은 값
```

### 2. 느린 데이터 로딩

Worker 수를 조정하세요:

```yaml
evaluation:
  num_workers: 8  # CPU 코어 수에 맞게 조정
```

### 3. 체크포인트 로딩 실패

체크포인트 경로가 정확한지 확인하세요:

```bash
ls -lh /home/najo/NAS/VLA/Insertion_VLAv4/Train/outputs/train/smolvla_vla_insertion_multigpu/checkpoints/
```

## 다음 단계

1. **여러 체크포인트 비교**: 여러 스텝의 체크포인트를 평가하여 학습 곡선 분석
2. **실제 로봇 테스트**: 시뮬레이션이 아닌 실제 로봇에서 정책 배포 및 테스트
3. **시각화**: 예측된 액션과 실제 액션을 시각화하여 모델 동작 분석
4. **오류 분석**: 높은 오차를 보이는 샘플 분석

## 관련 파일

- Training 스크립트: `../Train/train_smolvla.py`
- Training config: `../Train/outputs/train/smolvla_vla_insertion_multigpu/config.yaml`
- 데이터셋: `/home/najo/NAS/VLA/dataset/`
