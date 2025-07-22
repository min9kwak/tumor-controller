# YAML Config 시스템 사용 예시

## 1. 기본 사용법

### LDM 파인튜닝 실행
```bash
# 기본 YAML 파일 사용
python finetune_ldm.py

# 특정 YAML 파일 사용
python finetune_ldm.py --config yaml/ldm.yaml

# YAML 설정을 command line에서 override
python finetune_ldm.py --config yaml/ldm.yaml --learning_rate 2e-5 --train_batch_size 8
```

### ControlNet 파인튜닝 실행 (추후 구현시)
```bash
# ControlNet YAML 파일 사용
python finetune_controlnet.py --config yaml/controlnet.yaml
```

## 2. 다양한 모달리티로 실험

### T1 모달리티
```yaml
# yaml/experiments/t1.yaml
modality: "t1"
learning_rate: 1e-5
train_batch_size: 4
num_train_epochs: 10
```

### T2 모달리티
```yaml
# yaml/experiments/t2.yaml
modality: "t2"
learning_rate: 1e-5
train_batch_size: 4
num_train_epochs: 10
```

### FLAIR 모달리티
```yaml
# yaml/experiments/flair.yaml
modality: "flair"
learning_rate: 1e-5
train_batch_size: 4
num_train_epochs: 10
```

## 3. 실험별 설정 예시

### 고해상도 실험
```yaml
# yaml/experiments/high_res.yaml
resolution: 1024
train_batch_size: 2  # 메모리 제약으로 배치 크기 감소
gradient_accumulation_steps: 2
mixed_precision: "fp16"
```

### 빠른 테스트
```yaml
# yaml/experiments/quick_test.yaml
num_train_epochs: 1
max_train_steps: 100
validation_epochs: 1
checkpointing_steps: 50
```

### 대용량 배치 실험
```yaml
# yaml/experiments/large_batch.yaml
train_batch_size: 16
gradient_accumulation_steps: 4
learning_rate: 2e-5  # 큰 배치에 맞춰 learning rate 조정
```

## 4. 실행 예시

```bash
# 다양한 실험 실행
python finetune_ldm.py --config yaml/experiments/t1.yaml
python finetune_ldm.py --config yaml/experiments/t2.yaml
python finetune_ldm.py --config yaml/experiments/high_res.yaml

# 특정 설정만 override
python finetune_ldm.py --config yaml/ldm.yaml --modality t2 --learning_rate 2e-5

# 여러 설정 동시 override
python finetune_ldm.py --config yaml/ldm.yaml \
    --modality flair \
    --train_batch_size 8 \
    --num_train_epochs 20 \
    --output_dir outputs/flair_experiment
```

## 5. 주요 설정 설명

### 학습 파라미터
- `train_batch_size`: 배치 크기 (메모리에 따라 조정)
- `learning_rate`: 학습률 (LDM: 1e-5, ControlNet: 1e-4)
- `num_train_epochs`: 학습 epoch 수
- `gradient_accumulation_steps`: 그래디언트 누적 단계

### 모델 파라미터
- `modality`: MRI 모달리티 (t1, t2, flair, t1ce)
- `resolution`: 이미지 해상도
- `mixed_precision`: 혼합 정밀도 (fp16, bf16)

### 검증 파라미터
- `validation_epochs`: 검증 주기
- `num_validation_images`: 검증 이미지 수
- `tumor_size`: 검증용 종양 크기 (small, medium, large) 