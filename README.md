# 한국어 MCQ Fine-tuning

LoRA + SFTTrainer 기반 `beomi/gemma-ko-2b` 모델 학습 프로젝트

## 설치

```bash
# 프로덕션 의존성
make install

# 개발 의존성 포함
make dev
```

## 사용법

### 학습

```bash
# 기본 config로 학습
make train

# 커스텀 config로 학습
make train TRAIN_CONFIG=configs/my_experiment.yaml
```

### 추론

```bash
# 기본 config로 추론
make inference

# 커스텀 config로 추론
make inference INFERENCE_CONFIG=configs/my_inference.yaml
```

## Config 파일 구조

### configs/train.yaml

```yaml
model:
  name: "beomi/gemma-ko-2b"

data:
  train_path: "data/train.csv"
  eval_ratio: 0.1

training:
  output_dir: "outputs_gemma"
  max_seq_length: 1024
  batch_size: 1
  epochs: 3
  learning_rate: 2.0e-5
  weight_decay: 0.01
  logging_steps: 1
  seed: 42
  report_to: "none"  # "wandb", "tensorboard", "none"

lora:
  r: 6
  alpha: 8
  dropout: 0.05
```

### configs/inference.yaml

```yaml
model:
  checkpoint_path: "outputs_gemma/checkpoint-xxx"

data:
  test_path: "data/test.csv"

output:
  path: "output.csv"
```

## 새 옵션 추가 방법

새로운 학습 옵션을 추가하려면 3곳을 수정합니다.

### 예시: `gradient_accumulation_steps` 옵션 추가

#### 1. config_loader.py에 필드 추가

```python
# src/utils/config_loader.py

@dataclass
class TrainConfig:
    # ... 기존 필드들 ...

    # 새 필드 추가
    gradient_accumulation_steps: int = 1
```

#### 2. config_loader.py에 YAML 매핑 추가

```python
# src/utils/config_loader.py

_yaml_key_mapping: ClassVar[dict[str, str]] = {
    # ... 기존 매핑들 ...

    # 새 매핑 추가
    "training_gradient_accumulation_steps": "gradient_accumulation_steps",
}
```

#### 3. configs/train.yaml에 값 추가

```yaml
training:
  # ... 기존 값들 ...

  # 새 값 추가
  gradient_accumulation_steps: 4
```

#### 4. train.py에서 사용

```python
# src/trainers/train.py

sft_config = SFTConfig(
    # ... 기존 설정들 ...

    # 새 옵션 사용
    gradient_accumulation_steps=config.gradient_accumulation_steps,
)
```

## 코드 품질

```bash
# 린트 검사
make lint

# 코드 포맷팅
make format

# 타입 검사
make typecheck

# 전체 검사
make check
```

## 정리

```bash
# Python 캐시 삭제
make clean

# 툴 캐시 삭제
make clean-cache

# 모든 캐시 + 출력 삭제
make clean-all
```
