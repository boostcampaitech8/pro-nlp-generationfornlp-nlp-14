# unsloth 도입 가이드 (재현성 문서)
# unsloth 도입 가이드 (재현성)

## 목적
- 이 문서는 다른 개발자가 이 브랜치를 풀 받아 로컬 또는 서버에서 `unsloth` 기반 학습 환경을 재현하고 실행할 수 있도록, 필수 의존성·설치 절차·주요 파일·실행 예시·트러블슈팅을 간결하게 정리합니다.

---

## 빠른 시작 (Quickstart)
- 의존성 설치 (레포의 `Makefile` 사용)
```bash
make dev
```
- 학습 시작 (항상 Makefile 사용)
```bash
nohup make train > train.log 2>&1 &
```
- 추론 시작
```bash
nohup make inference > inference.log 2>&1 &
```

**중요:** 개별 Python 파일(`src/...`)을 직접 실행하는 방법은 허용되지 않습니다. 항상 `Makefile`의 타겟을 사용하세요.

---

## 필수 시스템 의존성
- 운영체제: Linux 권장
- C 컴파일러: `gcc`, `g++`, `make` (triton 또는 일부 확장 빌드에 필요)

---


## further details

### 설치 및 재현 절차 (상세)
1. 시스템 패키지 설치
```bash
sudo apt update
sudo apt install -y build-essential
```

2. 가상환경 관리
- 레포에서는 `uv`를 사용합니다. (가상환경 생성/활성화는 `uv` 명령을 따르세요)

4. 버전 확인
- 현재 운용중인 V100 서버의 torch 및 CUDA 버전을 확인하세요.
```bash
uv run python -c "import torch; print(torch.__version__);"
```

현재 troch version: 2.9.1+cu128

---

## 구성 파일
- 모델/학습/데이터 파라미터: [configs/config.yaml](configs/config.yaml)
- 의존성/버전: [pyproject.toml](pyproject.toml)

---

## 주요 파일 맵 (검토 포인트)
- 학습 진입점: [src/trainers/train.py](src/trainers/train.py)
- 데이터 콜레이터: [src/trainers/CompletionOnlyDataCollator.py](src/trainers/CompletionOnlyDataCollator.py)
- 데이터 전처리: [src/data/data_processing.py](src/data/data_processing.py)
- 모델 로더: [src/models/model_loader.py](src/models/model_loader.py)
- 추론 진입점: [src/inference/inference.py](src/inference/inference.py)
- 설정 로더: [src/utils/config_loader.py](src/utils/config_loader.py)

---

## 데이터 콜레이터 변경 요약
- 배경: `trl`/`transformers` 버전 변화로 기존 `DataCollatorForCausalLM` API가 맞지 않아 `CompletionOnlyDataCollator`를 도입했습니다.
- 관련 파일: [src/trainers/CompletionOnlyDataCollator.py](src/trainers/CompletionOnlyDataCollator.py)
- 점검 항목: 토크나이저 인코딩, 라벨 시프트(shift), `padding`/`truncation` 동작이 기대 동작과 일치하는지 확인하세요.

---

## 디버깅 & 트러블슈팅
- 컴파일/빌드 오류: `gcc`/`g++`와 `make`가 설치되어 있는지 확인하세요.
- 패키지 호환성 문제: 가상환경에서 `pip list`로 `transformers`, `trl`, `unsloth` 버전을 확인하고 [`pyproject.toml`](pyproject.toml)과 대조하세요.
- 모델·토크나이저 불일치: 체크포인트의 `tokenizer.json`, `vocab.json` 등이 `configs/config.yaml`의 경로와 일치하는지 확인하세요.
- 로그·증상 수집: 재현 이슈 발생 시 `train.log`(또는 `inference.log`)와 `pip list` 출력을 함께 첨부하세요.


---
최종 업데이트: 2025-12-20

추가 참고
- 이 문서의 설치/버전 관련 정확한 값은 반드시 [pyproject.toml](pyproject.toml)에서 확인하고 맞춰 설치하세요.
문의 또는 재현 문제 발생 시, @nerdchanii 에게 알려주세요.
