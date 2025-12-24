.PHONY: help install dev lint format typecheck check clean clean-cache clean-all train inference

# Python path 설정
PYTHONPATH := $(shell pwd)/src
export PYTHONPATH

# Config 파일 경로
CONFIG := configs/config.yaml

# 기본 타겟
.DEFAULT_GOAL := help

# 도움말
help:
	@echo "사용 가능한 명령어:"
	@echo ""
	@echo "  === 프로젝트 설정 ==="
	@echo "  make install    - 프로덕션 의존성 설치"
	@echo "  make dev        - 개발 의존성 포함 설치"
	@echo ""
	@echo "  === 코드 품질 ==="
	@echo "  make lint       - 코드 린트 검사 (ruff)"
	@echo "  make format     - 코드 포맷팅 (ruff)"
	@echo "  make typecheck  - 타입 검사 (mypy)"
	@echo "  make check      - 모든 코드 품질 검사"
	@echo ""
	@echo "   === 데이터 전처리 ==="
	@echo "   make preprocess  - 지문/보기 분리 전처리 ($(CONFIG))"
	@echo ""
	@echo "  === 학습/추론 ==="
	@echo "  make train                          - 모델 학습 ($(CONFIG))"
	@echo "  make train CONFIG=path.yaml         - 커스텀 config로 학습"
	@echo "  make inference                      - 모델 추론 ($(CONFIG))"
	@echo ""
	@echo "  === 데이터 전처리 ==="
	@echo "  make preprocess                     - 데이터 전처리 (소스 태깅 + Fold 분할)"
	@echo "  make preprocess CONFIG=path.yaml    - 커스텀 config로 전처리"
	@echo ""
	@echo "  === 정리 ==="
	@echo "  make clean       - Python 캐시 삭제"
	@echo "  make clean-cache - 툴 캐시 삭제"
	@echo "  make clean-all   - 모든 캐시 삭제"

# 설치
install: system-deps
	uv sync

dev: system-deps
	uv sync --all-extras

# 코드 품질
lint:
	uv run ruff check src/

format:
	uv run ruff format src/
	uv run ruff check --fix src/

typecheck:
	uv run mypy src/

check: lint typecheck

# 데이터 전처리
preprocess:
	uv run python src/data/seperate_question_plus.py

# 학습/추론
train:
	uv run python src/trainers/train.py $(CONFIG)

inference:
	uv run python src/inference/inference.py $(CONFIG)

# 데이터 전처리 (소스 태깅 + Fold 분할)
preprocess:
	uv run python src/data/preprocess/preprocess.py configs/preprocess.yaml

# 결과 분석
analysis:
	uv run streamlit run src/analysis/streamlit_app.py

# 정리
clean:
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete

clean-cache:
	find . -type d -name ".mypy_cache" -prune -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -prune -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -prune -exec rm -rf {} +

clean-all: clean clean-cache
	rm -rf outputs_gemma/
	rm -f output.csv

system-deps:
	@if ! command -v gcc > /dev/null 2>&1; then \
		echo "gcc not found, installing build-essential..."; \
		sudo apt-get update && sudo apt-get install -y build-essential; \
	fi