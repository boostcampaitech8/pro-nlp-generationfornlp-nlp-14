.PHONY: help install dev lint format clean clean-cache clean-all train inference

# Python path setup
PYTHONPATH := $(shell pwd)/src
export PYTHONPATH

# Training command
train:
	uv run python src/trainers/train.py $(ARGS)

# Inference command
inference:
	uv run python src/inference/inference.py $(ARGS)