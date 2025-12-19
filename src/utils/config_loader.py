"""YAML 설정 파일 로더

YAML config 파일에서 설정을 로드하는 유틸리티.
CLI arguments 없이 YAML 파일만으로 설정 관리.

Config 구조:
    configs/config.yaml 하나로 train, inference, wandb 설정을 통합 관리.
    - train: 학습 관련 설정
    - inference: 추론 관련 설정
    - wandb: W&B 로깅 설정

새 옵션 추가 시:
    1. dataclass 필드 추가: report_to: str = "none"
    2. YAML에 값 추가: train.training.report_to: "wandb"
    3. _yaml_key_mapping에 매핑 추가: "training_report_to": "report_to"
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import yaml


def load_yaml_config(config_path: str | Path) -> dict[str, Any]:
    """YAML 설정 파일 로드"""
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def flatten_config(config: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    """중첩된 config를 flat하게 변환

    예: {"training": {"epochs": 3}} -> {"training_epochs": 3}
    """
    flat = {}
    for key, value in config.items():
        full_key = f"{prefix}{key}" if prefix else key
        if isinstance(value, dict):
            flat.update(flatten_config(value, f"{full_key}_"))
        else:
            flat[full_key] = value
    return flat


@dataclass
class TrainConfig:
    """학습 설정

    configs/config.yaml의 train 섹션과 wandb 섹션에서 설정을 로드함.
    """

    # 모델 설정
    model_name: str

    # 데이터 설정
    train_data: str
    eval_ratio: float

    # 학습 설정
    output_dir: str
    max_seq_length: int
    batch_size: int
    epochs: int
    learning_rate: float
    weight_decay: float
    logging_steps: int
    logging_strategy: str
    seed: int

    # LoRA 설정
    lora_r: int
    lora_alpha: int
    lora_dropout: float

    # wandb 설정
    wandb_project: str
    wandb_run_name: str | None

    # YAML 키 -> 필드 매핑 (train 섹션 기준)
    _yaml_key_mapping: ClassVar[dict[str, str]] = {
        "model_name": "model_name",
        "data_train_path": "train_data",
        "data_eval_ratio": "eval_ratio",
        "training_output_dir": "output_dir",
        "training_max_seq_length": "max_seq_length",
        "training_batch_size": "batch_size",
        "training_epochs": "epochs",
        "training_learning_rate": "learning_rate",
        "training_weight_decay": "weight_decay",
        "training_logging_steps": "logging_steps",
        "training_logging_strategy": "logging_strategy",
        "training_seed": "seed",
        "lora_r": "lora_r",
        "lora_alpha": "lora_alpha",
        "lora_dropout": "lora_dropout",
    }

    @classmethod
    def from_yaml(cls, config_path: str | Path) -> "TrainConfig":
        """YAML 파일에서 TrainConfig 생성

        config.yaml의 train 섹션과 wandb 섹션을 읽어서 설정 생성.
        """
        yaml_config = load_yaml_config(config_path)

        # train 섹션 flatten
        train_config = yaml_config.get("train", {})
        flat_config = flatten_config(train_config)

        # YAML 키를 필드명으로 변환
        kwargs = {}
        for yaml_key, attr_name in cls._yaml_key_mapping.items():
            if yaml_key in flat_config:
                kwargs[attr_name] = flat_config[yaml_key]

        # wandb 섹션 추가
        wandb_config = yaml_config.get("wandb", {})
        kwargs["wandb_project"] = wandb_config.get("project", "default-project")
        kwargs["wandb_run_name"] = wandb_config.get("run_name")

        return cls(**kwargs)


@dataclass
class InferenceConfig:
    """추론 설정

    configs/config.yaml의 inference 섹션에서 설정을 로드함.
    """

    # 모델 설정, hf 모델명 또는 체크포인트 경로
    checkpoint_path: str

    # 데이터 설정
    test_data: str

    # 출력 설정
    output_path: str

    # YAML 키 -> 필드 매핑 (inference 섹션 기준)
    _yaml_key_mapping: ClassVar[dict[str, str]] = {
        "model_checkpoint_path": "checkpoint_path",
        "data_test_path": "test_data",
        "output_path": "output_path",
    }

    @classmethod
    def from_yaml(cls, config_path: str | Path) -> "InferenceConfig":
        """YAML 파일에서 InferenceConfig 생성

        config.yaml의 inference 섹션을 읽어서 설정 생성.
        """
        yaml_config = load_yaml_config(config_path)

        # inference 섹션 flatten
        inference_config = yaml_config.get("inference", {})
        flat_config = flatten_config(inference_config)

        # YAML 키를 필드명으로 변환
        kwargs = {}
        for yaml_key, attr_name in cls._yaml_key_mapping.items():
            if yaml_key in flat_config:
                kwargs[attr_name] = flat_config[yaml_key]

        return cls(**kwargs)
