"""YAML 설정 파일 로더

YAML config 파일에서 설정을 로드하는 유틸리티.
CLI arguments 없이 YAML 파일만으로 설정 관리.

새 옵션 추가 시:
    1. dataclass 필드 추가: report_to: str = "none"
    2. YAML에 값 추가: training.report_to: "wandb"
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

    YAML 파일에서 설정을 로드함. 기본값 없이 YAML 필수.
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
    report_to: str

    # LoRA 설정
    lora_r: int
    lora_alpha: int
    lora_dropout: float

    # YAML 키 -> 필드 매핑
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
        "training_report_to": "report_to",
        "lora_r": "lora_r",
        "lora_alpha": "lora_alpha",
        "lora_dropout": "lora_dropout",
    }

    @classmethod
    def from_yaml(cls, config_path: str | Path) -> "TrainConfig":
        """YAML 파일에서 TrainConfig 생성"""
        yaml_config = load_yaml_config(config_path)
        flat_config = flatten_config(yaml_config)

        # YAML 키를 필드명으로 변환
        kwargs = {}
        for yaml_key, attr_name in cls._yaml_key_mapping.items():
            if yaml_key in flat_config:
                kwargs[attr_name] = flat_config[yaml_key]

        return cls(**kwargs)


@dataclass
class InferenceConfig:
    """추론 설정

    YAML 파일에서 설정을 로드함. 기본값 없이 YAML 필수.
    """

    # 모델 설정, hf 모델명
    checkpoint_path: str

    # 데이터 설정
    test_data: str

    # 출력 설정
    output_path: str

    # YAML 키 -> 필드 매핑
    _yaml_key_mapping: ClassVar[dict[str, str]] = {
        "model_checkpoint_path": "checkpoint_path",
        "data_test_path": "test_data",
        "output_path": "output_path",
    }

    @classmethod
    def from_yaml(cls, config_path: str | Path) -> "InferenceConfig":
        """YAML 파일에서 InferenceConfig 생성"""
        yaml_config = load_yaml_config(config_path)
        flat_config = flatten_config(yaml_config)

        # YAML 키를 필드명으로 변환
        kwargs = {}
        for yaml_key, attr_name in cls._yaml_key_mapping.items():
            if yaml_key in flat_config:
                kwargs[attr_name] = flat_config[yaml_key]

        return cls(**kwargs)
