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

    YAML 파일에서 설정을 로드함. 새 옵션 추가 시:
        1. 필드 추가: report_to: str = "none"
        2. _yaml_key_mapping에 매핑 추가
    """

    # 모델 설정
    model_name: str = "beomi/gemma-ko-2b"

    # 데이터 설정
    train_data: str = "data/train.csv"
    eval_ratio: float = 0.1

    # 학습 하이퍼파라미터
    output_dir: str = "outputs"
    max_seq_length: int = 1024
    batch_size: int = 1
    epochs: int = 3
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    logging_steps: int = 1
    seed: int = 42
    report_to: str = "none"

    # LoRA 설정
    lora_r: int = 6
    lora_alpha: int = 8
    lora_dropout: float = 0.05

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
        "training_seed": "seed",
        "training_report_to": "report_to",
        "lora_r": "lora_r",
        "lora_alpha": "lora_alpha",
        "lora_dropout": "lora_dropout",
    }

    @classmethod
    def from_yaml(cls, config_path: str | Path) -> "TrainConfig":
        """YAML 파일에서 TrainConfig 생성"""
        config = cls()
        config._load_from_yaml(config_path)
        return config

    def _load_from_yaml(self, config_path: str | Path) -> None:
        """YAML 파일에서 설정 로드"""
        yaml_config = load_yaml_config(config_path)
        flat_config = flatten_config(yaml_config)

        for yaml_key, attr_name in self._yaml_key_mapping.items():
            if yaml_key in flat_config and hasattr(self, attr_name):
                setattr(self, attr_name, flat_config[yaml_key])


@dataclass
class InferenceConfig:
    """추론 설정

    YAML 파일에서 설정을 로드함.
    """

    # 모델 설정
    checkpoint_path: str = "outputs/checkpoint-xxx"

    # 데이터 설정
    test_data: str = "data/test.csv"

    # 출력 설정
    output_path: str = "output.csv"

    # YAML 키 -> 필드 매핑
    _yaml_key_mapping: ClassVar[dict[str, str]] = {
        "model_checkpoint_path": "checkpoint_path",
        "data_test_path": "test_data",
        "output_path": "output_path",
    }

    @classmethod
    def from_yaml(cls, config_path: str | Path) -> "InferenceConfig":
        """YAML 파일에서 InferenceConfig 생성"""
        config = cls()
        config._load_from_yaml(config_path)
        return config

    def _load_from_yaml(self, config_path: str | Path) -> None:
        """YAML 파일에서 설정 로드"""
        yaml_config = load_yaml_config(config_path)
        flat_config = flatten_config(yaml_config)

        for yaml_key, attr_name in self._yaml_key_mapping.items():
            if yaml_key in flat_config and hasattr(self, attr_name):
                setattr(self, attr_name, flat_config[yaml_key])
