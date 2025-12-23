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

from abc import ABC
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, ClassVar

import yaml
from typing_extensions import Self


def _load_yaml_config(config_path: str | Path) -> dict[str, Any]:
    """YAML 설정 파일 로드"""
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _flatten_config(config: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    """중첩된 config를 flat하게 변환

    예: {"training": {"epochs": 3}} -> {"training_epochs": 3}
    """
    flat = {}
    for key, value in config.items():
        full_key = f"{prefix}{key}" if prefix else key
        if isinstance(value, dict):
            flat.update(_flatten_config(value, f"{full_key}_"))
        else:
            flat[full_key] = value
    return flat


class BaseConfig(ABC):
    """Config 기본 클래스

    YAML 로드와 wandb config 변환 기능 제공.
    서브클래스에서 _yaml_key_mapping과 _yaml_sections를 정의해야 함.
    """

    _yaml_key_mapping: ClassVar[dict[str, str]]
    _yaml_sections: ClassVar[list[str]]  # ["train", "wandb"] 등 여러 섹션 지정 가능

    @classmethod
    def from_yaml(cls, config_path: str | Path) -> Self:
        """YAML 파일에서 Config 생성

        _yaml_sections에 정의된 모든 섹션을 병합하여 로드.
        """
        yaml_config = _load_yaml_config(config_path)

        # 여러 섹션을 병합
        merged_flat = {}
        for section in cls._yaml_sections:
            section_config = yaml_config.get(section, {})
            flat_config = _flatten_config(section_config)
            merged_flat.update(flat_config)

        # YAML 키를 필드명으로 변환
        kwargs = {}
        for yaml_key, attr_name in cls._yaml_key_mapping.items():
            if yaml_key in merged_flat:
                kwargs[attr_name] = merged_flat[yaml_key]

        return cls(**kwargs)

@dataclass
class TrainConfig(BaseConfig):
    """학습 설정

    configs/config.yaml의 train 섹션과 wandb 섹션에서 설정을 로드함.
    """

    _yaml_sections: ClassVar[list[str]] = ["train", "wandb"]

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
    wandb_entity: str
    wandb_project: str
    wandb_run_name: str | None

    _yaml_key_mapping: ClassVar[dict[str, str]] = {
        # train 섹션
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
        # wandb 섹션
        "entity": "wandb_entity",
        "project": "wandb_project",
        "run_name": "wandb_run_name",
    }

    def to_wandb_config(self) -> dict[str, Any]:
        """wandb.init()에 전달할 config dict 생성

        dataclass의 모든 필드를 자동으로 포함.
        새 필드 추가 시 이 메서드 수정 불필요.
        """
        return {
            field.name: getattr(self, field.name)
            for field in fields(self)
            if not field.name.startswith("_")
        }


@dataclass
class InferenceConfig(BaseConfig):
    """추론 설정

    configs/config.yaml의 inference 섹션에서 설정을 로드함.
    """

    _yaml_sections: ClassVar[list[str]] = ["inference"]

    # 모델 설정, hf 모델명 또는 체크포인트 경로
    checkpoint_path: str

    # 데이터 설정
    test_data: str

    # FLM ONLY 토큰 길이 설정
    max_seq_length: int

    # 출력 설정
    output_path: str

    _yaml_key_mapping: ClassVar[dict[str, str]] = {
        "model_checkpoint_path": "checkpoint_path",
        "data_test_path": "test_data",
        "output_path": "output_path",
        "FLM_max_seq_length" : "max_seq_length"
    }
