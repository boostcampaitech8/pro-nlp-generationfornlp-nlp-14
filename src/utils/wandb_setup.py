"""Weights & Biases 설정 유틸리티"""

import os

import wandb
from dotenv import load_dotenv

from utils.config_loader import TrainConfig


def setup_wandb(config: TrainConfig) -> None:
    """Weights & Biases 설정 및 초기화

    Args:
        config: TrainConfig 객체 (wandb_project, wandb_run_name 필드 포함)
    """
    # .env 파일에서 환경 변수 로드
    load_dotenv()

    # W&B API 키 설정
    wandb_api_key = os.getenv("WANDB_API_KEY")
    if wandb_api_key is None:
        raise ValueError("WANDB_API_KEY 환경 변수가 설정되어 있지 않습니다.")

    wandb.login(key=wandb_api_key)

    # W&B 초기화
    wandb.init(
        entity=config.wandb_entity,
        project=config.wandb_project,
        name=config.wandb_run_name,
        config=config.to_wandb_config(),
    )
