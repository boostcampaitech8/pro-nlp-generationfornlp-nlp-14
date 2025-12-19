"""Weights & Biases 설정 유틸리티"""

import os

import wandb
from dotenv import load_dotenv
from src.utils.config_loader import BaseConfig


def setup_wandb(config: BaseConfig) -> None:
    """Weights & Biases 설정 및 초기화

    Args:
        config: BaseConfig를 상속한 설정 객체 (TrainConfig, InferenceConfig 등)
                wandb_project, wandb_run_name 필드와 to_wandb_config() 메서드 사용
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
        project=config.wandb_project,
        name=config.wandb_run_name,
        config=config.to_wandb_config(),
    )
