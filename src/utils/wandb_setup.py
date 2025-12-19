import os

import wandb
from dotenv import load_dotenv


def setup_wandb(project: str, run_name: str | None = None):
    """Weights & Biases 설정 함수

    Args:
        project: W&B 프로젝트 이름
        run_name: W&B 실행 이름 (기본값: None, 자동 생성)
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
        project=project,
        name=run_name,
    )
