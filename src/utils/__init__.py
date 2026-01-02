"""유틸리티 패키지

다른 모듈에서 쉽게 접근할 수 있도록 주요 함수와 상수를 노출합니다.

사용 예시:
    from utils import INT_TO_STR_MAP, extract_choice_logits
    from utils import CHAT_TEMPLATE
    from utils import TrainConfig, InferenceConfig
"""

# Config 로더 및 설정 클래스
# 선택지 토큰 유틸 (torch 불필요)
from utils.choice_utils import get_choice_token_ids
from utils.config_loader import (
    InferenceConfig,
    TrainConfig,
)

# 상수
from utils.constants import CHOICE_TOKENS, INT_TO_STR_MAP, STR_TO_INT_MAP

# 프롬프트 템플릿
from utils.prompts import CHAT_TEMPLATE

__all__ = [
    # 상수
    "INT_TO_STR_MAP",
    "STR_TO_INT_MAP",
    "CHOICE_TOKENS",
    # Config 클래스
    "TrainConfig",
    "InferenceConfig",
    # 선택지 토큰 유틸 (torch 불필요)
    "get_choice_token_ids",
    # 예측 함수 (lazy import, torch 필요)
    "extract_choice_logits",
    "logits_to_prediction",
    "decode_labels",
    # 템플릿
    "CHAT_TEMPLATE",
    # wandb 설정 함수 (lazy import)
    "setup_wandb",
]

# torch 의존성이 있는 함수들 (lazy import)
_PREDICTION_FUNCS = {"extract_choice_logits", "logits_to_prediction", "decode_labels"}


def __getattr__(name):
    """torch 의존성 함수들은 실제 사용 시점에 lazy import."""
    if name == "setup_wandb":
        from utils.wandb_setup import setup_wandb

        return setup_wandb
    if name in _PREDICTION_FUNCS:
        from utils import prediction

        return getattr(prediction, name)
    raise AttributeError(f"module 'utils' has no attribute '{name}'")
