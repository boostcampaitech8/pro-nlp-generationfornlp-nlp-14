"""선택지 토큰 관련 유틸리티 (torch 의존성 없음)"""

from utils.constants import CHOICE_TOKENS


def get_choice_token_ids(tokenizer, num_choices: int = 5) -> list[int]:
    """토크나이저에서 선택지 토큰 ID 추출"""
    return [tokenizer.vocab[token] for token in CHOICE_TOKENS[:num_choices]]
