"""예측 관련 유틸리티 모듈"""

import numpy as np
import torch

from utils.constants import CHOICE_TOKENS, INT_TO_STR_MAP, STR_TO_INT_MAP


def get_choice_token_ids(tokenizer, num_choices: int = 5) -> list[int]:
    """토크나이저에서 선택지 토큰 ID 추출"""
    return [tokenizer.vocab[token] for token in CHOICE_TOKENS[:num_choices]]


def extract_choice_logits(logits: torch.Tensor, tokenizer, position: int = -1) -> torch.Tensor:
    """logits에서 선택지 토큰에 해당하는 logit만 추출

    Args:
        logits: 모델 출력 logits
        tokenizer: 토크나이저
        position: 추출할 위치 (기본값: 마지막 토큰)

    Returns:
        선택지 토큰의 logits (shape: [batch_size, num_choices])
    """
    logits = logits if not isinstance(logits, tuple) else logits[0]
    choice_ids = get_choice_token_ids(tokenizer)
    return logits[:, position, choice_ids]


def logits_to_prediction(logits: torch.Tensor, num_choices: int = 5) -> str:
    """logits를 예측 문자열로 변환

    Args:
        logits: 선택지 토큰의 logits (1D tensor)
        num_choices: 선택지 개수

    Returns:
        예측된 정답 문자열 ("1" ~ "5")
    """
    target_logits = logits[:num_choices]
    probs = torch.nn.functional.softmax(target_logits.float(), dim=-1)
    pred_idx = int(np.argmax(probs.detach().cpu().numpy()))
    return INT_TO_STR_MAP[pred_idx]


def decode_labels(labels: np.ndarray, tokenizer) -> list[int]:
    """라벨을 정수 인덱스 리스트로 디코딩

    Args:
        labels: 라벨 배열
        tokenizer: 토크나이저

    Returns:
        정수 인덱스 리스트
    """
    cleaned_labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(cleaned_labels, skip_special_tokens=True)
    parsed_labels = [x.split("<end_of_turn>")[0].strip() for x in decoded_labels]
    return [STR_TO_INT_MAP[x] for x in parsed_labels]
