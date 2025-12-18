"""모델 및 토크나이저 로딩 모듈"""

import os

import torch
from dotenv import load_dotenv
from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils import CHAT_TEMPLATE


def huggingface_login(token: str | None = None) -> None:
    """Hugging Face 로그인

    Args:
        token: HF 토큰. None이면 .env 파일의 HF_TOKEN 사용
    """
    load_dotenv()

    if token is None:
        token = os.environ.get("HF_TOKEN")

    if token:
        login(token=token)
    else:
        login()  # 대화형 로그인


def load_tokenizer(model_name: str, use_chat_template: bool = True) -> AutoTokenizer:
    """토크나이저 로드 및 설정

    Args:
        model_name: 모델 이름 또는 경로
        use_chat_template: Chat template 적용 여부

    Returns:
        설정된 토크나이저
    """
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
    )

    if use_chat_template:
        tokenizer.chat_template = CHAT_TEMPLATE

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"

    return tokenizer


def load_model_for_training(model_name: str) -> AutoModelForCausalLM:
    """학습용 모델 로드

    Args:
        model_name: 모델 이름 또는 경로

    Returns:
        학습용 모델
    """
    return AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
