"""
Reranker 데이터 변환 및 유틸리티 모듈.

이 모듈은 리랭커(Reranker) 체인의 전후 처리 로직을 담당합니다.
주요 기능:
1. 입력 데이터의 유효성 검증 (Question, Documents 유무 확인)
2. 원본 문제(질문 및 선택지)를 결합하여 리랭킹 전용 'Rich Query' 생성
3. 모델의 토큰 입력 제한을 고려한 텍스트 절단(Truncation)
4. 리랭킹 결과(점수 및 순위)에 대한 로깅 및 모니터링
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)


def truncate_text(text: str, max_chars: int = 200) -> str:
    """
    리랭커 모델의 입력 제한을 고려하여 텍스트 길이를 제한합니다.
    질문(Question)이나 선택지(Choice)가 너무 길 경우 모델 에러를 방지합니다.
    """
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "..."


def validate_rerank_input(input_data: dict[str, Any]) -> bool:
    """
    리랭커가 필요한 필수 데이터(data, multi_docs)가 포함되어 있는지 검증합니다.
    """
    if "data" not in input_data or "multi_docs" not in input_data:
        logger.error("Reranker input missing 'data' or 'multi_docs'")
        return False

    if not isinstance(input_data["multi_docs"], list):
        logger.error("Reranker input 'multi_docs' must be a List[List[Document]]")
        return False

    return True


def format_rich_query(original_data: dict[str, Any]) -> str:
    """
    원본 데이터로부터 리랭킹에 최적화된 'Rich Query'를 생성합니다.
    """
    question = original_data.get("question", "")
    choice = original_data.get("choice", "")

    # 질문과 선택지를 명확히 구분하여 모델이 맥락을 이해하기 쉽게 구성
    query = f"질문: {question}\n선택지: {choice}"

    # 리랭커 모델(BGE 등)의 일반적인 토큰 제한을 고려하여 글자 수 제한
    return truncate_text(query, max_chars=1500)


def log_rerank_results(reranked_results: list[Any]):
    for i, res_obj in enumerate(reranked_results):
        # QueryResult 객체인지 확인 (documents 속성이 있는지)
        docs = getattr(res_obj, "documents", [])
        if docs and hasattr(docs[0], "metadata"):
            top_score = docs[0].metadata.get("rerank_score", 0.0)
            logger.info(f"Query {i} Reranked - Top Score: {top_score:.4f}, Count: {len(docs)}")
        else:
            logger.warning(f"Query {i} has no ranked documents.")
