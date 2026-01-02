import logging
from typing import Any

from langchain_core.documents import Document

logger = logging.getLogger(__name__)


def truncate_text(text: str, max_chars: int = 1000) -> str:
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


def log_rerank_results(reranked_results: list[list[Document]]):
    """
    리랭킹 완료 후 각 그룹의 최상단 문서 점수를 로그로 출력합니다.
    """
    for i, group in enumerate(reranked_results):
        if group:
            top_score = group[0].metadata.get("rerank_score", 0.0)
            logger.info(f"Group {i} Reranked - Top Score: {top_score:.4f}, Count: {len(group)}")
        else:
            logger.warning(f"Group {i} has no documents after retrieval.")
