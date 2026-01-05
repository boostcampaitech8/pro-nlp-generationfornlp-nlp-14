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
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from langchain_core.documents import Document

logger = logging.getLogger(__name__)


def truncate_text(text: str, max_chars: int) -> str:
    """
    리랭커 모델의 입력 제한을 고려하여 텍스트 길이를 제한합니다.
    질문(Question)이나 선택지(Choice)가 너무 길 경우 모델 에러를 방지합니다.
    """
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "..."


def validate_rerank_input(input_data: dict[str, Any]) -> bool:
    """
    리랭커가 필요한 필수 데이터(multi_docs)가 포함되어 있는지 검증합니다.
    """
    if "multi_docs" not in input_data:
        logger.error("Reranker input missing 'multi_docs'")
        return False

    if not isinstance(input_data["multi_docs"], list):
        logger.error("Reranker input 'multi_docs' must be a List[List[Document]]")
        return False

    return True


def format_rich_query(data: dict[str, Any]) -> str:
    """
    원본 데이터로부터 리랭킹에 최적화된 'Rich Query'를 생성합니다.

    Args:
        data: question, choices(리스트), paragraph 등을 포함한 MCQ 데이터
    """
    question = data.get("question", "")
    choices = data.get("choices", [])
    paragraph = data.get("paragraph", "")

    # choices가 리스트면 번호 붙여서 문자열로 변환
    if isinstance(choices, list):
        choices_str = " ".join(f"({i + 1}) {c}" for i, c in enumerate(choices))
    else:
        choices_str = str(choices)

    # 질문과 선택지를 명확히 구분하여 모델이 맥락을 이해하기 쉽게 구성
    parts = [f"질문: {question}"]
    if choices_str:
        parts.append(f"선택지: {choices_str}")
    if paragraph:
        parts.append(f"지문: {truncate_text(paragraph, max_chars=600)}")

    query = "\n".join(parts)

    # 리랭커 모델(BGE 등)의 일반적인 토큰 제한을 고려하여 글자 수 제한
    return truncate_text(query, max_chars=2500)


def log_rerank_results(multi_docs: "list[list[Document]]") -> None:
    """리랭킹 결과 요약 로그 출력."""
    for i, docs in enumerate(multi_docs):
        if docs:
            top_score = docs[0].metadata.get("rerank_score", 0.0)
            logger.info(f"Query {i} Reranked - Top Score: {top_score:.4f}, Count: {len(docs)}")
        else:
            logger.warning(f"Query {i} has no ranked documents.")
