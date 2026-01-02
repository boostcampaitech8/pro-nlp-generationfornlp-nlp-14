"""
Pipeline 전반에서 사용되는 공통 유틸리티 함수.
"""

from itertools import zip_longest

from schemas.retrieval.plan import RetrievalRequest
from schemas.retrieval.response import RetrievalResponse


def normalize_request(req: RetrievalRequest | dict) -> RetrievalRequest:
    """
    RetrievalRequest로 정규화.

    LLM structured output이 dict로 올 수 있으므로,
    확실히 RetrievalRequest 타입으로 변환합니다.

    Args:
        req: RetrievalRequest 또는 dict

    Returns:
        RetrievalRequest 인스턴스
    """
    if isinstance(req, dict):
        return RetrievalRequest(**req)
    return req


def round_robin_merge(
    responses_by_query: list[list[RetrievalResponse]],
) -> list[RetrievalResponse]:
    """
    여러 쿼리의 검색 결과를 라운드로빈 방식으로 섞음.

    각 쿼리의 첫 번째 결과, 두 번째 결과, ... 순서로 번갈아가며
    결과를 배치하여 다양성을 높입니다.

    Args:
        responses_by_query: 쿼리별 검색 결과 리스트

    Returns:
        라운드로빈으로 섞인 검색 결과 리스트

    Example:
        >>> q1_results = [resp1, resp2]
        >>> q2_results = [resp3, resp4, resp5]
        >>> round_robin_merge([q1_results, q2_results])
        [resp1, resp3, resp2, resp4, resp5]
    """
    responses: list[RetrievalResponse] = []
    if responses_by_query:
        for grouped in zip_longest(*responses_by_query):
            responses.extend(res for res in grouped if res)
    return responses
