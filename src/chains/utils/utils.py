"""
Pipeline 전반에서 사용되는 공통 유틸리티 함수.
"""

from itertools import zip_longest

from langchain_core.documents import Document

from schemas.retrieval.plan import RetrievalRequest


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
    multi_docs: list[list[Document]],
) -> list[Document]:
    """
    여러 쿼리의 검색 결과를 라운드로빈 방식으로 섞음.

    각 쿼리의 첫 번째 결과, 두 번째 결과, ... 순서로 번갈아가며
    결과를 배치하여 다양성을 높입니다.

    Args:
        multi_docs: 쿼리별 Document 리스트 (list[list[Document]])

    Returns:
        라운드로빈으로 섞인 Document 리스트

    Example:
        >>> q1_docs = [doc1, doc2]
        >>> q2_docs = [doc3, doc4, doc5]
        >>> round_robin_merge([q1_docs, q2_docs])
        [doc1, doc3, doc2, doc4, doc5]
    """
    documents: list[Document] = []
    if multi_docs:
        for grouped in zip_longest(*multi_docs):
            documents.extend(doc for doc in grouped if doc)
    return documents
