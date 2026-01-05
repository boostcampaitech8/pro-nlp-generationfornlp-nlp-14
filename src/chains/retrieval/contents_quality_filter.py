"""
문서 품질 필터링.

Retrieval 결과의 품질을 검증하고 낮은 품질의 문서를 필터링합니다.
"""

from __future__ import annotations

from langchain_core.documents import Document
from langchain_core.runnables import Runnable, RunnableLambda


def is_quality_content(
    content: str,
    min_length: int = 50,
    max_length: int = 5000,
    min_korean_ratio: float = 0.1,
) -> bool:
    """
    컨텐츠 품질을 검증합니다.

    Args:
        content: 검증할 컨텐츠
        min_length: 최소 길이 (기본값: 50자)
        max_length: 최대 길이 (기본값: 5000자)
        min_korean_ratio: 최소 한글 비율 (기본값: 0.1 = 10%)

    Returns:
        품질이 충분하면 True, 아니면 False

    검증 기준:
        1. 길이가 min_length 이상, max_length 이하
        2. 한글 비율이 min_korean_ratio 이상
        3. 제어 문자(바이너리 데이터)가 과도하지 않음

    Examples:
        >>> is_quality_content("한국의 역사는 매우 오래되었습니다.")
        True
        >>> is_quality_content("짧음")
        False
        >>> is_quality_content("English only content with no Korean")
        False
    """
    if not content:
        return False

    # 길이 체크
    if len(content) < min_length or len(content) > max_length:
        return False

    # 바이너리 데이터 감지 (제어 문자 체크, 단 \n, \t, \r 제외)
    sample = content[:200] if len(content) > 200 else content
    control_chars = sum(1 for c in sample if ord(c) < 32 and c not in "\n\t\r")
    if control_chars > 5:  # 너무 많은 제어 문자
        return False

    # 한글 비율 체크
    korean_chars = sum(1 for c in content if "가" <= c <= "힣")
    korean_ratio = korean_chars / len(content)

    return korean_ratio >= min_korean_ratio


def contents_quality_filter(
    min_length: int = 50,
    max_length: int = 5000,
    min_korean_ratio: float = 0.1,
) -> Runnable[list[list[Document]], list[list[Document]]]:
    """
    문서 품질 필터링 Runnable을 생성합니다.

    Multi-query retrieval 결과(list[list[Document]])를 받아서
    각 query별 문서 리스트에서 품질이 낮은 문서를 필터링합니다.

    Args:
        min_length: 최소 길이 (기본값: 50자)
        max_length: 최대 길이 (기본값: 5000자)
        min_korean_ratio: 최소 한글 비율 (기본값: 0.1 = 10%)

    Returns:
        Runnable[list[list[Document]], list[list[Document]]]

    Examples:
        >>> from chains.retrieval.chain import build_multi_query_retriever
        >>> from chains.retrieval.filters import quality_filter
        >>>
        >>> retriever = build_multi_query_retriever(...)
        >>> filter_node = quality_filter(min_length=50)
        >>>
        >>> # Chain 연결
        >>> chain = retriever | filter_node
        >>> result = chain.invoke(plan)
    """

    def filter_documents(multi_docs: list[list[Document]]) -> list[list[Document]]:
        """
        각 query별 문서 리스트에서 품질이 낮은 문서를 필터링합니다.
        """
        filtered_multi_docs: list[list[Document]] = []

        for docs in multi_docs:
            filtered_docs = [
                doc
                for doc in docs
                if is_quality_content(
                    doc.page_content,
                    min_length=min_length,
                    max_length=max_length,
                    min_korean_ratio=min_korean_ratio,
                )
            ]
            filtered_multi_docs.append(filtered_docs)

        return filtered_multi_docs

    return RunnableLambda(filter_documents)
