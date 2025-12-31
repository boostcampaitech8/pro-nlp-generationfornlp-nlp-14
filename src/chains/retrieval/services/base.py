"""
Retrieval service port (Protocol).

Service 구현체들이 따라야 할 인터페이스를 정의합니다.
Protocol을 사용하여 duck typing을 지원하며, 명시적인 상속 없이도 호환됩니다.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from schemas.retrieval.plan import RetrievalRequest
from schemas.retrieval.response import RetrievalResponse


@runtime_checkable
class RetrievalServicePort(Protocol):
    """
    Retrieval service port.

    구현체 예시:
    - TavilyWebSearchService: Tavily API를 사용한 웹 검색
    - ESRetrieverService: Elasticsearch 기반 검색 (미래)

    Note:
        search_batch 메서드는 나중에 성능 최적화를 위해 추가될 예정입니다.
        현재는 search() 메서드만 구현하면 됩니다.
    """

    def search(self, req: RetrievalRequest, **kwargs: Any) -> list[RetrievalResponse]:
        """
        단일 검색 요청 처리.

        Args:
            req: 검색 요청 (query, top_k 포함)
            **kwargs: 추가 파라미터 (확장 가능)

        Returns:
            검색 결과 리스트
        """
        ...
