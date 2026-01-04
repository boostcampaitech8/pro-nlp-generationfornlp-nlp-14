"""
웹 검색 서비스.

WebSearchClientProtocol을 사용하여 RetrievalServicePort를 구현합니다.
벤더 독립적입니다.
"""

from __future__ import annotations

from typing import Any

from chains.retrieval.services.base import RetrievalServicePort
from core.protocols import WebSearchClientProtocol
from schemas.retrieval.plan import RetrievalRequest
from schemas.retrieval.response import RetrievalResponse


class WebSearchService(RetrievalServicePort):
    """
    웹 검색 서비스.

    WebSearchClientProtocol 구현체를 주입받아 RetrievalServicePort를 구현합니다.
    벤더 독립적이므로 Tavily, Bing 등 어떤 클라이언트든 사용 가능합니다.
    """

    def __init__(self, client: WebSearchClientProtocol) -> None:
        """
        Args:
            client: WebSearchClientProtocol 구현체 (예: TavilyClientWrapper)
        """
        self._client = client

    def search(
        self,
        request: RetrievalRequest,
        **kwargs: Any,
    ) -> list[RetrievalResponse]:
        """
        웹 검색 수행.

        Args:
            request: 검색 요청 (query, top_k 포함)
            **kwargs: 추가 파라미터

        Returns:
            검색 결과 리스트
        """
        results = self._client.search(
            query=request.query,
            max_results=request.top_k,
            **kwargs,
        )

        return [
            RetrievalResponse(
                question=request.query,
                context=result.content,
                metadata={
                    "topic": result.title,
                    "url": result.url,
                    "score": result.score,
                    "source": "web",
                },
            )
            for result in results
        ]

    def search_all(
        self,
        requests: list[RetrievalRequest],
        **kwargs: Any,
    ) -> list[list[RetrievalResponse]]:
        """배치 검색 수행."""
        return [self.search(request, **kwargs) for request in requests]
