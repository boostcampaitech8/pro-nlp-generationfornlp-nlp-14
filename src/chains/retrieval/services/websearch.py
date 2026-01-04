"""
웹 검색 서비스.

WebSearchClientProtocol을 사용하여 RetrievalServicePort를 구현합니다.
벤더 독립적입니다.
"""

from __future__ import annotations

from typing import Any

from chains.retrieval.services.base import RetrievalServicePort
from core.protocols import WebSearchClientProtocol
from core.types import WebSearchResult
from schemas.retrieval.plan import RetrievalRequest
from schemas.retrieval.response import RetrievalResponse


def _format_context(result: WebSearchResult) -> str:
    """검색 결과를 context 문자열로 포맷팅."""
    parts: list[str] = []
    if result.title:
        parts.append(f"[TITLE] {result.title}")
    if result.content:
        parts.append(result.content)
    return "\n".join(parts).strip()


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
                context=_format_context(result),
                metadata={
                    "title": result.title,
                    "url": result.url,
                    "score": result.score,
                    "source": "web",
                    "raw_result": result.raw,
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
