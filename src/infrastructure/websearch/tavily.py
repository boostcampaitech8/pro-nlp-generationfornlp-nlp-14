"""
Tavily API 클라이언트 래퍼.

순수 Tavily API 호출만 담당합니다.
WebSearchClientProtocol을 구현합니다.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal, TypedDict

from tavily import TavilyClient

from core.types import WebSearchResult


class TavilySearchParams(TypedDict, total=False):
    """Tavily API 검색 파라미터."""

    search_depth: Literal["basic", "advanced", "fast", "ultra-fast"]
    topic: Literal["general", "news", "finance"]
    time_range: Literal["day", "week", "month", "year"]
    start_date: str
    end_date: str
    days: int
    max_results: int
    include_domains: Sequence[str]
    exclude_domains: Sequence[str]
    include_raw_content: bool | Literal["markdown", "text"]


class TavilyClientWrapper:
    """
    Tavily API 클라이언트 래퍼.

    WebSearchClientProtocol을 구현합니다.
    순수하게 Tavily API를 호출하고 WebSearchResult를 반환합니다.
    """

    def __init__(
        self,
        client: TavilyClient | None = None,
        api_key: str | None = None,
        default_params: TavilySearchParams | None = None,
    ) -> None:
        """
        Args:
            client: TavilyClient 인스턴스 (None이면 새로 생성)
            api_key: API 키 (client가 None일 때 사용)
            default_params: 기본 검색 파라미터
        """
        self._client = client or TavilyClient(api_key=api_key)
        self._default_params = default_params or {}

    def search(
        self,
        query: str,
        max_results: int = 10,
        **kwargs: Any,
    ) -> list[WebSearchResult]:
        """
        웹 검색 수행.

        Args:
            query: 검색 쿼리
            max_results: 최대 결과 수
            **kwargs: 추가 Tavily API 파라미터

        Returns:
            검색 결과 리스트
        """
        params: dict[str, Any] = dict(self._default_params)
        params.update(kwargs)
        params["max_results"] = max_results

        response = self._client.search(query=query, **params)

        raw_results = response.get("results", []) or []
        if not isinstance(raw_results, list):
            raw_results = []

        results: list[WebSearchResult] = []
        for item in raw_results[:max_results]:
            if isinstance(item, dict):
                results.append(
                    WebSearchResult(
                        title=str(item.get("title") or ""),
                        url=str(item.get("url") or ""),
                        content=str(item.get("content") or ""),
                        score=item.get("score") or item.get("relevance_score"),
                        raw=item,
                    )
                )

        return results
