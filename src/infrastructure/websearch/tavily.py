"""
Tavily API 클라이언트 래퍼.

순수 Tavily API 호출만 담당합니다.
WebSearchClientProtocol을 구현합니다.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal, TypedDict
from urllib.parse import urlparse

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
    search_depth: str


def _is_whitelisted_domain(url: str, whitelist: Sequence[str]) -> bool:
    """
    URL이 화이트리스트 도메인에 속하는지 확인합니다.

    Args:
        url: 검증할 URL
        whitelist: 허용된 도메인 리스트

    Returns:
        화이트리스트에 속하면 True, 아니면 False

    Examples:
        >>> _is_whitelisted_domain("https://www.history.go.kr/page", ["history.go.kr"])
        True
        >>> _is_whitelisted_domain("https://db.history.go.kr/page", ["history.go.kr"])
        True
        >>> _is_whitelisted_domain("https://evil.com", ["history.go.kr"])
        False
    """
    if not whitelist:
        return True

    try:
        # URL에서 도메인 추출 (포트 번호 포함 가능)
        parsed = urlparse(url)
        domain = parsed.netloc.lower()

        # 포트 번호 제거 (예: example.com:8080 -> example.com)
        if ":" in domain:
            domain = domain.split(":")[0]

        for allowed_domain in whitelist:
            allowed_lower = allowed_domain.lower().strip()

            # 정확한 도메인 매칭
            if domain == allowed_lower:
                return True

            # 서브도메인 매칭 (예: db.history.go.kr은 history.go.kr에 매칭)
            if domain.endswith(f".{allowed_lower}"):
                return True

        return False
    except Exception:
        return False


class TavilyClientWrapper:
    """
    Tavily API 클라이언트 래퍼.

    WebSearchClientProtocol을 구현합니다.
    순수하게 Tavily API를 호출하고 WebSearchResult를 반환합니다.
    """

    def __init__(
        self,
        client: TavilyClient | None = None,
        default_params: TavilySearchParams | None = None,
    ) -> None:
        """
        Args:
            client: TavilyClient 인스턴스 (None이면 TAVILY_API_KEY env로 생성)
            default_params: 기본 검색 파라미터
        """
        self._client = client or TavilyClient()  # env에서 TAVILY_API_KEY 읽음
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

        # 화이트리스트 도메인 가져오기 (include_domains 파라미터)
        whitelist = params.get("include_domains", [])

        results: list[WebSearchResult] = []
        for item in raw_results[:max_results]:
            if isinstance(item, dict):
                url = str(item.get("url") or "")

                # URL 도메인 검증: 화이트리스트에 없으면 스킵
                if not url or not _is_whitelisted_domain(url, whitelist):
                    continue

                results.append(
                    WebSearchResult(
                        title=str(item.get("title") or ""),
                        url=url,
                        content=str(item.get("content") or ""),
                        score=item.get("score") or item.get("relevance_score"),
                        raw=item,
                    )
                )

        return results
