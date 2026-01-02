from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal, TypedDict

from tavily import TavilyClient

from chains.retrieval.services.base import RetrievalServicePort
from schemas.retrieval.plan import RetrievalRequest
from schemas.retrieval.response import RetrievalResponse


class TavilyWebServiceParams(TypedDict, total=False):
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


def _format_context(item: dict[str, Any]) -> str:
    title = str(item.get("title", "") or "")
    content = str(item.get("content", "") or "")

    parts: list[str] = []
    if title:
        parts.append(f"[TITLE] {title}")
    if content:
        parts.append(content)
    return "\n".join(parts).strip()


@dataclass
class TavilyWebSearchService(RetrievalServicePort):
    _client: TavilyClient
    options: TavilyWebServiceParams

    def __init__(
        self,
        client: TavilyClient,
        options: TavilyWebServiceParams | None = None,
    ) -> None:
        self._client = client
        self.options = options or {}  # ✅ TypedDict “인스턴스”는 그냥 dict

    # 포트 명세가 retrieve이면 retrieve로 맞추는 게 좋아. (지금은 임시로 search라도 OK)
    def search(self, request: RetrievalRequest, **kwargs: Any) -> list[RetrievalResponse]:
        # 1) params 합치기(우선순위: options < kwargs < 강제값)
        params: dict[str, Any] = dict(self.options)
        params.update(kwargs)

        # 2) max_results는 한 번만 결정해서 넣기 (중복 방지)
        params["max_results"] = request.top_k

        # 3) query는 TavilyClient.search의 필수 인자
        response = self._client.search(query=request.query, **params)

        results = response.get("results", []) or []
        if not isinstance(results, list):
            results = []

        out: list[RetrievalResponse] = []
        for item in results[: params["max_results"]]:
            if isinstance(item, dict):
                metadata = {
                    "title": item.get("title"),
                    "url": item.get("url"),
                    "score": item.get("score") or item.get("relevance_score"),
                    "raw_result": item,
                }
                out.append(
                    RetrievalResponse(
                        question=request.query,
                        context=_format_context(item),
                        metadata=metadata,
                    )
                )
        return out

    def search_all(
        self, requests: list[RetrievalRequest], **kwargs: Any
    ) -> list[list[RetrievalResponse]]:
        return [self.search(request, **kwargs) for request in requests]
