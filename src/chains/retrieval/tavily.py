"""
Tavily 웹 검색 기반 Retriever 빌더.

BaseRetriever를 생성하여 반환합니다.
"""

from __future__ import annotations

from langchain_core.retrievers import BaseRetriever
from tavily import TavilyClient

from chains.retrieval.adapter.RetrievalAdapter import LangChainRetrievalAdapter
from chains.retrieval.services.tavily import TavilyWebSearchService

WEBSEARCH_EXCLUDE_DOMAINS = [
    "reddit.com",
    "www.reddit.com",
    "instagram.com",
    "www.instagram.com",
    "facebook.com",
    "www.facebook.com",
    "twitter.com",
    "www.twitter.com",
    "x.com",
    "t.co",
    "pinterest.com",
    "imgur.com",
    "youtube.com",
    "m.youtube.com",
    "discord.com",
    "linkedin.com",
    "weibo.com",
    "telegram.me",
    "tumblr.com",
    "medium.com",
]


def build_tavily_retriever() -> BaseRetriever:
    """
    Tavily 웹 검색 서비스를 사용하는 BaseRetriever를 생성합니다.

    Returns:
        LangChainRetrievalAdapter로 감싼 BaseRetriever

    Note:
        - api_key가 None이면 env(TAVILY_API_KEY)에서 읽습니다.
        - options는 TypedDict 타입이지만, 실제 값은 'dict'로 넘겨야 합니다.
        - Adapter + Service 패턴으로 구성됩니다.
    """

    service = TavilyWebSearchService(
        TavilyClient(),
        options={
            "max_results": 20,
            "exclude_domains": WEBSEARCH_EXCLUDE_DOMAINS,
            "topic": "general",
        },
    )
    return LangChainRetrievalAdapter(service=service, source_name="web")
