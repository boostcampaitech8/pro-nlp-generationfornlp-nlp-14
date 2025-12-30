from __future__ import annotations

from langchain_core.retrievers import BaseRetriever
from tavily import TavilyClient

from chains.retrieval.adapter.RetrievalAdapter import LangChainRetrievalAdapter
from chains.retrieval.impl.tavily_web_search import TavilyWebSearchService

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


def build_retriever() -> BaseRetriever:
    """
    지금은 Tavily 웹 검색만 쓰므로 TavilyWebSearchService를 만들어 반환한다.
    - api_key가 None이면 env(TAVILY_API_KEY)에서 읽는다.
    - options는 TypedDict 타입이지만, 실제 값은 'dict'로 넘겨야 한다.
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
