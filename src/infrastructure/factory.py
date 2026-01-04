"""
Infrastructure 팩토리.

외부 서비스 클라이언트들을 생성하는 팩토리 함수들.
"""

from __future__ import annotations

from collections.abc import Sequence

from core.protocols import EmbedderProtocol, WebSearchClientProtocol
from infrastructure.embedders import SolarEmbedder, SolarEmbedderConfig
from infrastructure.websearch import TavilyClientWrapper, TavilySearchParams


# 웹 검색에서 제외할 도메인 기본값
DEFAULT_WEBSEARCH_EXCLUDE_DOMAINS: Sequence[str] = (
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
)


def create_embedder(
    config: SolarEmbedderConfig | None = None,
) -> EmbedderProtocol:
    """
    Embedder 인스턴스를 생성합니다.

    Args:
        config: SolarEmbedder 설정 (None이면 기본값 사용)

    Returns:
        EmbedderProtocol 구현체
    """
    return SolarEmbedder(config)


def create_websearch_client(
    api_key: str | None = None,
    exclude_domains: Sequence[str] | None = None,
) -> WebSearchClientProtocol:
    """
    웹 검색 클라이언트를 생성합니다.

    Args:
        api_key: Tavily API 키 (None이면 env에서 읽음)
        exclude_domains: 제외할 도메인 목록 (None이면 기본값 사용)

    Returns:
        WebSearchClientProtocol 구현체
    """
    default_params: TavilySearchParams = {
        "exclude_domains": list(exclude_domains or DEFAULT_WEBSEARCH_EXCLUDE_DOMAINS),
        "topic": "general",
    }

    return TavilyClientWrapper(api_key=api_key, default_params=default_params)
