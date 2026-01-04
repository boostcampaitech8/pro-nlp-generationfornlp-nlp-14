"""
Infrastructure 팩토리.

외부 서비스 클라이언트들을 생성하는 팩토리 함수들.
"""

from __future__ import annotations

import os
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


def create_embedder() -> EmbedderProtocol:
    """
    Embedder 인스턴스를 생성합니다.

    환경변수에서 설정을 읽습니다:
    - EMBEDDING_DIMS: 임베딩 차원 (필수)
    - SOLAR_PRO2_API_KEY: Solar API 키 (필수)

    Returns:
        EmbedderProtocol 구현체

    Raises:
        KeyError: EMBEDDING_DIMS 환경변수가 설정되지 않은 경우
        ValueError: SOLAR_PRO2_API_KEY가 설정되지 않은 경우
    """
    dim = int(os.environ["EMBEDDING_DIMS"])
    config = SolarEmbedderConfig(dimensions=dim)
    return SolarEmbedder(config)


def create_websearch_client() -> WebSearchClientProtocol:
    """
    웹 검색 클라이언트를 생성합니다.

    환경변수에서 설정을 읽습니다:
    - TAVILY_API_KEY: Tavily API 키 (필수)

    Returns:
        WebSearchClientProtocol 구현체
    """
    default_params: TavilySearchParams = {
        "exclude_domains": list(DEFAULT_WEBSEARCH_EXCLUDE_DOMAINS),
        "topic": "general",
    }

    return TavilyClientWrapper(default_params=default_params)
