"""
Infrastructure 팩토리.

외부 서비스 클라이언트들을 생성하는 팩토리 함수들.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass

from elasticsearch import Elasticsearch

from core.protocols import (
    DocumentRepositoryProtocol,
    DocumentSearcherProtocol,
    EmbedderProtocol,
    WebSearchClientProtocol,
)
from infrastructure.embedders import SolarEmbedder, SolarEmbedderConfig
from infrastructure.vectorstore.client import create_es_client
from infrastructure.vectorstore.config import ESConfig
from infrastructure.vectorstore.repository import ParentRepository
from infrastructure.vectorstore.search import HybridSearcher
from infrastructure.websearch import TavilyClientWrapper, TavilySearchParams

# 웹 검색에서 허용할 도메인 화이트리스트 (치팅 방지)
WEBSEARCH_WHITELIST_DOMAINS: Sequence[str] = (
    "encykorea.aks.ac.kr",
    "history.go.kr",  # www.history.go.kr, db.history.go.kr, contents.history.go.kr 모두 매칭
    "scourt.go.kr",  # www.scourt.go.kr 및 서브도메인 매칭
    "law.go.kr",  # www.law.go.kr 및 서브도메인 매칭
    "wikidata.org",  # www.wikidata.org 및 서브도메인 매칭
    "history.state.gov",
    "openstax.org",
    "wikipedia.org",
    "ccourt.go.kr",
    "easylaw.go.kr",
    "archive.much.go.kr",
    "terms.naver.com",
)

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
        "include_domains": list(WEBSEARCH_WHITELIST_DOMAINS),
        "exclude_domains": list(DEFAULT_WEBSEARCH_EXCLUDE_DOMAINS),
        "topic": "general",
        "search_depth": "advanced",
    }

    return TavilyClientWrapper(default_params=default_params)


@dataclass
class ESComponents:
    """Elasticsearch 관련 컴포넌트 묶음."""

    es: Elasticsearch
    searcher: DocumentSearcherProtocol
    parent_reader: DocumentRepositoryProtocol


def create_es_components(
    config: ESConfig | None = None,
) -> ESComponents:
    """
    Elasticsearch 관련 컴포넌트를 생성합니다.

    Args:
        config: ES 설정 (None이면 기본값 사용)

    Returns:
        ESComponents (es, searcher, parent_reader)
    """
    cfg = config or ESConfig()
    es = create_es_client(cfg)
    searcher = HybridSearcher(es, cfg)
    parent_reader = ParentRepository(es, cfg)

    return ESComponents(
        es=es,
        searcher=searcher,
        parent_reader=parent_reader,
    )
