"""
Vectorstore 팩토리.

Elasticsearch 관련 컴포넌트를 생성하는 팩토리 함수.
"""

from __future__ import annotations

from dataclasses import dataclass

from elasticsearch import Elasticsearch

from core.protocols import ParentReaderProtocol, SearcherProtocol
from vectorstore.client import create_es_client
from vectorstore.config import ESConfig
from vectorstore.repository import ParentRepository
from vectorstore.search import HybridSearcher


@dataclass
class ESComponents:
    """Elasticsearch 관련 컴포넌트 묶음."""

    es: Elasticsearch
    searcher: SearcherProtocol
    parent_reader: ParentReaderProtocol


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
