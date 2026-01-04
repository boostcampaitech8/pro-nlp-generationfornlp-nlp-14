"""
Retrieval 팩토리.

vectorstore와 infrastructure의 컴포넌트를 조합하여 Retriever를 생성합니다.
"""

from __future__ import annotations

from langchain_core.retrievers import BaseRetriever

from chains.retrieval.adapter.RetrievalAdapter import LangChainRetrievalAdapter
from chains.retrieval.domain.pdr_retriever import PDRConfig, PDRRetriever
from chains.retrieval.services.local import (
    LocalRetrieverConfig,
    LocalRetrieverService,
)
from chains.retrieval.services.websearch import WebSearchService
from infrastructure.embedders import SolarEmbedderConfig
from infrastructure.factory import create_embedder, create_websearch_client
from vectorstore.config import ESConfig
from vectorstore.factory import create_es_components


def create_local_retriever(
    es_config: ESConfig | None = None,
    embedder_config: SolarEmbedderConfig | None = None,
    pdr_config: PDRConfig | None = None,
    retriever_config: LocalRetrieverConfig | None = None,
) -> BaseRetriever:
    """
    로컬 ES 기반 Retriever를 생성합니다.

    vectorstore와 infrastructure 팩토리를 조합하여
    완전히 구성된 BaseRetriever를 반환합니다.

    Args:
        es_config: Elasticsearch 설정
        embedder_config: Embedder 설정
        pdr_config: PDR 검색기 설정
        retriever_config: LocalRetriever 설정

    Returns:
        LangChain BaseRetriever 호환 retriever
    """
    # 1) ES 컴포넌트 생성
    es_components = create_es_components(es_config)

    # 2) Embedder 생성
    embedder = create_embedder(embedder_config)

    # 3) PDRRetriever + LocalRetrieverService 생성
    pdr = PDRRetriever(
        es_components.searcher,
        es_components.parent_reader,
        pdr_config or PDRConfig(),
    )
    service = LocalRetrieverService(
        pdr_retriever=pdr,
        embedder=embedder,
        config=retriever_config,
    )

    # 4) LangChain Adapter로 감싸서 반환
    return LangChainRetrievalAdapter(service=service, source_name="local_es")


def create_websearch_retriever(
    api_key: str | None = None,
) -> BaseRetriever:
    """
    웹 검색 기반 Retriever를 생성합니다.

    Args:
        api_key: Tavily API 키 (None이면 env에서 읽음)

    Returns:
        LangChain BaseRetriever 호환 retriever
    """
    # 1) WebSearch 클라이언트 생성 (infrastructure)
    client = create_websearch_client(api_key=api_key)

    # 2) WebSearchService 생성 (벤더 독립)
    service = WebSearchService(client=client)

    # 3) LangChain Adapter로 감싸서 반환
    return LangChainRetrievalAdapter(service=service, source_name="web")
