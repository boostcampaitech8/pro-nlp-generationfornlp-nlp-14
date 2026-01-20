"""
Retrieval 팩토리.

vectorstore와 infrastructure의 컴포넌트를 조합하여 Retriever를 생성합니다.

사용법:
    # config.yaml에서 로드
    config = RetrievalConfig.from_yaml("configs/config.yaml")
    retriever = create_local_retriever(config)

    # 또는 기본값 사용
    retriever = create_local_retriever()

ES/Embedder 연결 정보는 환경변수에서 읽습니다 (.env):
    - ES_URL, ES_USERNAME, ES_PASSWORD
    - SOLAR_PRO2_API_KEY
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
from infrastructure.factory import (
    create_embedder,
    create_es_components,
    create_websearch_client,
)
from utils.config_loader import RetrievalConfig


def create_local_retriever(
    config: RetrievalConfig | None = None,
) -> BaseRetriever:
    """
    로컬 ES 기반 Retriever를 생성합니다.

    config.yaml의 retrieval 섹션에서 설정을 로드합니다.
    ES/Embedder 연결 정보는 환경변수에서 읽습니다.

    Args:
        config: Retrieval 설정 (None이면 기본값 사용)

    Returns:
        LangChain BaseRetriever 호환 retriever
    """
    cfg = config or RetrievalConfig(RetrievalConfig())

    # 1) ES 컴포넌트 생성 (env에서 연결 정보 읽음)
    es_components = create_es_components()

    # 2) Embedder 생성 (env에서 API key, EMBEDDING_DIMS 읽음)
    embedder = create_embedder()

    # 3) PDRRetriever 생성
    pdr_config = PDRConfig(
        parents_index=cfg.parents_index,
        chunks_index=cfg.chunks_index,
    )
    pdr = PDRRetriever(
        es_components.searcher,
        es_components.parent_reader,
        pdr_config,
    )

    # 4) LocalRetrieverService 생성
    retriever_config = LocalRetrieverConfig(
        parent_size=cfg.parent_size,
        parent_sparse_weight=cfg.parent_sparse_weight,
        parent_dense_weight=cfg.parent_dense_weight,
        chunk_size=cfg.chunk_size,
        chunk_sparse_weight=cfg.chunk_sparse_weight,
        chunk_dense_weight=cfg.chunk_dense_weight,
        use_rrf=cfg.use_rrf,
    )
    service = LocalRetrieverService(
        pdr_retriever=pdr,
        embedder=embedder,
        config=retriever_config,
    )

    # 5) LangChain Adapter로 감싸서 반환
    return LangChainRetrievalAdapter(service=service, source_name="local_es")


def create_websearch_retriever() -> BaseRetriever:
    """
    웹 검색 기반 Retriever를 생성합니다.

    환경변수에서 TAVILY_API_KEY를 읽습니다.

    Returns:
        LangChain BaseRetriever 호환 retriever
    """
    # 1) WebSearch 클라이언트 생성 (infrastructure, env에서 API key 읽음)
    client = create_websearch_client()

    # 2) WebSearchService 생성 (벤더 독립)
    service = WebSearchService(client=client)

    # 3) LangChain Adapter로 감싸서 반환
    return LangChainRetrievalAdapter(service=service, source_name="web")
