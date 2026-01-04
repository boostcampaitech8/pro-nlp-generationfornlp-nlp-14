"""Parent-Document Retrieval (PDR) 검색기.

2단계 검색 전략:
1. Parent 레벨에서 후보 토픽/지문 검색 (sparse 비중 높게)
2. Chunk 레벨에서 세부 검색 (dense 비중 높게)

Note:
    CRUD 작업은 repository 레이어를 직접 사용하세요.
    이 모듈은 검색 전용입니다.

    이 클래스는 Elasticsearch를 직접 의존하지 않습니다.
    Searcher와 ParentReader를 주입받아 사용합니다.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from core.protocols import DocumentRepositoryProtocol, DocumentSearcherProtocol
from core.types import DocumentSearchHit, DocumentSearchParams


@dataclass
class PDRConfig:
    """PDR Retriever 설정."""

    parents_index: str = "parents"
    chunks_index: str = "chunks"


class PDRRetriever:
    """PDR(Parent-Document Retrieval) 전략을 구현하는 검색기.

    2단계 하이브리드 검색:
    - Stage 1: Parent(토픽/지문) 후보 검색 - sparse 비중 높게
    - Stage 2: Chunk 검색 - dense 비중 높게, parent ID로 필터링

    이 클래스는 Elasticsearch를 직접 의존하지 않습니다.
    의존성은 생성자를 통해 주입받습니다.
    """

    def __init__(
        self,
        searcher: DocumentSearcherProtocol,
        parent_reader: DocumentRepositoryProtocol,
        config: PDRConfig | None = None,
    ):
        self._searcher = searcher
        self._parent_reader = parent_reader
        self._config = config or PDRConfig()

    # =========================================================================
    # Stage 1: Parent (Topic) Search
    # =========================================================================

    def search_parents(
        self,
        *,
        query: str,
        query_vector: list[float] | None = None,
        subject: str | None = None,
        size: int = 20,
        sparse_weight: float = 2.0,
        dense_weight: float = 0.5,
    ) -> list[DocumentSearchHit]:
        """Parent 레벨 하이브리드 검색.

        토픽/지문 후보를 찾는 1단계 검색. BM25 비중을 높게 설정.

        Args:
            query: 검색 쿼리 텍스트
            query_vector: 쿼리 임베딩 벡터
            subject: 과목 필터 (예: "수학", "국어")
            size: 반환 결과 수
            sparse_weight: BM25 가중치
            dense_weight: kNN 가중치

        Returns:
            검색 결과 리스트
        """
        filter_q = {"term": {"subject": subject}} if subject else None

        params = DocumentSearchParams(
            query=query,
            query_vector=query_vector,
            size=size,
            sparse_weight=sparse_weight,
            dense_weight=dense_weight,
            filter_query=filter_q,
            text_fields=["topic^2", "parent_text"],
            vector_field="topic_vector",
            source_fields=["doc_id", "subject", "topic", "parent_text", "version"],
        )

        return self._searcher.search(index=self._config.parents_index, params=params)

    # =========================================================================
    # Stage 2: Chunk Search
    # =========================================================================

    def search_chunks(
        self,
        *,
        query: str,
        query_vector: list[float],
        doc_ids: list[str] | None = None,
        subject: str | None = None,
        size: int = 30,
        sparse_weight: float = 1.0,
        dense_weight: float = 2.0,
    ) -> list[DocumentSearchHit]:
        """Chunk 레벨 하이브리드 검색.

        세부 컨텍스트를 찾는 2단계 검색. kNN 비중을 높게 설정.

        Args:
            query: 검색 쿼리 텍스트
            query_vector: 쿼리 임베딩 벡터
            doc_ids: Parent ID 필터 (1단계 결과에서 추출)
            subject: 과목 필터
            size: 반환 결과 수
            sparse_weight: BM25 가중치
            dense_weight: kNN 가중치

        Returns:
            검색 결과 리스트
        """
        filters: list[dict[str, Any]] = []
        if subject:
            filters.append({"term": {"subject": subject}})
        if doc_ids:
            filters.append({"terms": {"doc_id": doc_ids}})

        filter_q: dict[str, Any] | None = None
        if len(filters) == 1:
            filter_q = filters[0]
        elif len(filters) > 1:
            filter_q = {"bool": {"filter": filters}}

        params = DocumentSearchParams(
            query=query,
            query_vector=query_vector,
            size=size,
            sparse_weight=sparse_weight,
            dense_weight=dense_weight,
            filter_query=filter_q,
            text_fields=["chunk_text", "topic"],
            vector_field="chunk_vector",
            source_fields=[
                "chunk_id",
                "doc_id",
                "subject",
                "topic",
                "chunk_idx",
                "chunk_text",
                "start_char",
                "end_char",
                "version",
            ],
        )

        return self._searcher.search(index=self._config.chunks_index, params=params)

    # =========================================================================
    # PDR: Chunk -> Parent Context Fetch (읽기 전용)
    # =========================================================================

    def fetch_parent_contexts(
        self, chunk_hits: list[DocumentSearchHit]
    ) -> dict[str, dict[str, Any]]:
        """Chunk 검색 결과로부터 Parent 컨텍스트 조회.

        Args:
            chunk_hits: Chunk 검색 결과

        Returns:
            {doc_id: parent_source} 형태의 딕셔너리
        """
        doc_ids = sorted({h.source["doc_id"] for h in chunk_hits if "doc_id" in h.source})
        return self._parent_reader.get_documents(doc_ids)

    def mget_parents(self, doc_ids: list[str]) -> dict[str, dict[str, Any]]:
        """여러 Parent 문서 조회 (raw dict 반환)."""
        return self._parent_reader.get_documents(doc_ids)
