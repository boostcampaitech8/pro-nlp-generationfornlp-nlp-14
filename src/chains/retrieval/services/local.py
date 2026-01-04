"""
Local VectorDB 기반 Retrieval Service.

PDRRetriever를 사용하여 2단계 Parent-Document Retrieval을 수행합니다.
RetrievalServicePort를 구현하여 chains 레이어와 통합됩니다.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from chains.retrieval.domain.pdr_retriever import PDRRetriever
from chains.retrieval.services.base import RetrievalServicePort
from core.protocols import EmbedderProtocol
from schemas.retrieval.plan import RetrievalRequest
from schemas.retrieval.response import RetrievalResponse


@dataclass
class LocalRetrieverConfig:
    """LocalRetrieverService 설정."""

    # Parent 검색 설정
    parent_size: int = 10
    parent_sparse_weight: float = 2.0
    parent_dense_weight: float = 0.5

    # Chunk 검색 설정
    chunk_size: int = 20
    chunk_sparse_weight: float = 1.0
    chunk_dense_weight: float = 2.0
    use_rrf: bool = False


class LocalRetrieverService(RetrievalServicePort):
    """
    Local Elasticsearch 기반 Retrieval Service.

    PDR(Parent-Document Retrieval) 전략을 사용하여
    2단계 하이브리드 검색을 수행합니다.

    RetrievalServicePort를 명시적으로 구현합니다.
    """

    def __init__(
        self,
        pdr_retriever: PDRRetriever,
        embedder: EmbedderProtocol,
        config: LocalRetrieverConfig | None = None,
    ) -> None:
        self._pdr = pdr_retriever
        self._embedder = embedder
        self._config = config or LocalRetrieverConfig()

    def search(
        self,
        request: RetrievalRequest,
        **kwargs: Any,
    ) -> list[RetrievalResponse]:
        """
        PDR 전략으로 검색 수행.

        Args:
            request: 검색 요청 (query, top_k 포함)
            **kwargs: 추가 파라미터
                - subject: 과목 필터 (예: "수학", "국어")
                - skip_parent_search: True면 parent 검색 건너뜀

        Returns:
            검색 결과 리스트 (RetrievalResponse)
        """
        query = request.query
        top_k = request.top_k
        subject = kwargs.get("subject")
        skip_parent = kwargs.get("skip_parent_search", False)

        # 1) Query embedding 생성
        query_vector = self._embedder.embed(query)

        # 2) Stage 1: Parent 검색 (후보 토픽 찾기)
        doc_ids: list[str] | None = None
        if not skip_parent:
            parent_hits = self._pdr.search_parents(
                query=query,
                query_vector=query_vector,
                subject=subject,
                size=self._config.parent_size,
                sparse_weight=self._config.parent_sparse_weight,
                dense_weight=self._config.parent_dense_weight,
                use_rrf=self._config.use_rrf,
            )
            doc_ids = [
                str(h.source.get("doc_id"))
                for h in parent_hits
                if h.source.get("doc_id") is not None
            ]

        # 3) Stage 2: Chunk 검색
        chunk_hits = self._pdr.search_chunks(
            query=query,
            query_vector=query_vector,
            doc_ids=doc_ids,
            subject=subject,
            size=self._config.chunk_size,
            sparse_weight=self._config.chunk_sparse_weight,
            dense_weight=self._config.chunk_dense_weight,
        )

        # 4) Parent context 조회 (PDR: chunk로 검색, parent만 반환)
        parent_contexts = self._pdr.fetch_parent_contexts(chunk_hits) if chunk_hits else {}

        # 5) 결과 변환 (parent 기준 deduplicate)
        seen_doc_ids: set[str] = set()
        results: list[RetrievalResponse] = []

        for chunk in chunk_hits:
            doc_id = chunk.source.get("doc_id", "")

            # 같은 parent는 한 번만 반환
            if doc_id in seen_doc_ids:
                continue
            seen_doc_ids.add(doc_id)

            parent_ctx = parent_contexts.get(doc_id, {})
            context = parent_ctx.get("parent_text", "")

            metadata = {
                "doc_id": doc_id,
                "subject": chunk.source.get("subject"),
                "topic": chunk.source.get("topic"),
                "score": chunk.score,
                "rank": len(results) + 1,
                "source": "local_es",
            }

            results.append(
                RetrievalResponse(
                    question=query,
                    context=context,
                    metadata=metadata,
                )
            )

            if len(results) >= top_k:
                break

        return results
