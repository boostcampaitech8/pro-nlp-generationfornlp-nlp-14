"""Hybrid (sparse + dense) search implementation.

ES 8.x의 linear retriever를 활용한 하이브리드 검색.
"""

from __future__ import annotations

from typing import Any

from elasticsearch import Elasticsearch

from core.types import DocumentSearchHit, DocumentSearchParams
from vectorstore.config import ESConfig


class HybridSearcher:
    """ES 8.x linear/rrf retriever 기반 하이브리드 검색기.

    SearcherProtocol을 구현하여 PDRRetriever 등에서 직접 사용 가능.
    """

    def __init__(self, es: Elasticsearch, cfg: ESConfig):
        self.es = es
        self.cfg = cfg

    # =========================================================================
    # Building Blocks: Standard & kNN Retrievers
    # =========================================================================

    def _build_standard_query(
        self,
        text_query: str,
        text_fields: list[str],
        filter_query: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """BM25 standard query 빌드."""
        query: dict[str, Any] = {
            "multi_match": {
                "query": text_query,
                "fields": text_fields,
                "type": "best_fields",
            }
        }
        if filter_query:
            query = {"bool": {"must": [query], "filter": [filter_query]}}
        return {"standard": {"query": query}}

    def _build_knn_retriever(
        self,
        query_vector: list[float],
        vector_field: str,
        size: int,
        filter_query: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """kNN retriever 빌드."""
        knn_obj: dict[str, Any] = {
            "field": vector_field,
            "query_vector": query_vector,
            "k": size,
            "num_candidates": max(50, size * 10),
        }
        if filter_query:
            knn_obj["filter"] = filter_query
        return {"knn": knn_obj}

    # =========================================================================
    # Composite Retrievers: Linear & RRF
    # =========================================================================

    def _build_linear_retriever(
        self,
        *,
        text_query: str,
        query_vector: list[float] | None,
        text_fields: list[str],
        vector_field: str,
        size: int,
        sparse_weight: float,
        dense_weight: float,
        filter_query: dict[str, Any] | None,
        rank_window_size: int,
    ) -> dict[str, Any]:
        """Linear retriever 쿼리 빌드 (가중 합산)."""
        standard = self._build_standard_query(text_query, text_fields, filter_query)
        retrievers: list[dict[str, Any]] = [
            {"retriever": standard, "weight": sparse_weight, "normalizer": "minmax"}
        ]

        if query_vector is not None and vector_field:
            knn = self._build_knn_retriever(query_vector, vector_field, size, filter_query)
            retrievers.append(
                {"retriever": knn, "weight": dense_weight, "normalizer": "minmax"}
            )

        return {"linear": {"retrievers": retrievers, "rank_window_size": rank_window_size}}

    def _build_rrf_retriever(
        self,
        *,
        text_query: str,
        query_vector: list[float] | None,
        text_fields: list[str],
        vector_field: str,
        size: int,
        filter_query: dict[str, Any] | None,
        rank_constant: int,
        rank_window_size: int,
    ) -> dict[str, Any]:
        """RRF retriever 쿼리 빌드 (순위 기반 융합)."""
        standard = self._build_standard_query(text_query, text_fields, filter_query)
        retrievers: list[dict[str, Any]] = [standard]

        if query_vector is not None and vector_field:
            knn = self._build_knn_retriever(query_vector, vector_field, size, filter_query)
            retrievers.append(knn)

        return {
            "rrf": {
                "retrievers": retrievers,
                "rank_constant": rank_constant,
                "rank_window_size": rank_window_size,
            }
        }

    def search(self, *, index: str, params: DocumentSearchParams) -> list[DocumentSearchHit]:
        """하이브리드 검색 수행.

        SearcherProtocol 구현. PDRRetriever 등에서 직접 사용 가능.

        Args:
            index: 대상 인덱스명
            params: 검색 파라미터 (일반화된 SearchParams)

        Returns:
            검색 결과 리스트
        """
        rank_window_size = max(50, params.size * 3)

        if params.use_rrf:
            retriever = self._build_rrf_retriever(
                text_query=params.query,
                query_vector=params.query_vector,
                text_fields=params.text_fields,
                vector_field=params.vector_field or "",
                size=params.size,
                filter_query=params.filter_query,
                rank_constant=params.rrf_rank_constant,
                rank_window_size=rank_window_size,
            )
        else:
            retriever = self._build_linear_retriever(
                text_query=params.query,
                query_vector=params.query_vector,
                text_fields=params.text_fields,
                vector_field=params.vector_field or "",
                size=params.size,
                sparse_weight=params.sparse_weight,
                dense_weight=params.dense_weight,
                filter_query=params.filter_query,
                rank_window_size=rank_window_size,
            )

        search_kwargs: dict[str, Any] = {
            "index": index,
            "size": params.size,
            "retriever": retriever,
        }
        if params.source_fields:
            search_kwargs["_source"] = params.source_fields

        resp = self.es.search(**search_kwargs)

        return [
            DocumentSearchHit(id=h["_id"], score=h["_score"], source=h["_source"])
            for h in resp["hits"]["hits"]
        ]
