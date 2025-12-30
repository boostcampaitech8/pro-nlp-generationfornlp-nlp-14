"""Hybrid (sparse + dense) search implementation.

ES 8.x의 linear retriever를 활용한 하이브리드 검색.
"""

from __future__ import annotations

from typing import Any

from elasticsearch import Elasticsearch

from ..config import ESConfig
from .types import SearchHit, SearchParams


class HybridSearcher:
    """ES 8.x linear retriever 기반 하이브리드 검색기.

    SearcherProtocol을 구현하여 PDRRetriever 등에서 직접 사용 가능.
    """

    def __init__(self, es: Elasticsearch, cfg: ESConfig):
        self.es = es
        self.cfg = cfg

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
        """Linear retriever 쿼리 빌드.

        sparse(BM25)와 dense(kNN)를 가중 합산하는 하이브리드 검색 구성.
        """
        # Standard (BM25) retriever query
        standard_query: dict[str, Any] = {
            "multi_match": {
                "query": text_query,
                "fields": text_fields,
                "type": "best_fields",
            }
        }
        if filter_query:
            standard_query = {"bool": {"must": [standard_query], "filter": [filter_query]}}

        retrievers: list[dict[str, Any]] = [
            {
                "retriever": {"standard": {"query": standard_query}},
                "weight": sparse_weight,
                "normalizer": "minmax",
            }
        ]

        # Dense (kNN) retriever
        if query_vector is not None and vector_field:
            knn_obj: dict[str, Any] = {
                "field": vector_field,
                "query_vector": query_vector,
                "k": size,
                "num_candidates": max(50, size * 10),
            }
            if filter_query:
                knn_obj["filter"] = filter_query

            retrievers.append(
                {
                    "retriever": {"knn": knn_obj},
                    "weight": dense_weight,
                    "normalizer": "minmax",
                }
            )

        return {"linear": {"retrievers": retrievers, "rank_window_size": rank_window_size}}

    def search(self, *, index: str, params: SearchParams) -> list[SearchHit]:
        """하이브리드 검색 수행.

        SearcherProtocol 구현. PDRRetriever 등에서 직접 사용 가능.

        Args:
            index: 대상 인덱스명
            params: 검색 파라미터 (일반화된 SearchParams)

        Returns:
            검색 결과 리스트
        """
        retriever = self._build_linear_retriever(
            text_query=params.query,
            query_vector=params.query_vector,
            text_fields=params.text_fields,
            vector_field=params.vector_field or "",
            size=params.size,
            sparse_weight=params.sparse_weight,
            dense_weight=params.dense_weight,
            filter_query=params.filter_query,
            rank_window_size=max(50, params.size * 3),
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
            SearchHit(id=h["_id"], score=h["_score"], source=h["_source"])
            for h in resp["hits"]["hits"]
        ]
