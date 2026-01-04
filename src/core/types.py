"""검색 관련 공용 타입 정의.

이 모듈은 인프라에 의존하지 않습니다.
vectorstore, chains, indexing 등 어디서든 import할 수 있습니다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class DocumentSearchHit:
    """검색 결과 히트."""

    id: str
    score: float
    source: dict[str, Any]


@dataclass
class DocumentSearchParams:
    """일반화된 검색 파라미터.

    Dense, Sparse, Hybrid 검색 모두 대응 가능.

    Attributes:
        query: 검색 쿼리 텍스트
        query_vector: 쿼리 임베딩 벡터 (Dense/Hybrid용)
        size: 반환 결과 수
        filter_query: 필터 쿼리 (ES DSL 형식)
        sparse_weight: BM25 가중치 (Hybrid용)
        dense_weight: kNN 가중치 (Hybrid용)
        text_fields: 텍스트 검색 대상 필드 목록
        vector_field: 벡터 검색 대상 필드명
        source_fields: 반환할 필드 목록
    """

    # 기본 검색 파라미터
    query: str
    query_vector: list[float] | None = None
    size: int = 20
    filter_query: dict[str, Any] | None = None

    # 가중치 (Hybrid 검색용, Dense/Sparse-only는 무시 가능)
    sparse_weight: float = 1.0
    dense_weight: float = 1.0

    # RRF 옵션 (use_rrf=True면 가중치 대신 RRF 사용)
    use_rrf: bool = False
    rrf_rank_constant: int = 60  # RRF k 파라미터

    # 필드 설정
    text_fields: list[str] = field(default_factory=list)
    vector_field: str | None = None
    source_fields: list[str] | None = None


@dataclass
class WebSearchResult:
    """웹 검색 결과 단일 항목.

    벤더 독립적인 웹 검색 결과 타입.
    """

    title: str
    url: str
    content: str
    score: float | None = None
    raw: dict[str, Any] = field(default_factory=dict)
