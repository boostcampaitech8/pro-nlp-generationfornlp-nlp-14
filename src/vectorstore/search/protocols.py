"""검색 관련 Protocol(인터페이스) 정의.

이 모듈은 Elasticsearch나 다른 인프라에 의존하지 않습니다.
상위 레이어에서 자유롭게 import할 수 있습니다.

Protocol은 구조적 서브타이핑(Structural Subtyping)을 지원합니다.
구현체가 이 Protocol을 상속하지 않아도, 시그니처만 맞으면 호환됩니다.
"""

from __future__ import annotations

from typing import Any, Protocol

from .types import SearchHit, SearchParams


class SearcherProtocol(Protocol):
    """검색기 인터페이스.

    Dense, Sparse, Hybrid 검색 모두 이 인터페이스를 구현할 수 있음.

    Example:
        >>> class HybridSearcherAdapter:
        ...     def search(self, *, index: str, params: SearchParams) -> list[SearchHit]:
        ...         ...  # HybridSearcher 위임
        >>>
        >>> class DenseSearcher:
        ...     def search(self, *, index: str, params: SearchParams) -> list[SearchHit]:
        ...         ...  # kNN 전용 검색
    """

    def search(self, *, index: str, params: SearchParams) -> list[SearchHit]: ...


class ParentReaderProtocol(Protocol):
    """Parent 문서 읽기 인터페이스.

    PDR 검색에서 Chunk → Parent 컨텍스트 조회에 사용.
    """

    def mget_raw(self, doc_ids: list[str]) -> dict[str, dict[str, Any]]: ...
