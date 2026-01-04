"""검색 관련 Protocol(인터페이스) 정의.

이 모듈은 인프라에 의존하지 않습니다.
vectorstore, chains, indexing 등 어디서든 import할 수 있습니다.

Protocol은 구조적 서브타이핑(Structural Subtyping)을 지원합니다.
구현체가 이 Protocol을 상속하지 않아도, 시그니처만 맞으면 호환됩니다.
"""

from __future__ import annotations

from typing import Any, Protocol

from core.types import DocumentSearchHit, DocumentSearchParams, WebSearchResult


class DocumentSearcherProtocol(Protocol):
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

    def search(self, *, index: str, params: DocumentSearchParams) -> list[DocumentSearchHit]: ...


class DocumentRepositoryProtocol(Protocol):
    """Parent 문서 읽기 인터페이스.

    PDR 검색에서 Chunk → Parent 컨텍스트 조회에 사용.
    """

    def get_documents(self, doc_ids: list[str]) -> dict[str, dict[str, Any]]: ...


class EmbedderProtocol(Protocol):
    """텍스트 임베딩 인터페이스.

    텍스트를 벡터로 변환하는 임베더들이 구현해야 할 인터페이스.

    Example:
        >>> class SolarEmbedder:
        ...     def embed(self, text: str) -> list[float]:
        ...         ...  # Solar API 호출
        ...     def embed_batch(self, texts: list[str]) -> list[list[float]]:
        ...         ...  # 배치 처리
    """

    def embed(self, text: str) -> list[float]: ...

    def embed_batch(self, texts: list[str]) -> list[list[float]]: ...


class WebSearchClientProtocol(Protocol):
    """웹 검색 클라이언트 인터페이스.

    외부 웹 검색 API를 래핑하는 클라이언트들이 구현해야 할 인터페이스.

    Example:
        >>> class TavilyClientWrapper:
        ...     def search(self, query: str, max_results: int, **kwargs) -> list[WebSearchResult]:
        ...         ...  # Tavily API 호출
        >>>
        >>> class BingClientWrapper:
        ...     def search(self, query: str, max_results: int, **kwargs) -> list[WebSearchResult]:
        ...         ...  # Bing API 호출
    """

    def search(
        self,
        query: str,
        max_results: int = 10,
        **kwargs: Any,
    ) -> list[WebSearchResult]: ...
