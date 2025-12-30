"""Search layer for Elasticsearch hybrid and vector search."""

from .hybrid_search import HybridSearcher
from .protocols import ParentReaderProtocol, SearcherProtocol
from .types import SearchHit, SearchParams

__all__ = [
    # 공용 타입 (상위 레이어에서 사용)
    "SearchHit",
    "SearchParams",
    # 인터페이스 (Protocol)
    "SearcherProtocol",
    "ParentReaderProtocol",
    # Hybrid 구현체
    "HybridSearcher",
]
