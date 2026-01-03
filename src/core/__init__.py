"""Core 타입 및 프로토콜.

이 모듈은 인프라에 의존하지 않습니다.
vectorstore, chains, indexing 등 어디서든 import할 수 있습니다.
"""

from core.protocols import (
    EmbedderProtocol,
    ParentReaderProtocol,
    SearcherProtocol,
)
from core.types import SearchHit, SearchParams

__all__ = [
    # Types
    "SearchHit",
    "SearchParams",
    # Protocols
    "EmbedderProtocol",
    "SearcherProtocol",
    "ParentReaderProtocol",
]
