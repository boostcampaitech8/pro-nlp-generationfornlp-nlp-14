"""Elasticsearch 기반 벡터스토어 (Repository Layer).

이 모듈은 Elasticsearch 기반 CRUD 및 검색 기능을 제공합니다.

주요 컴포넌트:
    - Repository: ParentRepository, ChunkRepository (CRUD)
    - HybridSearcher: ES 하이브리드 검색 구현체
    - WriteThroughStore: DB 쓰기 시 로컬 백업 동시 저장
    - VectorStoreBackup: 전체 스냅샷 백업/복구

Usage:
    >>> from src.vectorstore import ESConfig, create_es_client
    >>> from src.vectorstore import ParentRepository, ChunkRepository
    >>> from src.vectorstore import HybridSearcher
    >>>
    >>> cfg = ESConfig()
    >>> es = create_es_client(cfg)
    >>>
    >>> # CRUD
    >>> parent_repo = ParentRepository(es, cfg)
    >>> chunk_repo = ChunkRepository(es, cfg)
    >>> parent_repo.upsert(parent_doc)
"""

from infrastructure.vectorstore.backup import (
    LocalBackupManager,
    VectorStoreBackup,
    WriteThroughStore,
)
from infrastructure.vectorstore.client import check_connection, create_es_client
from infrastructure.vectorstore.config import ESConfig
from infrastructure.vectorstore.documents import ChunkDoc, ParentDoc
from infrastructure.vectorstore.repository import BaseRepository, ChunkRepository, ParentRepository
from infrastructure.vectorstore.search import HybridSearcher

__all__ = [
    # Config
    "ESConfig",
    # Client
    "create_es_client",
    "check_connection",
    # Documents
    "ParentDoc",
    "ChunkDoc",
    # Repository
    "BaseRepository",
    "ParentRepository",
    "ChunkRepository",
    # Search
    "HybridSearcher",
    # Backup
    "LocalBackupManager",
    "WriteThroughStore",
    "VectorStoreBackup",
]
