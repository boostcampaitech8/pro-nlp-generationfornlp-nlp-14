"""Elasticsearch 기반 벡터스토어 (Repository Layer).

이 모듈은 Elasticsearch 기반 CRUD 및 검색 기능을 제공합니다.

주요 컴포넌트:
    - Repository: ParentRepository, ChunkRepository (CRUD)
    - HybridSearcher: ES 하이브리드 검색 구현체
    - PDRRetriever: 2단계 하이브리드 검색 전략 (Parent → Chunk)
    - WriteThroughStore: DB 쓰기 시 로컬 백업 동시 저장
    - VectorStoreBackup: 전체 스냅샷 백업/복구

Usage:
    >>> from src.vectorstore import ESConfig, create_es_client
    >>> from src.vectorstore import ParentRepository, ChunkRepository
    >>> from src.vectorstore import HybridSearcher, PDRRetriever, PDRConfig
    >>>
    >>> cfg = ESConfig()
    >>> es = create_es_client(cfg)
    >>>
    >>> # CRUD
    >>> parent_repo = ParentRepository(es, cfg)
    >>> chunk_repo = ChunkRepository(es, cfg)
    >>> parent_repo.upsert(parent_doc)
    >>>
    >>> # 검색 (PDR 전략)
    >>> searcher = HybridSearcher(es, cfg)
    >>> pdr = PDRRetriever(searcher, parent_repo, PDRConfig(
    ...     parents_index=cfg.parents_index,
    ...     chunks_index=cfg.chunks_index,
    ... ))
    >>> parents = pdr.search_parents(query="수학 함수", query_vector=emb)
    >>> chunks = pdr.search_chunks(query="미분", query_vector=emb, doc_ids=[...])

마이그레이션 (인덱스 DDL):
    $ python -m migrations.migrate create   # 인덱스 생성
    $ python -m migrations.migrate status   # 상태 확인
"""

from .backup import LocalBackupManager, VectorStoreBackup, WriteThroughStore
from .client import check_connection, create_es_client
from .config import ESConfig
from .documents import ChunkDoc, ParentDoc
from .repository import BaseRepository, ChunkRepository, ParentRepository
from .search import HybridSearcher, PDRConfig, PDRRetriever, SearchHit, SearchParams

__all__ = [
    # Config
    "ESConfig",
    # Client
    "create_es_client",
    "check_connection",
    # Documents
    "ParentDoc",
    "ChunkDoc",
    # Search Types
    "SearchHit",
    "SearchParams",
    # Repository
    "BaseRepository",
    "ParentRepository",
    "ChunkRepository",
    # Search
    "HybridSearcher",
    "PDRConfig",
    "PDRRetriever",
    # Backup
    "LocalBackupManager",
    "WriteThroughStore",
    "VectorStoreBackup",
]
