"""Elasticsearch 인덱스 마이그레이션 모듈.

인프라 설정을 위한 독립적인 모듈입니다.
src/vectorstore와 의존성이 없습니다.

Usage:
    python -m migrations.migrate create
    python -m migrations.migrate status
    python -m migrations.migrate drop --confirm
"""

from .mappings import (
    INDEX_MAPPINGS,
    chunks_index_mapping,
    parents_index_mapping,
)
from .migrate import (
    IndexInfo,
    MigrationConfig,
    Migrator,
    create_es_client,
)

__all__ = [
    # Mappings
    "parents_index_mapping",
    "chunks_index_mapping",
    "INDEX_MAPPINGS",
    # Migration
    "MigrationConfig",
    "Migrator",
    "IndexInfo",
    "create_es_client",
]
