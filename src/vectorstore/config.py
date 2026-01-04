"""Elasticsearch 설정 관리.

환경변수로 설정을 관리합니다.
로컬 서버 환경만 지원합니다 (Cloud 미사용).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def _get_backup_dir() -> Path:
    """기본 백업 디렉토리 경로."""
    return Path(os.getenv("ES_BACKUP_DIR", "./data/vectorstore_backup"))


@dataclass(frozen=True)
class ESConfig:
    """Elasticsearch 연결 및 인덱스 설정.

    Attributes:
        es_url: Elasticsearch 서버 URL (예: http://localhost:9200)
        es_username: HTTP Basic Auth 사용자명 (선택)
        es_password: HTTP Basic Auth 비밀번호 (선택)
        verify_certs: SSL 인증서 검증 여부
        request_timeout_s: 요청 타임아웃 (초)
        parents_index: Parent 문서 인덱스명
        chunks_index: Chunk 문서 인덱스명
        backup_dir: 로컬 백업 저장 디렉토리
        enable_write_through: DB 쓰기 시 로컬 파일 동시 저장 여부
    """

    # Connection
    es_url: str = field(default_factory=lambda: os.environ["ES_URL"])
    es_username: str | None = field(default_factory=lambda: os.getenv("ES_USERNAME"))
    es_password: str | None = field(default_factory=lambda: os.getenv("ES_PASSWORD"))

    verify_certs: bool = field(
        default_factory=lambda: os.getenv("ES_VERIFY_CERTS", "true").lower() == "true"
    )
    request_timeout_s: int = field(
        default_factory=lambda: int(os.getenv("ES_REQUEST_TIMEOUT_S", "30"))
    )

    # Index names
    parents_index: str = field(
        default_factory=lambda: os.getenv("ES_PARENTS_INDEX", "kb_parents_v1")
    )
    chunks_index: str = field(default_factory=lambda: os.getenv("ES_CHUNKS_INDEX", "kb_chunks_v1"))

    # Embedding (필수 - EMBEDDING_DIMS 환경변수에서 읽음)
    embedding_dims: int = field(default_factory=lambda: int(os.environ["EMBEDDING_DIMS"]))

    # Backup (write-through 패턴)
    backup_dir: Path = field(default_factory=_get_backup_dir)
    enable_write_through: bool = field(
        default_factory=lambda: os.getenv("ES_ENABLE_WRITE_THROUGH", "true").lower() == "true"
    )

    def get_backup_path(self, filename: str) -> Path:
        """백업 파일 전체 경로 반환."""
        return self.backup_dir / filename
