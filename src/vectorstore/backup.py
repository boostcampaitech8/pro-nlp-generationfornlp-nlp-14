"""벡터스토어 백업/복구 및 Write-Through 캐싱.

DB 쓰기 시 로컬 JSON 파일에도 동시 저장하여 서버 불안정에 대비합니다.

Write-Through 패턴:
    - DB에 쓰기 전 로컬 백업 파일에 먼저 저장
    - DB 쓰기 실패 시에도 로컬 데이터 보존
    - 복구 시 로컬 파일에서 DB로 복원

Usage:
    >>> store = WriteThroughStore(es, cfg)
    >>> store.upsert_parent(doc)  # ES + 로컬 파일 동시 저장
    >>> store.bulk_upsert_chunks(chunks)  # 마찬가지
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from elasticsearch import Elasticsearch

from vectorstore.config import ESConfig
from vectorstore.documents import ChunkDoc, ParentDoc
from vectorstore.repository import ChunkRepository, ParentRepository

logger = logging.getLogger(__name__)


class LocalBackupManager:
    """로컬 JSON 백업 파일 관리."""

    PARENTS_FILE = "parents_backup.json"
    CHUNKS_FILE = "chunks_backup.json"
    METADATA_FILE = "backup_metadata.json"

    def __init__(self, backup_dir: Path):
        self.backup_dir = backup_dir
        self._ensure_dir()

    def _ensure_dir(self) -> None:
        """백업 디렉토리 생성."""
        self.backup_dir.mkdir(parents=True, exist_ok=True)

    def _load_json(self, filename: str) -> dict[str, Any]:
        """JSON 파일 로드. 없으면 빈 딕셔너리 반환."""
        path = self.backup_dir / filename
        if not path.exists():
            return {}
        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
                return dict(data) if isinstance(data, dict) else {}
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"백업 파일 로드 실패 ({path}): {e}")
            return {}

    def _save_json(self, filename: str, data: dict[str, Any]) -> None:
        """JSON 파일 저장."""
        path = self.backup_dir / filename
        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def _update_metadata(self) -> None:
        """메타데이터 갱신."""
        parents = self._load_json(self.PARENTS_FILE)
        chunks = self._load_json(self.CHUNKS_FILE)
        metadata = {
            "last_updated": datetime.now(timezone.utc).isoformat(),
            "total_parents": len(parents),
            "total_chunks": len(chunks),
        }
        self._save_json(self.METADATA_FILE, metadata)

    # =========================================================================
    # Parents
    # =========================================================================

    def save_parent(self, doc: ParentDoc) -> None:
        """Parent 문서를 로컬에 저장."""
        data = self._load_json(self.PARENTS_FILE)
        data[doc.doc_id] = doc.to_es()
        self._save_json(self.PARENTS_FILE, data)
        self._update_metadata()

    def save_parents(self, docs: list[ParentDoc]) -> None:
        """여러 Parent 문서를 로컬에 저장."""
        data = self._load_json(self.PARENTS_FILE)
        for doc in docs:
            data[doc.doc_id] = doc.to_es()
        self._save_json(self.PARENTS_FILE, data)
        self._update_metadata()

    def delete_parent(self, doc_id: str) -> None:
        """Parent 문서를 로컬에서 삭제."""
        data = self._load_json(self.PARENTS_FILE)
        if doc_id in data:
            del data[doc_id]
            self._save_json(self.PARENTS_FILE, data)
            self._update_metadata()

    def load_all_parents(self) -> list[ParentDoc]:
        """모든 Parent 문서 로드."""
        data = self._load_json(self.PARENTS_FILE)
        return [
            ParentDoc(
                doc_id=v["doc_id"],
                subject=v["subject"],
                topic=v["topic"],
                parent_text=v["parent_text"],
                version=v.get("version", "v1"),
                created_at=v.get("created_at"),
                topic_vector=v.get("topic_vector"),
                parent_vector=v.get("parent_vector"),
            )
            for v in data.values()
        ]

    # =========================================================================
    # Chunks
    # =========================================================================

    def save_chunk(self, doc: ChunkDoc) -> None:
        """Chunk 문서를 로컬에 저장."""
        data = self._load_json(self.CHUNKS_FILE)
        data[doc.chunk_id] = doc.to_es()
        self._save_json(self.CHUNKS_FILE, data)
        self._update_metadata()

    def save_chunks(self, docs: list[ChunkDoc]) -> None:
        """여러 Chunk 문서를 로컬에 저장."""
        data = self._load_json(self.CHUNKS_FILE)
        for doc in docs:
            data[doc.chunk_id] = doc.to_es()
        self._save_json(self.CHUNKS_FILE, data)
        self._update_metadata()

    def delete_chunk(self, chunk_id: str) -> None:
        """Chunk 문서를 로컬에서 삭제."""
        data = self._load_json(self.CHUNKS_FILE)
        if chunk_id in data:
            del data[chunk_id]
            self._save_json(self.CHUNKS_FILE, data)
            self._update_metadata()

    def delete_chunks_by_doc_id(self, doc_id: str) -> int:
        """특정 Parent의 모든 Chunk를 로컬에서 삭제."""
        data = self._load_json(self.CHUNKS_FILE)
        to_delete = [k for k, v in data.items() if v.get("doc_id") == doc_id]
        for k in to_delete:
            del data[k]
        if to_delete:
            self._save_json(self.CHUNKS_FILE, data)
            self._update_metadata()
        return len(to_delete)

    def load_all_chunks(self) -> list[ChunkDoc]:
        """모든 Chunk 문서 로드."""
        data = self._load_json(self.CHUNKS_FILE)
        return [
            ChunkDoc(
                chunk_id=v["chunk_id"],
                doc_id=v["doc_id"],
                subject=v["subject"],
                topic=v["topic"],
                chunk_idx=v["chunk_idx"],
                chunk_text=v["chunk_text"],
                chunk_vector=v["chunk_vector"],
                start_char=v.get("start_char"),
                end_char=v.get("end_char"),
                version=v.get("version", "v1"),
                created_at=v.get("created_at"),
            )
            for v in data.values()
        ]

    # =========================================================================
    # Utility
    # =========================================================================

    def get_metadata(self) -> dict[str, Any]:
        """백업 메타데이터 조회."""
        return self._load_json(self.METADATA_FILE)

    def clear_all(self) -> None:
        """모든 로컬 백업 삭제."""
        for filename in [self.PARENTS_FILE, self.CHUNKS_FILE, self.METADATA_FILE]:
            path = self.backup_dir / filename
            if path.exists():
                path.unlink()


class WriteThroughStore:
    """Write-Through 패턴을 적용한 벡터스토어.

    ES에 쓰기 전 로컬에 먼저 저장하여 서버 불안정에 대비합니다.
    """

    def __init__(self, es: Elasticsearch, cfg: ESConfig):
        self.es = es
        self.cfg = cfg

        # Repositories
        self.parent_repo = ParentRepository(es, cfg)
        self.chunk_repo = ChunkRepository(es, cfg)

        # Local backup (write-through 활성화 시)
        self._backup: LocalBackupManager | None = None
        if cfg.enable_write_through:
            self._backup = LocalBackupManager(cfg.backup_dir)

    # =========================================================================
    # Parents CRUD (with write-through)
    # =========================================================================

    def upsert_parent(self, doc: ParentDoc, refresh: bool = False) -> None:
        """Parent 문서 upsert (로컬 + ES)."""
        if self._backup:
            self._backup.save_parent(doc)
        self.parent_repo.upsert(doc, refresh=refresh)

    def bulk_upsert_parents(self, docs: list[ParentDoc], refresh: bool = False) -> int:
        """Parent 대량 upsert (로컬 + ES)."""
        if self._backup:
            self._backup.save_parents(docs)
        return self.parent_repo.bulk_upsert(docs, refresh=refresh)

    def get_parent(self, doc_id: str) -> ParentDoc | None:
        """Parent 문서 조회."""
        return self.parent_repo.get(doc_id)

    def delete_parent(self, doc_id: str, refresh: bool = False) -> None:
        """Parent 문서 삭제 (로컬 + ES)."""
        if self._backup:
            self._backup.delete_parent(doc_id)
        self.parent_repo.delete(doc_id, refresh=refresh)

    # =========================================================================
    # Chunks CRUD (with write-through)
    # =========================================================================

    def upsert_chunk(self, doc: ChunkDoc, refresh: bool = False) -> None:
        """Chunk 문서 upsert (로컬 + ES)."""
        if self._backup:
            self._backup.save_chunk(doc)
        self.chunk_repo.upsert(doc, refresh=refresh)

    def bulk_upsert_chunks(self, docs: list[ChunkDoc], refresh: bool = False) -> int:
        """Chunk 대량 upsert (로컬 + ES)."""
        if self._backup:
            self._backup.save_chunks(docs)
        return self.chunk_repo.bulk_upsert(docs, refresh=refresh)

    def delete_chunk(self, chunk_id: str, refresh: bool = False) -> None:
        """Chunk 문서 삭제 (로컬 + ES)."""
        if self._backup:
            self._backup.delete_chunk(chunk_id)
        self.chunk_repo.delete(chunk_id, refresh=refresh)

    # =========================================================================
    # Restore from local backup
    # =========================================================================

    def restore_from_local(self, refresh: bool = True) -> tuple[int, int]:
        """로컬 백업에서 ES로 복구.

        Returns:
            (parents_count, chunks_count) 복구된 문서 수
        """
        if not self._backup:
            raise RuntimeError("Write-through가 비활성화되어 있습니다.")

        parents = self._backup.load_all_parents()
        chunks = self._backup.load_all_chunks()

        parents_count = self.parent_repo.bulk_upsert(parents, refresh=False)
        chunks_count = self.chunk_repo.bulk_upsert(chunks, refresh=refresh)

        return parents_count, chunks_count

    def get_backup_metadata(self) -> dict[str, Any]:
        """로컬 백업 메타데이터 조회."""
        if not self._backup:
            return {"enabled": False}
        meta = self._backup.get_metadata()
        meta["enabled"] = True
        meta["backup_dir"] = str(self.cfg.backup_dir)
        return meta


class VectorStoreBackup:
    """벡터스토어 전체 백업/복구 유틸리티.

    ES의 모든 데이터를 단일 JSON 파일로 백업/복구합니다.
    WriteThroughStore의 점진적 백업과 달리, 전체 스냅샷 용도입니다.
    """

    def __init__(self, es: Elasticsearch, cfg: ESConfig):
        self.es = es
        self.cfg = cfg
        self.parent_repo = ParentRepository(es, cfg)
        self.chunk_repo = ChunkRepository(es, cfg)

    def backup_to_json(self, backup_path: str | Path) -> int:
        """ES 데이터를 단일 JSON 파일로 백업.

        Args:
            backup_path: 백업 파일 경로

        Returns:
            백업된 총 문서 수
        """
        backup_path = Path(backup_path)

        # 모든 문서 수집
        parents = list(self.parent_repo.scroll_all())
        chunks = list(self.chunk_repo.scroll_all())

        backup_data = {
            "metadata": {
                "created_at": datetime.now(timezone.utc).isoformat(),
                "total_parents": len(parents),
                "total_chunks": len(chunks),
                "embedding_dims": self.cfg.embedding_dims,
                "parents_index": self.cfg.parents_index,
                "chunks_index": self.cfg.chunks_index,
            },
            "parents": [p.to_es() for p in parents],
            "chunks": [c.to_es() for c in chunks],
        }

        backup_path.parent.mkdir(parents=True, exist_ok=True)
        with backup_path.open("w", encoding="utf-8") as f:
            json.dump(backup_data, f, ensure_ascii=False, indent=2)

        return len(parents) + len(chunks)

    def restore_from_json(self, backup_path: str | Path, refresh: bool = True) -> tuple[int, int]:
        """JSON 파일에서 ES로 복구.

        Args:
            backup_path: 백업 파일 경로
            refresh: 복구 후 인덱스 refresh 여부

        Returns:
            (parents_count, chunks_count) 복구된 문서 수
        """
        backup_path = Path(backup_path)

        with backup_path.open("r", encoding="utf-8") as f:
            backup_data = json.load(f)

        # Parents 복구
        parent_docs = [
            ParentDoc(
                doc_id=p["doc_id"],
                subject=p["subject"],
                topic=p["topic"],
                parent_text=p["parent_text"],
                version=p.get("version", "v1"),
                created_at=p.get("created_at"),
                topic_vector=p.get("topic_vector"),
                parent_vector=p.get("parent_vector"),
            )
            for p in backup_data["parents"]
        ]
        parents_count = self.parent_repo.bulk_upsert(parent_docs, refresh=False)

        # Chunks 복구
        chunk_docs = [
            ChunkDoc(
                chunk_id=c["chunk_id"],
                doc_id=c["doc_id"],
                subject=c["subject"],
                topic=c["topic"],
                chunk_idx=c["chunk_idx"],
                chunk_text=c["chunk_text"],
                chunk_vector=c["chunk_vector"],
                start_char=c.get("start_char"),
                end_char=c.get("end_char"),
                version=c.get("version", "v1"),
                created_at=c.get("created_at"),
            )
            for c in backup_data["chunks"]
        ]
        chunks_count = self.chunk_repo.bulk_upsert(chunk_docs, refresh=refresh)

        return parents_count, chunks_count

    def get_backup_info(self, backup_path: str | Path) -> dict[str, Any]:
        """백업 파일 정보 조회."""
        backup_path = Path(backup_path)

        if not backup_path.exists():
            raise FileNotFoundError(f"백업 파일이 없습니다: {backup_path}")

        with backup_path.open("r", encoding="utf-8") as f:
            backup_data = json.load(f)

        metadata = backup_data.get("metadata", {})
        return dict(metadata) if isinstance(metadata, dict) else {}
