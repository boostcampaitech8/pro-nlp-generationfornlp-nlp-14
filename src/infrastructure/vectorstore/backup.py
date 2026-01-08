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
    """로컬 JSONL 백업 파일 관리 (append-only)."""

    PARENTS_FILE = "parents_backup.jsonl"
    CHUNKS_FILE = "chunks_backup.jsonl"
    METADATA_FILE = "backup_metadata.json"

    def __init__(self, backup_dir: Path):
        self.backup_dir = backup_dir
        self._ensure_dir()

    def _ensure_dir(self) -> None:
        """백업 디렉토리 생성."""
        self.backup_dir.mkdir(parents=True, exist_ok=True)

    def _append_jsonl(self, filename: str, records: list[dict[str, Any]]) -> None:
        """JSONL 파일에 레코드 추가 (append)."""
        if not records:
            return
        path = self.backup_dir / filename
        with path.open("a", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _load_jsonl(self, filename: str, key_field: str) -> dict[str, dict[str, Any]]:
        """JSONL 파일 로드. 중복 키는 마지막 값으로 덮어씀."""
        path = self.backup_dir / filename
        if not path.exists():
            return {}
        result: dict[str, dict[str, Any]] = {}
        try:
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                        key = record.get(key_field)
                        if key:
                            result[key] = record
                    except json.JSONDecodeError:
                        continue
        except OSError as e:
            logger.warning(f"백업 파일 로드 실패 ({path}): {e}")
        return result

    def _update_metadata(self) -> None:
        """메타데이터 갱신 (라인 수 기반)."""

        def count_lines(filename: str) -> int:
            path = self.backup_dir / filename
            if not path.exists():
                return 0
            with path.open("r", encoding="utf-8") as f:
                return sum(1 for line in f if line.strip())

        metadata = {
            "last_updated": datetime.now(timezone.utc).isoformat(),
            "total_parents": count_lines(self.PARENTS_FILE),
            "total_chunks": count_lines(self.CHUNKS_FILE),
        }
        path = self.backup_dir / self.METADATA_FILE
        with path.open("w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

    # =========================================================================
    # Parents
    # =========================================================================

    def save_parent(self, doc: ParentDoc) -> None:
        """Parent 문서를 로컬에 저장 (append)."""
        self._append_jsonl(self.PARENTS_FILE, [doc.to_es()])

    def save_parents(self, docs: list[ParentDoc]) -> None:
        """여러 Parent 문서를 로컬에 저장 (append)."""
        self._append_jsonl(self.PARENTS_FILE, [doc.to_es() for doc in docs])

    def delete_parent(self, doc_id: str) -> None:
        """Parent 삭제 마커 추가."""
        self._append_jsonl(self.PARENTS_FILE, [{"doc_id": doc_id, "_deleted": True}])

    def load_all_parents(self) -> list[ParentDoc]:
        """모든 Parent 문서 로드 (중복 제거)."""
        data = self._load_jsonl(self.PARENTS_FILE, "doc_id")
        return [
            ParentDoc(
                doc_id=v["doc_id"],
                subject=v["subject"],
                topic=v["topic"],
                parent_text=v["parent_text"],
                version=v.get("version", "v1"),
                created_at=v.get("created_at"),
                topic_vector=v.get("topic_vector") or [],
                parent_vector=v.get("parent_vector") or [],
            )
            for v in data.values()
            if not v.get("_deleted") and "subject" in v
        ]

    # =========================================================================
    # Chunks
    # =========================================================================

    def save_chunk(self, doc: ChunkDoc) -> None:
        """Chunk 문서를 로컬에 저장 (append)."""
        self._append_jsonl(self.CHUNKS_FILE, [doc.to_es()])

    def save_chunks(self, docs: list[ChunkDoc]) -> None:
        """여러 Chunk 문서를 로컬에 저장 (append)."""
        self._append_jsonl(self.CHUNKS_FILE, [doc.to_es() for doc in docs])

    def delete_chunk(self, chunk_id: str) -> None:
        """Chunk 삭제 마커 추가."""
        self._append_jsonl(self.CHUNKS_FILE, [{"chunk_id": chunk_id, "_deleted": True}])

    def delete_chunks_by_doc_id(self, doc_id: str) -> int:
        """특정 Parent의 모든 Chunk 삭제 마커 추가."""
        self._append_jsonl(self.CHUNKS_FILE, [{"doc_id": doc_id, "_deleted_by_parent": True}])
        return 0

    def load_all_chunks(self) -> list[ChunkDoc]:
        """모든 Chunk 문서 로드 (중복 제거)."""
        data = self._load_jsonl(self.CHUNKS_FILE, "chunk_id")
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
            if not v.get("_deleted") and "subject" in v
        ]

    # =========================================================================
    # Utility
    # =========================================================================

    def get_metadata(self) -> dict[str, Any]:
        """백업 메타데이터 조회."""
        path = self.backup_dir / self.METADATA_FILE
        if not path.exists():
            return {}
        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
                return dict(data) if isinstance(data, dict) else {}
        except (json.JSONDecodeError, OSError):
            return {}

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

    ES의 모든 데이터를 JSONL 파일로 스트리밍 백업/복구합니다.
    """

    def __init__(self, es: Elasticsearch, cfg: ESConfig):
        self.es = es
        self.cfg = cfg
        self.parent_repo = ParentRepository(es, cfg)
        self.chunk_repo = ChunkRepository(es, cfg)

    def backup_to_jsonl(self, output_dir: str | Path) -> tuple[int, int]:
        """ES 데이터를 JSONL 형식으로 스트리밍 백업.

        Args:
            output_dir: 백업 디렉토리 경로

        Returns:
            (parents_count, chunks_count) 백업된 문서 수
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        parents_file = output_dir / "parents.jsonl"
        chunks_file = output_dir / "chunks.jsonl"
        metadata_file = output_dir / "metadata.json"

        # Parents 스트리밍 저장
        parents_count = 0
        with parents_file.open("w", encoding="utf-8") as f:
            for doc in self.parent_repo.scroll_all():
                f.write(json.dumps(doc.to_es(), ensure_ascii=False) + "\n")
                parents_count += 1

        # Chunks 스트리밍 저장
        chunks_count = 0
        with chunks_file.open("w", encoding="utf-8") as f:
            for chunk_doc in self.chunk_repo.scroll_all():
                f.write(json.dumps(chunk_doc.to_es(), ensure_ascii=False) + "\n")
                chunks_count += 1

        # 메타데이터 저장
        metadata = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "total_parents": parents_count,
            "total_chunks": chunks_count,
            "embedding_dims": self.cfg.embedding_dims,
            "parents_index": self.cfg.parents_index,
            "chunks_index": self.cfg.chunks_index,
        }
        with metadata_file.open("w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        return parents_count, chunks_count

    def restore_from_jsonl(self, backup_dir: str | Path, refresh: bool = True) -> tuple[int, int]:
        """JSONL 백업에서 ES로 복구.

        Args:
            backup_dir: 백업 디렉토리 경로
            refresh: 복구 후 인덱스 refresh 여부

        Returns:
            (parents_count, chunks_count) 복구된 문서 수
        """
        backup_dir = Path(backup_dir)
        parents_file = backup_dir / "parents.jsonl"
        chunks_file = backup_dir / "chunks.jsonl"

        # Parents 복구
        parents_count = 0
        if parents_file.exists():
            batch: list[ParentDoc] = []
            with parents_file.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    p = json.loads(line)
                    batch.append(
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
                    )
                    if len(batch) >= 100:
                        self.parent_repo.bulk_upsert(batch, refresh=False)
                        parents_count += len(batch)
                        batch = []
            if batch:
                self.parent_repo.bulk_upsert(batch, refresh=False)
                parents_count += len(batch)

        # Chunks 복구
        chunks_count = 0
        if chunks_file.exists():
            batch_chunks: list[ChunkDoc] = []
            with chunks_file.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    c = json.loads(line)
                    batch_chunks.append(
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
                    )
                    if len(batch_chunks) >= 100:
                        self.chunk_repo.bulk_upsert(batch_chunks, refresh=False)
                        chunks_count += len(batch_chunks)
                        batch_chunks = []
            if batch_chunks:
                self.chunk_repo.bulk_upsert(batch_chunks, refresh=refresh)
                chunks_count += len(batch_chunks)

        return parents_count, chunks_count

    def get_backup_info(self, backup_dir: str | Path) -> dict[str, Any]:
        """백업 디렉토리 메타데이터 조회."""
        metadata_file = Path(backup_dir) / "metadata.json"

        if not metadata_file.exists():
            raise FileNotFoundError(f"메타데이터 파일이 없습니다: {metadata_file}")

        with metadata_file.open("r", encoding="utf-8") as f:
            data = json.load(f)
            return dict(data) if isinstance(data, dict) else {}
