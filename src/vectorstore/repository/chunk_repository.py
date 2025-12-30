"""Chunk document repository."""

from __future__ import annotations

from typing import Any

from ..documents import ChunkDoc
from .base import BaseRepository


class ChunkRepository(BaseRepository[ChunkDoc]):
    """Chunk 문서(청크 단위) CRUD 리포지토리."""

    @property
    def index_name(self) -> str:
        return self.cfg.chunks_index

    def _get_doc_id(self, doc: ChunkDoc) -> str:
        return doc.chunk_id

    def _to_es_dict(self, doc: ChunkDoc) -> dict[str, Any]:
        return doc.to_es()

    def _from_es_dict(self, source: dict[str, Any]) -> ChunkDoc:
        return ChunkDoc(
            chunk_id=source["chunk_id"],
            doc_id=source["doc_id"],
            subject=source["subject"],
            topic=source["topic"],
            chunk_idx=source["chunk_idx"],
            chunk_text=source["chunk_text"],
            chunk_vector=source["chunk_vector"],
            start_char=source.get("start_char"),
            end_char=source.get("end_char"),
            version=source.get("version", "v1"),
            created_at=source.get("created_at"),
        )

    def get_by_doc_id(self, doc_id: str) -> list[ChunkDoc]:
        """특정 parent의 모든 chunk 조회."""
        resp = self.es.search(
            index=self.index_name,
            query={"term": {"doc_id": doc_id}},
            size=1000,
            sort=[{"chunk_idx": {"order": "asc"}}],
        )
        return [self._from_es_dict(h["_source"]) for h in resp["hits"]["hits"]]

    def delete_by_doc_id(self, doc_id: str, refresh: bool = False) -> int:
        """특정 parent의 모든 chunk 삭제."""
        resp = self.es.delete_by_query(
            index=self.index_name,
            query={"term": {"doc_id": doc_id}},
            refresh=refresh,
        )
        return int(resp.get("deleted", 0))
