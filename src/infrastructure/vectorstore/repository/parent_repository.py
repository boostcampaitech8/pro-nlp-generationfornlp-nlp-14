"""Parent document repository."""

from __future__ import annotations

from typing import Any

from infrastructure.vectorstore.documents import ParentDoc
from infrastructure.vectorstore.repository.base import BaseRepository


class ParentRepository(BaseRepository[ParentDoc]):
    """Parent 문서(지문/문단 단위) CRUD 리포지토리."""

    @property
    def index_name(self) -> str:
        return self.cfg.parents_index

    def _get_doc_id(self, doc: ParentDoc) -> str:
        return doc.doc_id

    def _to_es_dict(self, doc: ParentDoc) -> dict[str, Any]:
        return doc.to_es()

    def _from_es_dict(self, source: dict[str, Any]) -> ParentDoc:
        return ParentDoc(
            doc_id=source["doc_id"],
            subject=source["subject"],
            topic=source["topic"],
            parent_text=source["parent_text"],
            topic_vector=source["topic_vector"],
            parent_vector=source["parent_vector"],
            version=source.get("version", "v1"),
            created_at=source.get("created_at"),
        )
