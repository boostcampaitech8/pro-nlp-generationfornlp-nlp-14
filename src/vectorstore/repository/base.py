"""Base repository with common Elasticsearch operations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import Any, Generic, TypeVar

from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk

from vectorstore.config import ESConfig

T = TypeVar("T")


class BaseRepository(ABC, Generic[T]):
    """Elasticsearch 문서 CRUD를 위한 기본 리포지토리."""

    def __init__(self, es: Elasticsearch, cfg: ESConfig):
        self.es = es
        self.cfg = cfg

    @property
    @abstractmethod
    def index_name(self) -> str:
        """대상 인덱스 이름."""
        ...

    @abstractmethod
    def _get_doc_id(self, doc: T) -> str:
        """문서의 ID를 반환."""
        ...

    @abstractmethod
    def _to_es_dict(self, doc: T) -> dict[str, Any]:
        """문서를 ES용 dict로 변환."""
        ...

    @abstractmethod
    def _from_es_dict(self, source: dict[str, Any]) -> T:
        """ES source를 도메인 객체로 변환."""
        ...

    def upsert(self, doc: T, refresh: bool = False) -> None:
        """단일 문서 upsert."""
        self.es.index(
            index=self.index_name,
            id=self._get_doc_id(doc),
            document=self._to_es_dict(doc),
            refresh="wait_for" if refresh else False,
        )

    def bulk_upsert(self, docs: Iterable[T], refresh: bool = False) -> int:
        """대량 문서 upsert. 성공 건수 반환."""
        actions = [
            {
                "_op_type": "index",
                "_index": self.index_name,
                "_id": self._get_doc_id(d),
                "_source": self._to_es_dict(d),
            }
            for d in docs
        ]
        if not actions:
            return 0
        ok, _ = bulk(self.es, actions, refresh="wait_for" if refresh else False)
        return int(ok)

    def get(self, doc_id: str) -> T | None:
        """ID로 단일 문서 조회."""
        try:
            resp = self.es.get(index=self.index_name, id=doc_id)
            return self._from_es_dict(resp["_source"])
        except Exception:
            return None

    def mget(self, doc_ids: list[str]) -> dict[str, T]:
        """여러 ID로 문서 조회. {id: doc} 형태 반환."""
        if not doc_ids:
            return {}
        resp = self.es.mget(index=self.index_name, ids=doc_ids)
        out: dict[str, T] = {}
        for d in resp.get("docs", []):
            if d.get("found"):
                out[d["_id"]] = self._from_es_dict(d["_source"])
        return out

    def mget_raw(self, doc_ids: list[str]) -> dict[str, dict[str, Any]]:
        """여러 ID로 문서 조회 (raw dict 반환)."""
        if not doc_ids:
            return {}
        resp = self.es.mget(index=self.index_name, ids=doc_ids)
        out: dict[str, dict[str, Any]] = {}
        for d in resp.get("docs", []):
            if d.get("found"):
                out[d["_id"]] = d["_source"]
        return out

    def get_documents(self, doc_ids: list[str]) -> dict[str, dict[str, Any]]:
        """여러 ID로 문서 조회 (DocumentRepositoryProtocol 호환용 alias)."""
        return self.mget_raw(doc_ids)

    def delete(self, doc_id: str, refresh: bool = False) -> None:
        """문서 삭제."""
        self.es.delete(
            index=self.index_name,
            id=doc_id,
            refresh="wait_for" if refresh else False,
        )

    def count(self) -> int:
        """인덱스 내 총 문서 수."""
        resp = self.es.count(index=self.index_name)
        return int(resp["count"])

    def scroll_all(self, batch_size: int = 1000) -> Iterable[T]:
        """인덱스 내 모든 문서를 스크롤하며 반환."""
        resp = self.es.search(
            index=self.index_name,
            query={"match_all": {}},
            size=batch_size,
            scroll="2m",
        )
        scroll_id = resp["_scroll_id"]
        hits = resp["hits"]["hits"]

        try:
            while hits:
                for h in hits:
                    yield self._from_es_dict(h["_source"])
                resp = self.es.scroll(scroll_id=scroll_id, scroll="2m")
                scroll_id = resp["_scroll_id"]
                hits = resp["hits"]["hits"]
        finally:
            self.es.clear_scroll(scroll_id=scroll_id)
