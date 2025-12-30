from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class ParentDoc:
    doc_id: str
    subject: str
    topic: str
    parent_text: str
    topic_vector: list[float]
    parent_vector: list[float]

    # optional
    version: str = "v1"
    created_at: str = utcnow_iso()

    def to_es(self) -> dict[str, Any]:
        d = asdict(self)
        # ES에 None은 굳이 넣지 않는 게 깔끔
        return {k: v for k, v in d.items() if v is not None}


@dataclass(frozen=True)
class ChunkDoc:
    chunk_id: str
    doc_id: str
    subject: str
    topic: str
    chunk_idx: int
    chunk_text: str
    chunk_vector: list[float]

    # optional
    start_char: int | None = None
    end_char: int | None = None
    version: str = "v1"
    created_at: str = utcnow_iso()

    def to_es(self) -> dict[str, Any]:
        d = asdict(self)
        return {k: v for k, v in d.items() if v is not None}
