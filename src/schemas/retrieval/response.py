from dataclasses import dataclass
from typing import Any


@dataclass
class RetrievalResponse:
    question: str
    context: str
    metadata: dict[str, Any] | None = None
