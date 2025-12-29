from dataclasses import dataclass
from typing import Literal


@dataclass
class RetrievalResponse:
    type: Literal["web", "local"]
    question: str
    context: str
    sources: list[str] | None = None  # optional but very useful
