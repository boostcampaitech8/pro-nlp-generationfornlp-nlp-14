from dataclasses import dataclass, field
from typing import Literal


@dataclass
class RetrievalRequest:
    type: Literal["web", "local"] = "web"  # local == db/vecstore 등으로 확장 가능
    query: str = ""
    top_k: int = 5


@dataclass
class RetrievalPlan:
    requests: list[RetrievalRequest] = field(default_factory=list)
    need_external_knowledge: bool = False
    final_query: str = ""
