from dataclasses import dataclass, field

from schemas.retrieval.request import RetrievalRequest


@dataclass
class RetrievalPlan:
    """
    검색할 쿼리를 얻기위해 LLM을 이용해 검색쿼리를 증강하기 위한 structed output schema 입니다
    - Request[RetrievealResponse]: {
        query: str,
        tok_k: int,
    }
    """

    requests: list[RetrievalRequest] = field(default_factory=list)
