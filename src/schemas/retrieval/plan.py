from pydantic import BaseModel, Field

from schemas.retrieval.request import RetrievalRequest


class RetrievalPlan(BaseModel):
    """
    검색할 쿼리를 얻기위해 LLM을 이용해 검색쿼리를 증강하기 위한 structed output schema 입니다
    - Request[RetrievealResponse]: {
        query: str,
        tok_k: int,
    }
    """

    requests: list[RetrievalRequest] = Field(default_factory=list)
