from typing import Literal

from pydantic import BaseModel, Field

from schemas.retrieval.request import RetrievalRequest


class RetrievalPlan(BaseModel):
    """
    검색할 쿼리를 얻기위해 LLM을 이용해 검색쿼리를 증강하기 위한 structed output schema 입니다
    - subject: 질문의 주제 분류 (한국사/정치와법/일반)
    - requests: 검색 요청 리스트 [{query: str, top_k: int}]
    """

    reasoning_steps: list[str] = Field(
        description="쿼리 생성에 이르는 사고 과정을 자유롭게 작성하는 공간입니다. "
        "질문을 분석하고, 검색 필요성을 판단하며, 효과적인 쿼리를 생성하는 과정을 논리적으로 추론하세요."
    )
    subject: Literal["한국사", "정치와 법", "general"] = Field(
        default="general",
        description="질문의 과목 분류. '한국사'는 역사적 사건/인물/제도, '정치와 법'은 현대 정치/법률/헌법, 'general'은 그 외 모든 주제",
    )
    requests: list[RetrievalRequest] = Field(
        default_factory=list,
        description="검색할 쿼리 리스트. 각 요청은 query(검색어)와 top_k(결과 개수)를 포함",
    )
