"""
Pipeline 전체에서 사용되는 TypedDict 기반 state 정의.

이 모듈은 pipeline의 모든 중간 state를 TypedDict로 정의하여
런타임 오버헤드 없이 타입 안정성을 제공합니다.
기존의 dataclass + dict + TypedDict 혼용 방식을 대체합니다.

Pipeline flow:
    QuestionState → PlanningState → RetrievalState → QAInputState
    → ForwardState → DecodedState → OutputState
"""

from typing import Any, TypedDict

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
from typing_extensions import NotRequired

from schemas.retrieval import RetrievalPlan, RetrievalResponse


class QuestionState(TypedDict):
    """
    입력 질문 데이터 (기존 ProcessedQuestion을 TypedDict로 변환).

    Pipeline의 시작점이며, 모든 단계에서 data 필드로 전달됩니다.
    """

    id: str
    paragraph: str
    question: str
    choices: list[str]
    question_plus: NotRequired[str | None]
    len_choices: int


class PlanningState(TypedDict):
    """
    Planning 단계 출력.

    Planner LLM이 검색 쿼리 계획을 생성한 결과입니다.
    """

    data: QuestionState
    plan: RetrievalPlan


class RetrievalState(TypedDict):
    """
    Retrieval 단계 출력.

    검색 서비스로부터 문서를 가져와 context로 포맷팅한 결과입니다.
    """

    data: QuestionState
    plan: RetrievalPlan
    external_knowledge: list[RetrievalResponse]
    context: str


class QAInputState(TypedDict):
    """
    QA chain 입력 (기존 McqRequest).

    Prompt가 빌드되어 LLM에 전달할 준비가 된 상태입니다.
    """

    id: str
    messages: list[BaseMessage]
    len_choices: int


class ForwardState(TypedDict):
    """
    Forward pass 출력 (기존 ForwardContext).

    LLM 추론 결과로 각 선택지의 점수를 포함합니다.
    """

    data: QAInputState
    score: list[float]


class DecodedState(ForwardState):
    """
    Decoded 출력 (기존 DecodedContext).

    점수를 기반으로 예측된 답변이 추가된 상태입니다.
    """

    pred: str  # "1".."len_choices"


class QueryResult(TypedDict):
    """
    단일 query의 검색 결과.

    Retrieval 단계에서 각 query별로 그룹화된 결과를 표현합니다.
    Reranker가 query 정보를 필요로 하므로 query를 함께 저장합니다.

    LangChain Document를 사용하여 service layer (RetrievalResponse)와
    chain layer를 디커플링합니다.
    """

    query: str
    top_k: int
    documents: list[Document]


class OutputState(TypedDict):
    """
    최종 출력 상태.

    Pipeline의 마지막 단계로, 예측 결과와 점수를 포함합니다.
    """

    pred_row: dict[str, str]  # PredRow
    score_row: dict[str, Any]  # ScoreRow
