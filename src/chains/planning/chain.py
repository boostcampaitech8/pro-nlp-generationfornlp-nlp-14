"""
Planning chain 구성.

Planner는 질문을 분석하여 필요한 검색 쿼리 계획(RetrievalPlan)을 생성합니다.
"""

from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableLambda

from chains.planning.transforms import to_prompt_input, validate_plan
from schemas.retrieval import RetrievalPlan


def build_planner(
    llm: BaseChatModel,
    prompt: ChatPromptTemplate,
) -> Runnable[dict, RetrievalPlan]:
    """
    Planning chain 생성.

    Pipeline: QuestionState → prompt input 변환 → LLM → 검증

    Args:
        llm: Planning에 사용할 LLM
        prompt: Planning prompt template (외부 주입)

    Returns:
        Runnable[QuestionState, RetrievalPlan]

    Example:
        >>> from langchain_openai import ChatOpenAI
        >>> from prompts.plan.plan import plan_prompt
        >>>
        >>> llm = ChatOpenAI(model="gpt-4")
        >>> planner = build_planner(llm, plan_prompt)
        >>>
        >>> question_state = {"id": "1", "question": "...", ...}
        >>> plan = planner.invoke(question_state)
    """
    return (
        RunnableLambda(to_prompt_input)
        | prompt
        | llm.with_structured_output(RetrievalPlan)
        | RunnableLambda(validate_plan)
    )
