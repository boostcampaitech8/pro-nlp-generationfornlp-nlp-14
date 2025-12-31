"""
Planning 단계의 입출력 변환 함수.

기존 chains/planning/mapper.py와 coercion.py를 통합합니다.
"""

import logging

from schemas.retrieval import RetrievalPlan

logger = logging.getLogger(__name__)


def to_prompt_input(data: dict) -> dict:
    """
    QuestionState를 prompt input 형식으로 변환.

    기존 ProcessedQuestion의 property 로직을 여기서 직접 처리합니다:
    - choices_text: choices를 번호 붙여 포맷팅
    - question_plus_block: question_plus가 있으면 <보기> 태그로 감싸기

    Args:
        data: QuestionState (dict)

    Returns:
        Prompt template에 전달할 dict
    """
    choices = data.get("choices", [])
    choices_text = "\n".join(f"{i+1}. {choice}" for i, choice in enumerate(choices))

    question_plus = data.get("question_plus")
    question_plus_block = ""
    if question_plus:
        question_plus_block = f"<보기>\n{question_plus}\n</보기>\n\n"

    return {
        "paragraph": data.get("paragraph", ""),
        "question": data.get("question", ""),
        "choices_text": choices_text,
        "question_plus_block": question_plus_block,
    }


def validate_plan(plan: RetrievalPlan | dict) -> RetrievalPlan:
    """
    LLM 출력을 RetrievalPlan으로 검증 및 변환.

    LLM structured output이 dict로 올 수 있으므로 타입을 보장합니다.
    예상치 못한 타입이 들어오면 빈 plan을 반환하여 pipeline이 계속 진행되도록 합니다.

    Args:
        plan: RetrievalPlan 또는 dict

    Returns:
        RetrievalPlan 인스턴스
    """
    if isinstance(plan, RetrievalPlan):
        return plan
    if isinstance(plan, dict):
        return RetrievalPlan(**plan)

    # 예상치 못한 타입 → 빈 plan 반환 (검색 스킵)
    # Pipeline은 계속 진행되며, 해당 문제는 검색 없이 QA 수행
    logger.warning(
        f"Unexpected plan type: {type(plan)}. Returning empty plan. "
        f"This question will proceed without retrieval."
    )
    return RetrievalPlan(requests=[])
