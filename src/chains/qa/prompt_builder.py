from chains.core.state import QuestionState
from prompts.base import BasePrompt


def build_mcq_request(
    prompt: BasePrompt,
    data: QuestionState,
    *,
    context: str = "",
) -> dict:
    """MCQ request 생성 (TypedDict 기반)

    Args:
        prompt (BasePrompt): 프롬프트 매니저
        data (QuestionState): 질문 상태 (TypedDict)
        context (str, optional): 외부 컨텍스트. Defaults to "".

    Returns:
        dict: QA chain 입력 형태 (id, messages, len_choices)
    """
    # QuestionState에 context 추가
    row = {**data, "context": f"### 힌트\n{context}" if context else ""}

    system_msg = prompt.make_system_prompt(row)
    user_msg = prompt.make_user_prompt(row)
    return {
        "id": data["id"],
        "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        "len_choices": data["len_choices"],
    }
