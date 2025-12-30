from prompts.base import BasePrompt
from schemas.processed_question import ProcessedQuestion


def build_mcq_request(
    prompt: BasePrompt,
    data: ProcessedQuestion,
    *,
    context: str = "",
) -> dict:
    """v7부터 사용가능

    Args:
        prompt (BasePrompt): _description_
        data (ProcessedQuestion): _description_
        context (str, optional): _description_. Defaults to "".

    Returns:
        dict: _description_
    """
    row = {
        "id": data.id,
        "paragraph": data.paragraph,
        "question": data.question,
        "choices": data.choices,
        "question_plus": data.question_plus,
        "context": context,
    }

    system_msg = prompt.make_system_prompt(row)
    user_msg = prompt.make_user_prompt(row)
    return {
        "id": data.id,
        "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        "len_choices": len(data.choices),
    }
