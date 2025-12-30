from schemas.processed_question import ProcessedQuestion


def _to_prompt_input(data: ProcessedQuestion) -> dict:
    # plan_prompt가 dict를 기대한다면 이렇게 변환
    # (ChatPromptTemplate.from_messages로 만든 템플릿은 보통 dict input)
    return {
        "id": data.id,
        "paragraph": data.paragraph,
        "question": data.question,
        "choices_text": data.choices_text,
        "question_plus_block": data.question_plus_block,
        # 필요하면 choices_text 같은 것도 여기서 계산해서 넣어도 됨
    }
