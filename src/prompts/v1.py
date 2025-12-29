from .base import BasePrompt


class V1Prompt(BasePrompt):
    """V1: 기본 프롬프트 (Hardcoded 1-5, Simple System Prompt)"""

    PROMPT_NO_QUESTION_PLUS = """지문:
{paragraph}

질문:
{question}

선택지:
{choices}

1, 2, 3, 4, 5 중에 하나를 정답으로 고르세요.
정답:"""

    PROMPT_QUESTION_PLUS = """지문:
{paragraph}

질문:
{question}

<보기>:
{question_plus}

선택지:
{choices}

1, 2, 3, 4, 5 중에 하나를 정답으로 고르세요.
정답:"""

    @property
    def system_prompt(self) -> str:
        return "지문을 읽고 질문의 답을 구하세요."

    def make_user_prompt(self, row: dict) -> str:
        choices = row["choices"]
        choices_string = "\n".join([f"{idx + 1} - {choice}" for idx, choice in enumerate(choices)])

        if row.get("question_plus"):
            return self.PROMPT_QUESTION_PLUS.format(
                paragraph=row["paragraph"],
                question=row["question"],
                question_plus=row["question_plus"],
                choices=choices_string,
            )
        else:
            return self.PROMPT_NO_QUESTION_PLUS.format(
                paragraph=row["paragraph"],
                question=row["question"],
                choices=choices_string,
            )
