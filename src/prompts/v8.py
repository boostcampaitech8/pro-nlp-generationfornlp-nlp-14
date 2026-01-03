from .base import BasePrompt


class V8Prompt(BasePrompt):
    """V8: 간결한 형식의 프롬프트"""

    PROMPT_NO_QUESTION_PLUS = """다음 지문과 문제를 읽고 정답을 고르세요.

[지문]
{paragraph}

[문제]
{question}

[선지]
{choices}

정답 번호만 출력하세요."""

    PROMPT_QUESTION_PLUS = """다음 지문과 문제를 읽고 정답을 고르세요.

[보기]
{question_plus}

[지문]
{paragraph}

[문제]
{question}

[선지]
{choices}

정답 번호만 출력하세요."""

    @property
    def system_prompt(self) -> str:
        return ""  # make_system_prompt에서 동적으로 생성

    def make_system_prompt(self, row: dict) -> str:
        """선택지 개수에 따라 동적으로 system prompt 생성"""
        num_choices = len(row.get("choices", []))
        return (
            f"너는 한국어 수능형 {num_choices}지선다 문제를 푸는 모델이다.\n"
            "지문과 선지를 근거로 대조하여 정답을 고른다.\n"
            "최종 출력은 반드시 정답 번호만 출력한다.\n"
            f"설명, 해설, 문장, 기호, 공백을 출력하지 마라. 오직 1~{num_choices} 중 하나만 출력하라."
        )

    def make_user_prompt(self, row: dict) -> str:
        choices = row["choices"]
        choices_string = "\n".join([f"{idx + 1}. {choice}" for idx, choice in enumerate(choices)])

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
