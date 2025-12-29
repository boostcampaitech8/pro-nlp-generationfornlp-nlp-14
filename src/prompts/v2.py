from .base import BasePrompt


class V2Prompt(BasePrompt):
    """V2: 동적 선택지 + 단순 시스템 프롬프트"""

    PROMPT_NO_QUESTION_PLUS = """### 지문:
{paragraph}

### 질문:
{question}

### 선택지:
{choices}

---
위 지문의 내용을 근거로 할 때, 질문에 대한 가장 적절한 답변을 선택지 {choice_range} 에서 하나만 고르세요.

정답:"""

    PROMPT_QUESTION_PLUS = """### 지문:
{paragraph}

### 질문:
{question}

### <보기>:
{question_plus}

### 선택지:
{choices}

---
지문과 <보기>의 내용을 근거로 할 때, 질문에 대한 가장 적절한 답변을 선택지 {choice_range} 에서 하나만 고르세요.

정답:"""

    @property
    def system_prompt(self) -> str:
        return "지문을 읽고 질문의 답을 구하세요."

    def make_user_prompt(self, row: dict) -> str:
        choices = row["choices"]
        choices_string = "\n".join([f"{idx + 1} - {choice}" for idx, choice in enumerate(choices)])
        choice_range = ", ".join([str(n) for n in range(1, len(choices) + 1)])

        if row.get("question_plus"):
            return self.PROMPT_QUESTION_PLUS.format(
                paragraph=row["paragraph"],
                question=row["question"],
                question_plus=row["question_plus"],
                choices=choices_string,
                choice_range=choice_range,
            )
        else:
            return self.PROMPT_NO_QUESTION_PLUS.format(
                paragraph=row["paragraph"],
                question=row["question"],
                choices=choices_string,
                choice_range=choice_range,
            )
