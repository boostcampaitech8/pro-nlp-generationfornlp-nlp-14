from .base import BasePrompt


class V7Prompt(BasePrompt):
    """V7: V6에 RAG용 {context}추가"""

    PROMPT_NO_QUESTION_PLUS = """아래 지문의 내용을 근거로 할 때, 질문에 대한 가장 적절한 답변을 선택지 {choice_range} 에서 하나만 고르세요.

### 지문:
{paragraph}

### 질문:
{question}

### 선택지:
{choices}

{context}

정답:"""

    PROMPT_QUESTION_PLUS = """아래 지문과 <보기>의 내용을 근거로 할 때, 질문에 대한 가장 적절한 답변을 선택지 {choice_range} 에서 하나만 고르세요.

### 지문:
{paragraph}

### 질문:
{question}

### <보기>:
{question_plus}

### 선택지:
{choices}

{context}

정답:"""

    @property
    def system_prompt(self) -> str:
        return (
            "당신은 대한민국 대학수학능력시험(CSAT) 평가 전문가입니다. "
            "모든 지문을 비판적이고 논리적으로 분석하며, 매력적인 오답에 속지 않고 "
            "지문의 근거만을 바탕으로 가장 적절한 답을 도출합니다. "
            "추론 과정 없이 즉시 정답 번호를 제시해야 하므로, 지문의 핵심 맥락에 집중하세요."
        )

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
                context=row.get("context", ""),
            )
        else:
            return self.PROMPT_NO_QUESTION_PLUS.format(
                paragraph=row["paragraph"],
                question=row["question"],
                choices=choices_string,
                choice_range=choice_range,
                context=row.get("context", ""),
            )
