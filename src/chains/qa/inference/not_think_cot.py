"""
Non-think COT-based MCQ chain.

Uses chain-of-thought reasoning with structured output (without native think mode)
to select an answer, and returns placeholder scores.
"""

from langchain_core.runnables import Runnable, RunnableLambda, chain
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from chains.qa.postprocess import format_rows
from chains.qa.prompt_builder import build_mcq_request
from schemas.mcq.request import McqRequest
from utils import InferenceConfig


class NonThinkCotAnswer(BaseModel):
    reasoning_steps: list[str] = Field(
        description="당신의 사고 과정을 자유롭게 작성하는 공간입니다. "
        "문제를 분석하고, 각 선택지를 검토하며, 논리적으로 추론하세요."
    )
    conclusion: str = Field(description="최종 결론에 도달한 이유를 간단히 요약하세요.")
    answer: int = Field(description="최종 주어진 선택지 번호 (1부터 시작하는 정수 형태).")


def _build_non_think_cot_forward() -> Runnable[McqRequest, dict]:
    import os

    import dotenv

    dotenv.load_dotenv()
    base_url = os.getenv("LLAMA_CPP_SERVER_URL")

    model = ChatOpenAI(
        base_url=base_url,
        api_key="NOT_NEED",
        model_name="LLama_cpp_model",
        name="CHOOSE_NUMBER_NON_THINK_COT",
        temperature=0.7,
        top_p=0.95,
        max_retries=2,
        timeout=600,  # 10분 timeout
        extra_body={
            "min_p": 0,
            "top_k": 20,
            "presence_penalty": 1.5,
        },
    )
    structured = model.with_structured_output(NonThinkCotAnswer)  # method 제거

    @chain
    def forward(data: McqRequest) -> dict:
        import json
        import logging

        logger = logging.getLogger(__name__)

        output: NonThinkCotAnswer = structured.invoke(data["messages"])
        # JSONL 형식으로 저장
        with open("outputs/non_think_cot.jsonl", "a", encoding="utf-8") as f:
            json.dump(
                {
                    "id": data["id"],
                    "answer": output.answer,
                    "reasoning_steps": output.reasoning_steps,
                    "conclusion": output.conclusion,
                },
                f,
                ensure_ascii=False,
            )
            f.write("\n")

        # Pydantic이 타입 보장
        pred = output.answer
        len_choices = data["len_choices"]

        # 답변 범위 검증 (1 ~ len_choices)
        scores = [0.0] * len_choices

        if pred < 1 or pred > len_choices:
            logger.warning(
                f"Invalid answer {pred} for question {data['id']} "
                f"(valid range: 1-{len_choices}). Setting all scores to 0."
            )
            pred = 0
            # scores는 이미 모두 0.0으로 초기화됨
        else:
            scores[pred - 1] = 1.0

        return {"data": data, "score": scores, "pred": pred}

    return forward


def build_non_think_cot_chain(
    config: InferenceConfig,
    prompt_manager,
) -> Runnable[dict, tuple]:
    """
    Non-think COT-based MCQ chain.

    Pipeline: {"data": QuestionState, "context": str}
             -> prompt building -> inference -> format
             -> (PredRow, ScoreRow)
    """
    if not config.use_remote:
        raise ValueError("Non-think COT chain requires remote LLM for structured output.")

    prompt_builder = RunnableLambda(
        lambda state: build_mcq_request(
            prompt_manager, state["data"], context=state.get("context", "")
        )
    )

    forward = _build_non_think_cot_forward()
    return prompt_builder | forward | format_rows
