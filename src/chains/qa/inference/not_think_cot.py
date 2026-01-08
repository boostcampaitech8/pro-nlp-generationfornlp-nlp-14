"""
Think COT-based MCQ chain.

Uses think-mode drafts, merges them, then produces structured output to select an answer.
"""

from typing import Annotated
import re

from typing_extensions import Literal
from langchain_core.runnables import Runnable, RunnableLambda, chain
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from chains.qa.postprocess import format_rows
from chains.qa.prompt_builder import build_mcq_request
from schemas.mcq.request import McqRequest
from utils import InferenceConfig

THINK_DRAFT_COUNT = 3
THINK_COT_LOG_PATH = "outputs/think_cot.jsonl"
_THINK_TAG_RE = re.compile(r"</?think>", re.IGNORECASE)


class ThinkCotAnswer(BaseModel):
    reasoning_steps: Annotated[
        list[Annotated[str, Field(min_length=10)]],
        Field(
            min_length=2,
            description="사고 과정을 자유롭게 작성하는 공간입니다. "
            "문제를 분석하고, 각 선택지를 검토하며, 논리적으로 추론하세요.",
        ),
    ]
    answer: Literal[1, 2, 3, 4, 5] = Field(description="주어진 선택지 내에서 최선의 ")


def _sanitize_think_output(text: str) -> str:
    cleaned = _THINK_TAG_RE.sub("", text)
    return cleaned.strip()


def _format_think_drafts(drafts: list[str]) -> str:
    blocks = []
    for idx, draft in enumerate(drafts, 1):
        cleaned = _sanitize_think_output(draft)
        if not cleaned:
            continue
        blocks.append(f"[DRAFT {idx}]\n{cleaned}")
    return "\n\n".join(blocks)


def _build_merge_instruction(draft_count: int, len_choices: int) -> str:
    return (
        f"아래는 think 모드에서 생성된 초안 {draft_count}개입니다. "
        "중복을 제거하고 근거 중심으로 reasoning_steps를 2개 이상 작성해 "
        f"최종 답을 선택지 1~{len_choices} 중에서 고르세요. "
        "초안 텍스트나 <think> 태그를 그대로 복사하지 말고 "
        "반드시 JSON만 출력하세요."
    )


def _build_think_cot_forward(config: InferenceConfig) -> Runnable[McqRequest, dict]:
    import os

    import dotenv

    dotenv.load_dotenv()
    base_url = os.getenv("LLAMA_CPP_SERVER_URL")
    if not base_url:
        raise ValueError("LLAMA_CPP_SERVER_URL is required for think COT chain.")

    draft_extra_body = {
        "min_p": 0,
        "top_k": 20,
        "presence_penalty": 1.5,
    }
    if config.enable_thinking:
        draft_extra_body["enable_thinking"] = True

    final_extra_body = {
        "min_p": 0,
        "top_k": 20,
    }

    think_model = ChatOpenAI(
        base_url=base_url,
        api_key="NOT_NEED",
        model_name="LLama_cpp_model",
        name="THINK_COT_DRAFT",
        temperature=0.7,
        top_p=0.95,
        max_retries=2,
        timeout=600,  # 10분 timeout
        extra_body=draft_extra_body,
    )
    final_model = ChatOpenAI(
        base_url=base_url,
        api_key="NOT_NEED",
        model_name="LLama_cpp_model",
        name="THINK_COT_FINAL",
        temperature=0.2,
        top_p=0.9,
        max_retries=2,
        timeout=600,  # 10분 timeout
        extra_body=final_extra_body,
    )
    structured = final_model.with_structured_output(ThinkCotAnswer)

    # init logger
    with open(THINK_COT_LOG_PATH, "w", encoding="utf-8") as f:
        f.write("")

    @chain
    def forward(data: McqRequest) -> dict:
        import json
        import logging

        logger = logging.getLogger(__name__)

        len_choices = data["len_choices"]
        messages = data["messages"]

        drafts: list[str] = []
        for idx in range(THINK_DRAFT_COUNT):
            try:
                result = think_model.invoke(messages)
            except Exception as exc:
                logger.warning(
                    "Think draft failed (sample %d/%d): %s",
                    idx + 1,
                    THINK_DRAFT_COUNT,
                    exc,
                )
                continue
            content = getattr(result, "content", "") or ""
            cleaned = _sanitize_think_output(content)
            if cleaned:
                drafts.append(cleaned)

        if not drafts:
            raise RuntimeError("Think model produced no usable drafts.")

        combined_drafts = _format_think_drafts(drafts)

        final_messages = list(messages)
        final_messages.append({"role": "assistant", "content": combined_drafts})
        final_messages.append(
            {
                "role": "user",
                "content": _build_merge_instruction(len(drafts), len_choices),
            }
        )

        def invoke_with_retry(messages: list[dict]) -> ThinkCotAnswer:
            max_attempts = 3
            retry_messages = list(messages)
            last_err = None
            for attempt in range(max_attempts):
                try:
                    return structured.invoke(retry_messages)
                except Exception as exc:
                    last_err = exc
                    logger.warning(
                        "Structured output validation failed (attempt %d/%d): %s",
                        attempt + 1,
                        max_attempts,
                        exc,
                    )
                    if attempt + 1 >= max_attempts:
                        raise
                    retry_messages = retry_messages + [
                        {
                            "role": "system",
                            "content": (
                                "이전 출력이 스키마를 위반했습니다. 반드시 JSON만 출력하세요. "
                                "reasoning_steps는 2개 이상, 각 항목은 10자 이상이며 지문/선지 "
                                "근거를 포함해야 합니다. "
                                f"answer는 1~{len_choices} 중 하나입니다."
                            ),
                        }
                    ]
            raise last_err

        output: ThinkCotAnswer = invoke_with_retry(final_messages)
        # JSONL 형식으로 저장
        with open(THINK_COT_LOG_PATH, "a", encoding="utf-8") as f:
            json.dump(
                {
                    "id": data["id"],
                    "answer": output.answer,
                    "reasoning_steps": output.reasoning_steps,
                },
                f,
                ensure_ascii=False,
            )
            f.write("\n")

        # Pydantic이 타입 보장
        pred = int(output.answer)
        # 답변 범위 검증 (1 ~ len_choices)
        scores = [0.0] * len_choices

        if pred < 1 or pred > len_choices:
            logger.warning(
                f"Invalid answer {pred} for question {data['id']} "
                f"(valid range: 1-{len_choices}). Setting all scores to 0."
            )
            pred_str = "0"
        else:
            scores[pred - 1] = 1.0
            pred_str = str(pred)

        return {"data": data, "score": scores, "pred": pred_str}

    return forward


def build_non_think_cot_chain(
    config: InferenceConfig,
    prompt_manager,
) -> Runnable[dict, tuple]:
    """
    Think COT-based MCQ chain.

    Pipeline: {"data": QuestionState, "context": str}
             -> prompt building -> think drafts -> structured output -> format
             -> (PredRow, ScoreRow)
    """
    if not config.use_remote:
        raise ValueError("Think COT chain requires remote LLM for structured output.")

    prompt_builder = RunnableLambda(
        lambda state: build_mcq_request(
            prompt_manager, state["data"], context=state.get("context", "")
        )
    )

    forward = _build_think_cot_forward(config)
    return prompt_builder | forward | format_rows


def build_non_think_cot_chain(
    config: InferenceConfig,
    prompt_manager,
) -> Runnable[dict, tuple]:
    return build_think_cot_chain(config=config, prompt_manager=prompt_manager)
