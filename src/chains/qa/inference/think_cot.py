"""
Think COT-based MCQ chain (llama.cpp + qwen3-32B).

Uses think-mode drafts and extracts an `Answer: N` line from the output.
"""

import re

from langchain_core.runnables import Runnable, RunnableLambda, chain
from langchain_openai import ChatOpenAI

from chains.qa.postprocess import format_rows
from chains.qa.prompt_builder import build_mcq_request
from schemas.mcq.request import McqRequest
from utils import InferenceConfig

THINK_DRAFT_COUNT = 1
THINK_COT_LOG_PATH = "outputs/think_cot.jsonl"
THINK_MODEL_NAME = "qwen3-32B"
_THINK_TAG_RE = re.compile(r"</?think>", re.IGNORECASE)
_ANSWER_RE = re.compile(r"(?:Answer|정답)\s*:\s*(\d+)", re.IGNORECASE)


def _sanitize_think_output(text: str) -> str:
    cleaned = _THINK_TAG_RE.sub("", text)
    return cleaned.strip()


def _build_answer_instruction(len_choices: int) -> str:
    return (
        "think 모드로 사고 과정을 작성한 뒤 마지막 줄에 "
        f"Answer: 1..{len_choices} 형식으로만 답을 출력하세요. "
        "예: Answer: 3"
    )


def _parse_answer(text: str, len_choices: int) -> str:
    matches = _ANSWER_RE.findall(text)
    if not matches:
        return "0"
    value = int(matches[-1])
    if value < 1 or value > len_choices:
        return "0"
    return str(value)


def _build_think_cot_forward() -> Runnable[McqRequest, dict]:
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
        "enable_thinking": True,
    }

    think_model = ChatOpenAI(
        base_url=base_url,
        api_key="NOT_NEED",
        model_name=THINK_MODEL_NAME,
        name="THINK_COT_DRAFT",
        temperature=0.7,
        top_p=0.95,
        max_retries=2,
        timeout=900,  # 10분 timeout
        extra_body=draft_extra_body,
    )

    # init logger
    with open(THINK_COT_LOG_PATH, "w", encoding="utf-8") as f:
        f.write("")

    @chain
    def forward(data: McqRequest) -> dict:
        import json
        import logging

        logger = logging.getLogger(__name__)

        len_choices = data["len_choices"]
        messages = list(data["messages"])

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

        pred_str = "0"
        for draft in reversed(drafts):
            candidate = _parse_answer(draft, len_choices)
            if candidate != "0":
                pred_str = candidate
                break

        # JSONL 형식으로 저장
        with open(THINK_COT_LOG_PATH, "a", encoding="utf-8") as f:
            json.dump(
                {
                    "id": data["id"],
                    "answer": pred_str,
                    "drafts": drafts,
                },
                f,
                ensure_ascii=False,
            )
            f.write("\n")

        scores = [0.0] * len_choices
        if pred_str == "0":
            logger.warning(
                f"Missing or invalid answer for question {data['id']} "
                f"(valid range: 1-{len_choices}). Setting all scores to 0."
            )
        else:
            scores[int(pred_str) - 1] = 1.0

        return {"data": data, "score": scores, "pred": pred_str}

    return forward


def build_think_cot_chain(
    config: InferenceConfig,
    prompt_manager,
) -> Runnable[dict, tuple]:
    """
    Think COT-based MCQ chain.

    Pipeline: {"data": QuestionState, "context": str}
             -> prompt building -> think drafts -> parse Answer line -> format
             -> (PredRow, ScoreRow)
    """
    if not config.use_remote:
        raise ValueError("Think COT chain requires remote LLM.")

    prompt_builder = RunnableLambda(
        lambda state: build_mcq_request(
            prompt_manager, state["data"], context=state.get("context", "")
        )
    )

    forward = _build_think_cot_forward()
    return prompt_builder | forward | format_rows
