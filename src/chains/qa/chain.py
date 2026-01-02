"""
QA chain 구성.

MCQ 문제에 대한 답변 생성 chain을 구성합니다.
"""

from langchain_core.runnables import Runnable

from chains.qa.inference import build_local_forward, build_remote_forward
from chains.qa.postprocess import decode_prediction, format_rows
from chains.qa.prompt_builder import build_mcq_request
from utils import InferenceConfig


def build_qa_chain(
    config: InferenceConfig,
    prompt_manager,
) -> Runnable[dict, tuple]:
    """
    QA chain 생성.

    Pipeline: {"data": QuestionState, "context": str}
             → prompt building → inference → decode → format
             → (PredRow, ScoreRow)

    Args:
        config: 추론 설정 (use_remote, model path 등)
        prompt_manager: Prompt 생성 매니저

    Returns:
        Runnable[dict, tuple[PredRow, ScoreRow]]

    Example:
        >>> from prompts import get_prompt_manager
        >>> from utils import InferenceConfig
        >>>
        >>> config = InferenceConfig(use_remote=True, ...)
        >>> prompt_manager = get_prompt_manager("v7")
        >>> qa_chain = build_qa_chain(config, prompt_manager)
        >>>
        >>> state = {"data": question_state, "context": "..."}
        >>> pred_row, score_row = qa_chain.invoke(state)

    Note:
        - Prompt manager는 외부 주입 (prompts/ 모듈)
        - Local vs Remote inference는 config.use_remote로 결정
    """
    # Prompt building
    from langchain_core.runnables import RunnableLambda

    prompt_builder = RunnableLambda(
        lambda state: build_mcq_request(
            prompt_manager, state["data"], context=state.get("context", "")
        )
    )

    # Inference (local or remote)
    if config.use_remote:
        forward = build_remote_forward()
    else:
        forward = build_local_forward(config)

    # Postprocess
    postprocess = decode_prediction | format_rows

    return prompt_builder | forward | postprocess
