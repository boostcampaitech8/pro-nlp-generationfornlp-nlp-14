from inference_utils import load_model

from utils import InferenceConfig

from .nodes import (
    create_llamacpp_forward,
    create_local_forward,
    decode_prediction,
    format_rows,
)

mcq_result = decode_prediction | format_rows


def create_local_mcq_chain(config: InferenceConfig):
    model, tokenizer = load_model(config.checkpoint_path, config.max_seq_length)
    forward = create_local_forward(model, tokenizer)
    return forward | mcq_result


def create_remote_mcq_chain(config: InferenceConfig):
    """remote 서버에 요청을 보내는 체인을 반환합니다.

    Args:
        config (InferenceConfig): Inference Config

    Returns:
        chain: Runnable한 체인을 리턴합니다.
    """
    forward = create_llamacpp_forward()
    return forward | mcq_result
