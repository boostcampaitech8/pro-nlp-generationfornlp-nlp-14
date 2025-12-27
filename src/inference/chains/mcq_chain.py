# from inference_utils import load_model

from utils import InferenceConfig

from .nodes import (
    create_llamacpp_forward,
    create_local_forward,
    decode_prediction,
    format_rows,
)

mcq_result = decode_prediction | format_rows


def create_mcq_chain(config: InferenceConfig):
    if config.use_remote:
        forward = create_llamacpp_forward()
    else:
        forward = create_local_forward(config)

    return forward | mcq_result
