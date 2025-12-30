# from inference_utils import load_model

from chains.mcq import (
    build_local_forward,
    build_remote_forward,
    decode_prediction,
    format_rows,
)
from utils import InferenceConfig

mcq_result = decode_prediction | format_rows


def build_mcq_chain(config: InferenceConfig):
    if config.use_remote:
        forward = build_remote_forward()
    else:
        forward = build_local_forward(config)

    return forward | mcq_result
