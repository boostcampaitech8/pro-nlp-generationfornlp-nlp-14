from inference.inference_utils import load_model

from .nodes import (
    create_local_forward,
    decode_prediction,
    format_rows,
)

mcq_result = decode_prediction | format_rows


def create_local_mcq_chain(config):
    model, tokenizer = load_model(config.model_path, config.max_seq_length)
    forward = create_local_forward(model, tokenizer)
    return forward | mcq_result
