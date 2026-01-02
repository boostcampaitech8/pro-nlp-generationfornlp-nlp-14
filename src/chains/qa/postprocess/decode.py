import numpy as np
from langchain_core.runnables import chain

from schemas.mcq.context import ForwardContext
from utils.constants import INT_TO_STR_MAP


@chain
def decode_prediction(ctx: ForwardContext) -> dict:
    data = ctx["data"]

    pred_idx = int(np.argmax(ctx["score"]))
    pred = INT_TO_STR_MAP[pred_idx]
    return {"data": data, "score": ctx["score"], "pred": pred}
