from typing import TypedDict

import torch

from .request import McqRequest


class ForwardContext(TypedDict):
    data: McqRequest
    score: torch.Tensor  # shape: (len_choices,)


class DecodedContext(ForwardContext):
    pred: str  # 1..len_choices
