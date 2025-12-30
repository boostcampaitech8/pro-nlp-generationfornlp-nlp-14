from typing import TypedDict

from .request import McqRequest


class ForwardContext(TypedDict):
    data: McqRequest
    score: list[float]  # shape: (len_choices,)


class DecodedContext(ForwardContext):
    pred: str  # 1..len_choices
