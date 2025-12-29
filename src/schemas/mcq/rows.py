from typing import TypedDict


class PredRow(TypedDict):
    id: str
    answer: str


class ScoreRow(TypedDict):
    id: str
    score: list[float]
