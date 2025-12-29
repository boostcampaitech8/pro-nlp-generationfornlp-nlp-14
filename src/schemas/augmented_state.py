from typing import TypedDict

from schemas.processed_question import ProcessedQuestion
from schemas.retrieval import RetrievalResponse


class AugmentedState(TypedDict):
    data: ProcessedQuestion
    external_knowledge: list[RetrievalResponse]
    context: str
