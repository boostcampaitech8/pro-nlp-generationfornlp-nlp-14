from dataclasses import dataclass


@dataclass
class RetrievalResponse:
    question: str
    context: str
