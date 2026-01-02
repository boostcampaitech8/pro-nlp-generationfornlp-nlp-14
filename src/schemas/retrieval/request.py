from dataclasses import dataclass


@dataclass
class RetrievalRequest:
    """
    - query topic
    - query chunked
    """

    query: str = ""
    top_k: int = 5
