from pydantic import BaseModel


class RetrievalRequest(BaseModel):
    """
    - query topic
    - query chunked
    """

    query: str = ""
    top_k: int = 5
