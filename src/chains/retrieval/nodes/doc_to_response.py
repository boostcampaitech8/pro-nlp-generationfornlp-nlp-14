from langchain_core.documents import Document

from schemas.retrieval.response import RetrievalResponse


def documents_to_retrieval_responses(
    question: str, docs: list[Document]
) -> list[RetrievalResponse]:
    out: list[RetrievalResponse] = []
    for d in docs:
        title = (d.metadata or {}).get("title", "")
        body = d.page_content or ""

        # 컨텍스트 조각 포맷은 여기서 통일
        parts = []
        if title:
            parts.append(f"[TITLE] {title}")
        if body:
            parts.append(body)

        chunk = "\n".join(parts).strip()
        out.append(RetrievalResponse(question=question, context=chunk))

    return out
