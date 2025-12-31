from langchain_core.documents import Document

from schemas.retrieval.response import RetrievalResponse


def documents_to_retrieval_responses(
    question: str, docs: list[Document]
) -> list[RetrievalResponse]:
    out: list[RetrievalResponse] = []
    for d in docs:
        title = (d.metadata or {}).get("title", "")
        body = (d.page_content or "").strip()
        metadata = dict(d.metadata or {})

        # 컨텍스트 조각 포맷은 여기서 통일
        parts = []
        if title:
            title_line = f"[TITLE] {title}"

            # 본문 첫 줄이 제목과 동일하면 중복 제거
            if body:
                first_line, *rest = body.splitlines()
                first_line_stripped = first_line.strip()
                if first_line_stripped in {title, title_line}:
                    body = "\n".join(rest).strip()

            parts.append(title_line)
        if body:
            parts.append(body)

        chunk = "\n".join(parts).strip()
        out.append(
            RetrievalResponse(
                question=question,
                context=chunk,
                metadata=metadata,
            )
        )

    return out
