from schemas.retrieval.response import RetrievalResponse


def build_context(items: list[RetrievalResponse], max_chars: int = 4000) -> str:
    """
    RetrievalResponse 리스트를 prompt에 넣을 context 문자열로 변환.
    구현은 프로젝트 스타일에 맞게 바꿔도 OK.
    """
    blocks: list[str] = []
    total = 0

    for _, it in enumerate(items, start=1):
        text = (it.context or "").strip()

        block = f"#### {it.question}\n - {text}\n"
        if total + len(block) > max_chars:
            break

        blocks.append(block)
        total += len(block)

    return "\n".join(blocks)
