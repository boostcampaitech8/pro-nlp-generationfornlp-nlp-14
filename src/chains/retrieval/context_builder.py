"""
Retrieval context building utilities.

LangChain Document를 사용하여 context를 구성합니다.
Service layer의 RetrievalResponse와 디커플링됩니다.
"""

from langchain_core.documents import Document


def build_context(docs: list[Document], max_chars: int = 4000) -> str:
    """
    LangChain Documents를 prompt에 넣을 context 문자열로 변환.

    Args:
        docs: LangChain Documents
        max_chars: 최대 문자 수

    Returns:
        포맷팅된 context 문자열

    Note:
        기존 nodes/build_context.py를 Document 기반으로 수정했습니다.
        query 정보는 Document metadata에서 가져옵니다.

        TODO: 문서 merge 전략 리팩토링 필요
        - Round-robin vs sequential
        - Query별 구분 포맷 개선
    """
    blocks: list[str] = []
    total = 0

    for doc in docs:
        # Document에서 정보 추출
        text = (doc.page_content or "").strip()
        metadata = doc.metadata or {}

        # Query는 metadata에서 가져오기 (adapter가 넣어줌)
        query = metadata.get("question", "")

        # 포맷: #### query\n - text
        if query:
            block = f"#### {query}\n - {text}\n"
        else:
            block = f" - {text}\n"

        if total + len(block) > max_chars:
            break

        blocks.append(block)
        total += len(block)

    return "\n".join(blocks)
