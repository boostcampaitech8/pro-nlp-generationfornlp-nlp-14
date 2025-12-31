from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from schemas.retrieval.plan import RetrievalRequest
from schemas.retrieval.response import RetrievalResponse


# ---- 서비스 포트(Protocol) ----
@runtime_checkable
class RetrievalServicePort(Protocol):
    """
    서버가 RetrievalResponse를 반환하는 'retrieval' 포트.
    - 명세 협의 전이라 파라미터는 최소만 강제하고 **kwargs로 확장 가능하게 둔다.
    """

    def search(self, req: RetrievalRequest, **kwargs: Any) -> list[RetrievalResponse]: ...


class LangChainRetrievalAdapter(BaseRetriever):
    """
    RetrievalServicePort를 LangChain BaseRetriever(List[Document])로 노출하는 Inbound Adapter.

    - 서버 반환(list[RetrievalResponse])을 Document 리스트로 그대로 변환
    - page_content = context
    - metadata에 question, rank 등 최소 정보 포함
    """

    def __init__(
        self,
        *,
        service,
        top_k: int = 5,
        default_kwargs: dict[str, Any] | None = None,
        source_name: str = "retrieval_service",
    ) -> None:
        """
        초기화 메서드.

            service (RetrievalServicePort): 문서 검색을 수행하는 리트리벌 서비스 구현체.
            top_k (int, optional): 검색 시 반환할 상위 결과 개수. 기본값은 5.
            default_kwargs (dict[str, Any] | None, optional): 서비스 호출에 사용할 기본 키워드 인자 딕셔너리. None인 경우 빈 딕셔너리로 대체된다. 기본값은 None.
            source_name (str, optional): 검색 결과의 출처를 식별하기 위한 이름(태깅/추적용). 기본값은 "retrieval_service".

        Behavior:
            부모 클래스의 초기화(super().__init__())를 호출하고,
            전달받은 인자를 사용해 내부 속성 _service, _top_k, _default_kwargs, _source_name 을 설정한다.

        Returns:
            None
        """
        super().__init__()
        self._service = service
        self._top_k = top_k
        self._default_kwargs = default_kwargs or {}
        self._source_name = source_name

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun | None = None,
        **kwargs: Any,
    ) -> list[Document]:
        k = kwargs.pop("top_k", None) or self._top_k
        req = RetrievalRequest(query=query, top_k=int(k))
        call_kwargs = {**self._default_kwargs, **kwargs}
        responses = self._service.search(req, **call_kwargs) or []

        # 서버가 top_k를 무시할 수도 있으니 방어적으로 컷
        responses = responses[: int(k)]

        docs: list[Document] = []
        for i, r in enumerate(responses):
            metadata = {
                "source": self._source_name,
                "question": r.question or query,
                "rank": i,
                "top_k": int(k),
            }
            if isinstance(r.metadata, dict):
                metadata.update(r.metadata)
            docs.append(
                Document(
                    page_content=r.context or "",
                    metadata=metadata,
                )
            )
        return docs
