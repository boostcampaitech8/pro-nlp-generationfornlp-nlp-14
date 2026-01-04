"""
Retrieval chain 구성.

Retriever는 검색 계획(RetrievalPlan)을 받아 query별로 그룹화된 문서를 반환합니다.
LangChain Document를 사용하여 service layer와 디커플링합니다.
"""

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import Runnable, chain

from chains.utils.utils import normalize_request
from schemas.retrieval import RetrievalPlan


def build_multi_query_retriever(
    retriever: BaseRetriever,
) -> Runnable[RetrievalPlan, list[list[Document]]]:
    """
    Multi-query retrieval chain 생성.

    BaseRetriever를 RetrievalPlan 처리기로 감쌉니다.
    각 query를 순차적으로 처리하며, query별로 결과를 그룹화합니다.

    Args:
        retriever: LangChain BaseRetriever (EnsembleRetriever 포함 가능)

    Returns:
        Runnable[RetrievalPlan, list[list[Document]]]

    Example:
        >>> from chains.retrieval.adapter import LangChainRetrievalAdapter
        >>> from chains.retrieval.services import TavilyWebSearchService
        >>> from tavily import TavilyClient
        >>>
        >>> service = TavilyWebSearchService(TavilyClient(), options={})
        >>> adapter = LangChainRetrievalAdapter(service=service, source_name="web")
        >>> retriever_chain = build_multi_query_retriever(adapter)
        >>>
        >>> plan = RetrievalPlan(requests=[...])
        >>> results = retriever_chain.invoke(plan)  # list[list[Document]]

    Note:
        - LangChain Document만 사용 (RetrievalResponse 의존성 제거)
        - Service layer와 chain layer 디커플링
        - 현재는 순차 호출 (for loop)
        - search_batch, EnsembleRetriever 등은 나중에 추가 예정
    """

    @chain
    def retrieve(plan: RetrievalPlan) -> list[list[Document]]:
        """
        RetrievalPlan의 각 request를 처리하여 query별 문서 리스트 반환.
        """
        multi_docs: list[list[Document]] = []

        for req in plan.requests:
            req = normalize_request(req)
            docs = retriever.invoke(req.query, top_k=req.top_k)
            multi_docs.append(docs)

        return multi_docs

    return retrieve
