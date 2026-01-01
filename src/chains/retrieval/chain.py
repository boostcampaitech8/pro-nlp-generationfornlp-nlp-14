"""
Retrieval chain 구성.

Retriever는 검색 계획(RetrievalPlan)을 받아 query별로 그룹화된 결과를 반환합니다.
LangChain Document를 사용하여 service layer와 디커플링합니다.
"""

from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import chain

from chains.core.state import QueryResult
from chains.core.utils import normalize_request
from schemas.retrieval import RetrievalPlan


def build_multi_query_retriever(
    retriever: BaseRetriever,
) -> chain:
    """
    Multi-query retrieval chain 생성.

    BaseRetriever를 RetrievalPlan 처리기로 감쌉니다.
    각 query를 순차적으로 처리하며, query별로 결과를 그룹화합니다.

    Args:
        retriever: LangChain BaseRetriever (EnsembleRetriever 포함 가능)

    Returns:
        Runnable[RetrievalPlan, list[QueryResult]]

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
        >>> results = retriever_chain.invoke(plan)  # list[QueryResult]

    Note:
        - LangChain Document만 사용 (RetrievalResponse 의존성 제거)
        - Service layer와 chain layer 디커플링
        - 현재는 순차 호출 (for loop)
        - search_batch, EnsembleRetriever 등은 나중에 추가 예정
    """

    @chain
    def retrieve(plan: RetrievalPlan) -> list[QueryResult]:
        """
        RetrievalPlan의 각 request를 처리하여 query별 결과 반환.
        """
        query_results = []

        for req in plan.requests:
            # Request 정규화
            req = normalize_request(req)

            # BaseRetriever 호출 (단일 query)
            # Adapter가 이미 Document로 변환해줌
            docs = retriever.invoke(req.query, top_k=req.top_k)

            # QueryResult로 그룹화 (Document 그대로 사용)
            query_results.append(
                QueryResult(
                    query=req.query,
                    top_k=req.top_k,
                    documents=docs,
                )
            )

        return query_results

    return retrieve
