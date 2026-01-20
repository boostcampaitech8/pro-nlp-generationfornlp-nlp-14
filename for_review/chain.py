"""
Retrieval chain 구성.

Retriever는 검색 계획(RetrievalPlan)을 받아 query별로 그룹화된 문서를 반환합니다.
LangChain Document를 사용하여 service layer와 디커플링합니다.
"""

import asyncio

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import Runnable, RunnableLambda

from chains.utils.utils import normalize_request
from schemas.retrieval import RetrievalPlan


def build_multi_query_retriever(
    retriever: BaseRetriever,
) -> Runnable[RetrievalPlan, list[list[Document]]]:
    """
    Multi-query retrieval chain 생성.

    BaseRetriever를 RetrievalPlan 처리기로 감쌉니다.
    각 query를 병렬로 처리하며, query별로 결과를 그룹화합니다.

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
        - 비동기 버전은 병렬 처리 (asyncio.gather)
        - search_batch, EnsembleRetriever 등은 나중에 추가 예정
    """

    async def retrieve_async(plan: RetrievalPlan) -> list[list[Document]]:
        """비동기 병렬 검색 - 모든 query를 동시에 처리"""

        async def fetch_one(req):
            req = normalize_request(req)
            # BaseRetriever의 ainvoke 메서드 사용
            return await retriever.ainvoke(req.query, top_k=req.top_k)

        # 모든 request를 병렬로 처리
        tasks = [fetch_one(req) for req in plan.requests]
        multi_docs = await asyncio.gather(*tasks)
        return list(multi_docs)

    def retrieve_sync(plan: RetrievalPlan) -> list[list[Document]]:
        """동기 wrapper - 비동기 함수를 동기 컨텍스트에서 실행"""
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        return loop.run_until_complete(retrieve_async(plan))

    return RunnableLambda(retrieve_sync, afunc=retrieve_async, name="multi_query_retriever")
