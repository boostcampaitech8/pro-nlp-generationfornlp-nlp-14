"""
Reranker 실행 체인 모듈.

이 모듈은 외부 Reranker API(BGE-Reranker 등)와 통신하여 리트리버가 검색한 문서들의
우선순위를 재조정합니다. 검색 쿼리가 아닌 원본 문제 데이터(질문+선택지)를 기준으로
채점을 수행하여, 실제 정답 단서가 포함된 문서가 상단에 배치되도록 합니다.

주요 로직:
1. Multi-Query로 검색된 문서 그룹(List[List[Document]]) 수신
2. 원본 데이터를 활용한 Rich Query 생성 및 API 호출
3. API 응답 점수를 각 Document의 metadata['rerank_score']에 주입
4. 각 그룹 내 문서들을 점수 순으로 재정렬하여 반환
"""

import asyncio
import logging
from typing import Any

import aiohttp
from langchain_core.documents import Document
from langchain_core.runnables import RunnableConfig, RunnableLambda

from .transformers import (
    format_rich_query,
    log_rerank_results,
    validate_rerank_input,
)

logger = logging.getLogger(__name__)


def build_reranker(base_url: str) -> RunnableLambda:
    """
    리트리버가 가져온 문서들을 원본 문제 데이터와 대조하여
    리랭킹 점수를 부여하는 체인을 생성합니다.

    Args:
        base_url: BGE-Reranker 서버 주소
    """

    async def _execute_rerank_async(
        data: dict[str, Any], config: RunnableConfig = None
    ) -> list[list[Document]]:
        """비동기 리랭킹 - 모든 query group을 병렬 처리"""
        if not validate_rerank_input(data):
            return data.get("multi_docs", [])

        multi_docs: list[list[Document]] = data.get("multi_docs", [])
        rich_query = format_rich_query(data)

        async def rerank_one_group(i: int, docs: list[Document]):
            """단일 그룹의 문서들을 리랭킹"""
            if not docs:
                return i, docs

            try:
                doc_contents = [d.page_content for d in docs]

                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        f"{base_url.rstrip('/')}/v1/rerank",
                        json={
                            "model": "bge-reranker-v2-m3",
                            "query": rich_query,
                            "documents": doc_contents,
                            "top_n": len(docs),
                        },
                        timeout=aiohttp.ClientTimeout(total=30),
                    ) as response:
                        response.raise_for_status()
                        res_json = await response.json()
                        api_results = res_json.get("data", res_json.get("results", []))

                # 점수 주입 및 재정렬
                scored_docs = []
                for item in api_results:
                    idx = item["index"]
                    if 0 <= idx < len(docs):
                        doc = docs[idx]
                        doc.metadata["rerank_score"] = item.get("relevance_score", 0.0)
                        scored_docs.append(doc)

                return i, scored_docs

            except Exception as e:
                logger.error(f"Reranking for query group {i} failed: {e}")
                return i, docs

        # 모든 그룹을 병렬로 리랭킹
        tasks = [rerank_one_group(i, docs) for i, docs in enumerate(multi_docs)]
        results = await asyncio.gather(*tasks)

        # 결과를 원래 순서대로 재배치
        for i, scored_docs in results:
            multi_docs[i] = scored_docs

        log_rerank_results(multi_docs)
        return multi_docs

    def _execute_rerank(
        data: dict[str, Any], config: RunnableConfig = None
    ) -> list[list[Document]]:
        """동기 wrapper - run_in_executor 패턴"""
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        return loop.run_until_complete(_execute_rerank_async(data, config))

    return RunnableLambda(_execute_rerank, afunc=_execute_rerank_async, name="reranker")
