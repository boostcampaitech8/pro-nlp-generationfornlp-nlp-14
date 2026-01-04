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

import logging
from typing import Any

import requests
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

    def _execute_rerank(
        data: dict[str, Any], config: RunnableConfig = None
    ) -> list[list[Document]]:
        # 1. 입력 데이터 검증 (data와 multi_docs가 있는지 확인)
        if not validate_rerank_input(data):
            return data.get("multi_docs", [])

        original_data = data.get("data", {})
        multi_docs = data.get("multi_docs", [])
        # 2. 리랭킹의 기준이 될 'Rich Query' 생성 (질문 + 선택지 결합)
        rich_query = format_rich_query(original_data)
        reranked_results = []

        # 3. 각 쿼리 그룹별로 리랭킹 수행
        for i, docs in enumerate(multi_docs):
            if not docs:
                reranked_results.append([])
                continue

            try:
                # BGE-Reranker API 호출
                # 검색어가 아닌 원본 문제를 'query'로 사용함
                response = requests.post(
                    f"{base_url.rstrip('/')}/v1/rerank",
                    json={
                        "model": "bge-reranker-v2-m3",
                        "query": rich_query,
                        "documents": [d.page_content for d in docs],
                        "top_n": len(docs),
                    },
                    timeout=30,
                )
                response.raise_for_status()
                res_json = response.json()
                # API 응답 규격 처리 (data 또는 results 키 대응)
                api_results = res_json.get("data", res_json.get("results", []))
                # 4. 결과 매핑 및 점수 주입
                scored_group = []
                for item in api_results:
                    idx = item["index"]
                    if 0 <= idx < len(docs):
                        doc = docs[idx]
                        # 메타데이터에 리랭킹 점수 기록 (나중에 merge_strategy에서 사용)
                        doc.metadata["rerank_score"] = item.get("relevance_score", 0.0)
                        scored_group.append(doc)
                reranked_results.append(scored_group)

            except Exception as e:
                logger.error(f"Reranking group {i} failed: {e}. Keeping original order.")
                reranked_results.append(docs)

        # 5. 리랭킹 결과 요약 로그 출력
        log_rerank_results(reranked_results)

        return reranked_results

    return RunnableLambda(_execute_rerank)
