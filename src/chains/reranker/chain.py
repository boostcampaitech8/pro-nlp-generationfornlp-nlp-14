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
    ) -> list[Any]:  # 반환 타입을 QueryResult 리스트 유지를 위해 Any로 변경
        if not validate_rerank_input(data):
            return data.get("multi_docs", [])

        original_data = data.get("data", {})
        multi_results = data.get("multi_docs", [])  # QueryResult 객체들의 리스트

        rich_query = format_rich_query(original_data)

        # 각 QueryResult 객체를 순회
        for res_obj in multi_results:
            # QueryResult 내부에 실제 Document 객체 리스트를 가져옴
            docs = getattr(res_obj, "documents", [])
            if not docs:
                continue

            try:
                # BGE-Reranker API 호출
                response = requests.post(
                    f"{base_url.rstrip('/')}/v1/rerank",
                    json={
                        "model": "bge-reranker-v2-m3",
                        "query": rich_query,
                        "documents": [d.page_content for d in docs],  # 여기서 .page_content 접근
                        "top_n": len(docs),
                    },
                    timeout=30,
                )
                response.raise_for_status()
                res_json = response.json()
                api_results = res_json.get("data", res_json.get("results", []))

                # 점수 주입 및 재정렬
                scored_docs = []
                for item in api_results:
                    idx = item["index"]
                    if 0 <= idx < len(docs):
                        doc = docs[idx]
                        doc.metadata["rerank_score"] = item.get("relevance_score", 0.0)
                        scored_docs.append(doc)
                # QueryResult 객체의 documents를 리랭킹된 결과로 교체
                res_obj.documents = scored_docs

            except Exception as e:
                logger.error(f"Reranking for query '{res_obj.query}' failed: {e}")

        # 리랭킹 결과 요약 로그 출력
        log_rerank_results(multi_results)

        return multi_results

    return RunnableLambda(_execute_rerank)
