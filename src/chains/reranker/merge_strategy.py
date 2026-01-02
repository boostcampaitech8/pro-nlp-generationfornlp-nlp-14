"""
Merge Strategy 로직.

리랭커를 거친 그룹별 문서들을 전략적으로 병합하여
최종 QA 체인에 전달할 RetrievalResponse 객체를 생성합니다.
"""

import logging

from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda

from chains.qa.postprocess import build_context
from schemas.reranker import RetrievalResponse

logger = logging.getLogger(__name__)


def merge_strategies(strategy_type: str, top_n: int, max_chars: int):
    """
    평탄화 전략을 수행하는 함수를 반환합니다.

    Args:
        strategy_type: "query_first" (쿼리당 1개 보장) 등
        top_n: 최종 선택할 총 문서 개수
        max_chars: 최종 컨텍스트 문자열의 최대 길이
    """

    def _merge(reranked_multi_docs: list[list[Document]]) -> RetrievalResponse:
        """
        Input: 리랭킹 점수가 주입된 List[List[Document]]
        Output: RetrievalResponse (최종 context 문자열 포함)
        """
        if not reranked_multi_docs:
            return RetrievalResponse(question="", context="")

        final_docs = []
        seen_contents = set()

        # [전략: query_first] 쿼리당 최소 하나 보장 (Diversity First)
        if strategy_type == "query_first":
            # 1. 각 그룹(각 쿼리 결과)의 1등 문서들을 최우선적으로 수집
            for group in reranked_multi_docs:
                if group:
                    # 리랭커 결과는 이미 점수순 정렬되어 있으므로 첫 번째 문서가 1등
                    best_doc = group[0]
                    content_hash = hash(best_doc.page_content)
                    if content_hash not in seen_contents:
                        final_docs.append(best_doc)
                        seen_contents.add(content_hash)
            # 2. 남은 자리가 있다면, 모든 그룹의 나머지 문서들을 통합하여 점수순으로 채움
            if len(final_docs) < top_n:
                all_remaining = []
                for group in reranked_multi_docs:
                    # 이미 뽑힌 1등을 제외한 나머지 추가
                    if len(group) > 1:
                        all_remaining.extend(group[1:])
                # 전체 문서 기준 리랭킹 점수(rerank_score) 내림차순 정렬
                all_remaining.sort(key=lambda x: x.metadata.get("rerank_score", 0.0), reverse=True)
                for doc in all_remaining:
                    if len(final_docs) >= top_n:
                        break
                    content_hash = hash(doc.page_content)
                    if content_hash not in seen_contents:
                        final_docs.append(doc)
                        seen_contents.add(content_hash)

        # [전략: global_top] 쿼리 구분 없이 전체 점수순 (기본값)
        else:
            all_flattened = [d for group in reranked_multi_docs for d in group]
            all_flattened.sort(key=lambda x: x.metadata.get("rerank_score", 0.0), reverse=True)
            for doc in all_flattened:
                if len(final_docs) >= top_n:
                    break
                content_hash = hash(doc.page_content)
                if content_hash not in seen_contents:
                    final_docs.append(doc)
                    seen_contents.add(content_hash)

        # 3. 문서 리스트를 하나의 컨텍스트 문자열로 변환
        # 기존에 정의된 build_context 함수를 호출하여 가독성 있게 합침
        context_string = build_context(final_docs, max_chars=max_chars)

        return RetrievalResponse(
            question="",  # 필요 시 외부에서 주입 가능
            context=context_string,
        )

    return RunnableLambda(_merge)
