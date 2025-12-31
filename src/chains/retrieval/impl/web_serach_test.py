"""
간단한 검색 확인용 스크립트.
- TAVILY_API_KEY가 설정돼 있어야 정상 동작한다.
- 기본 옵션은 `build_retriever`와 비슷하게 max_results를 top_k로 맞춘다.
"""

from __future__ import annotations

import argparse
from pprint import pprint

from tavily import TavilyClient

from chains.builder.retriever import WEBSEARCH_EXCLUDE_DOMAINS
from chains.retrieval.impl.tavily_web_search import TavilyWebSearchService
from schemas.retrieval.plan import RetrievalRequest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tavily 웹 검색 테스트")
    parser.add_argument("--top-k", type=int, default=5, help="가져올 문서 개수 (기본 5)")
    parser.add_argument(
        "--topic",
        choices=["general", "news", "finance"],
        default="general",
        help="검색 토픽 (기본 general)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    client = TavilyClient()  # TAVILY_API_KEY 필요
    service = TavilyWebSearchService(
        client,
        options={
            "max_results": args.top_k,
            "exclude_domains": WEBSEARCH_EXCLUDE_DOMAINS,
            "topic": args.topic,
            "include_domains": ["https://www.scourt.go.kr/", "https://www.law.go.kr/"],
        },
    )

    print("검색 쿼리를 입력하세요. 빈 줄(또는 Ctrl+D/Ctrl+C) 입력 시 종료합니다.")
    while True:
        try:
            query = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n종료합니다.")
            break

        if not query:
            print("빈 입력으로 종료합니다.")
            break

        req = RetrievalRequest(query=query, top_k=args.top_k)
        results = service.search(req)

        if not results:
            print("검색 결과가 없습니다.")
            continue

        for i, r in enumerate(results):
            title = (r.metadata or {}).get("title")
            print(f"\n[{i}] {title or '(제목 없음)'}")
            print("-" * 60)
            print(r.context)
            print("\nmetadata:")
            pprint(r.metadata)


if __name__ == "__main__":
    main()
