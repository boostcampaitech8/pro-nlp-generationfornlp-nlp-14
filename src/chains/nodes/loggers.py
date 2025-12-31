from pathlib import Path
from typing import Any, TypedDict

from langchain_core.runnables import RunnableLambda

from chains.nodes.tap import tap
from schemas.processed_question import ProcessedQuestion
from schemas.retrieval import RetrievalPlan
from schemas.retrieval.plan import RetrievalRequest


class QueryPlanLoggerState(TypedDict):
    data: ProcessedQuestion
    plan: RetrievalPlan


class WebSearchDocsPayload(TypedDict):
    data: ProcessedQuestion
    req: RetrievalRequest
    docs: list[Any]


def normalize_request(req: RetrievalRequest | dict[str, Any]) -> RetrievalRequest:
    return req if isinstance(req, RetrievalRequest) else RetrievalRequest(**req)


def build_query_plan_logger(path: str | Path) -> RunnableLambda:
    def build_record(state: QueryPlanLoggerState) -> dict[str, Any]:
        plan = state["plan"]
        data = state["data"]
        return {
            "question_id": data.id,
            "question": data.question,
            "requests": [
                {"query": r.query, "top_k": r.top_k}
                for r in (normalize_request(req) for req in plan.requests)
            ],
        }

    return tap(path, build_record)


def build_web_search_docs_logger(path: str | Path) -> RunnableLambda:
    def build_record(payload: WebSearchDocsPayload) -> list[dict[str, Any]]:
        data = payload["data"]
        req = payload["req"]
        docs = payload["docs"]
        return [
            {
                "question_id": data.id,
                "query": req.query,
                "top_k": req.top_k,
                "rank": i,
                "document": {
                    "page_content": getattr(doc, "page_content", ""),
                    "metadata": dict(getattr(doc, "metadata", {}) or {}),
                },
            }
            for i, doc in enumerate(docs)
        ]

    return tap(path, build_record)
