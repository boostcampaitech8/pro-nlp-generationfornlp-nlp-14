import os
import sys
from itertools import zip_longest
from typing import Any

import pandas as pd
from dotenv import load_dotenv
from langchain_core.runnables import (
    RunnableBranch,
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)
from langchain_openai import ChatOpenAI
from tqdm import tqdm

from chains.builder import build_mcq_chain, build_retriever
from chains.builder.mcq_request import build_mcq_request
from chains.nodes import (
    build_query_plan_logger,
    build_web_search_docs_logger,
    normalize_request,
)
from chains.planning.coercion import _coerce_plan
from chains.planning.mapper import _to_prompt_input
from chains.retrieval.nodes.build_context import build_context
from chains.retrieval.nodes.doc_to_response import (
    documents_to_retrieval_responses,
)
from data.data_processing import load_and_parse_data
from prompts import get_prompt_manager
from prompts.plan.plan import plan_prompt
from schemas.mcq.rows import PredRow, ScoreRow
from schemas.processed_question import ProcessedQuestion
from schemas.retrieval import RetrievalPlan
from schemas.retrieval.plan import RetrievalRequest
from schemas.retrieval.response import RetrievalResponse
from utils import InferenceConfig

MAX_PARA_CHARS = 10_000
MAX_CTX_CHARS = 4_000
QUERY_PLAN_LOG_PATH = "log/query_decompositions.jsonl"
WEB_SEARCH_DOCS_LOG_PATH = "log/web_search_docs.jsonl"


def main(config: InferenceConfig):
    """추론 메인 함수
    Args:
        config: 추론 설정 객체
    """
    # 테스트 데이터 로드 및 전처리
    test_df = load_and_parse_data(config.test_data)
    prompt_manager = get_prompt_manager(config.prompt_style)
    # logger
    query_plan_logger = build_query_plan_logger(QUERY_PLAN_LOG_PATH)
    web_search_docs_logger = build_web_search_docs_logger(WEB_SEARCH_DOCS_LOG_PATH)

    retriever = build_retriever()

    # -------------------------
    # 1) planner LLM
    # -------------------------
    planner_llm = ChatOpenAI(
        base_url=os.environ["LLAMA_CPP_SERVER_URL"],
        api_key="API_KEY_NOT_NEED",  # type: ignore
        model="LLama_cpp_model",
        temperature=1.0,
    )

    planner = (
        RunnableLambda(_to_prompt_input)
        | plan_prompt
        | planner_llm.with_structured_output(RetrievalPlan)
        | RunnableLambda(_coerce_plan)
    )

    def retrieval(state: dict) -> dict:
        plan: RetrievalPlan = state["plan"]
        data: ProcessedQuestion = state["data"]
        responses_by_query: list[list[RetrievalResponse]] = []

        for req in plan.requests:
            req = normalize_request(req)
            docs = retriever.invoke(req.query, top_k=req.top_k)
            if docs:
                web_search_docs_logger.invoke({"data": data, "req": req, "docs": docs})
            responses_by_query.append(documents_to_retrieval_responses(req.query, docs))

        # 라운드 로빈 방식으로 서로 다른 쿼리의 컨텍스트를 섞어서 배치 (zip_longest)
        responses: list[RetrievalResponse] = []
        if responses_by_query:
            for grouped in zip_longest(*responses_by_query):
                responses.extend(res for res in grouped if res)

        context = build_context(responses, max_chars=MAX_CTX_CHARS)
        return {
            "data": data,
            "plan": plan,
            "external_knowledge": responses,
            "context": context,
        }

    retrieval_step = RunnableLambda(retrieval)

    # -------MCQ chain---------
    qa_chain = build_mcq_chain(config)

    def _build_mcq_prompt_input(state: ProcessedQuestion | dict) -> RetrievalRequest:
        if isinstance(state, dict):
            data = state.get("data")
            context = state.get("context", "")
        return build_mcq_request(prompt_manager, data, context=context)

    mcp_prompt = RunnableLambda(_build_mcq_prompt_input)

    # long path: planner + retrieval
    # planner 출력 보정까지 붙일 거면:
    # planner_step = planner_step | RunnableLambda(fix_plan)
    long_path = (
        RunnableParallel(
            data=RunnablePassthrough(),  # data 그대로 유지
            plan=planner,  # planner 실행
        )
        | query_plan_logger
        | retrieval_step
        | mcp_prompt
        | qa_chain
    ).with_config(tags=["planned"], metadata={"path": "planned", "type": "planned"})
    # short path: planner/retrieval 모두 스킵
    short_path = (mcp_prompt | qa_chain).with_config(
        tags=["no_plan"], metadata={"path": "no_plan", "type": "no_plan"}
    )

    MIN_PARA_CHARS_FOR_PLANNER = 600

    whole_chain = RunnableBranch(
        (
            lambda data: len((data.paragraph or "").strip()) > MIN_PARA_CHARS_FOR_PLANNER,
            short_path,
        ),
        long_path,  # default
    )

    infer_results: list[tuple[PredRow, ScoreRow]] = []
    for _, row in tqdm(test_df.iterrows(), desc="Inference"):
        data = ProcessedQuestion(
            id=row["id"],
            paragraph=row["paragraph"],
            question=row["question"],
            choices=row["choices"],
            question_plus=row.get("question_plus"),
        )
        outs = whole_chain.invoke(
            data,
            config={"run_name": str(data.id)},
        )
        infer_results.append(outs)

    preds, score = map(list, zip(*infer_results, strict=True))
    # 결과 저장
    result_pred_df = pd.DataFrame(preds)
    result_pred_df.to_csv(config.output_path, index=False)

    # score 저장
    output_socore_path = config.output_path.replace(".csv", ".score.csv")
    result_score_df = pd.DataFrame(score)
    result_score_df.to_csv(output_socore_path, index=False)

    print(f"Inference completed. Results saved to {config.output_path}")
    print(result_pred_df)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python inference.py <config_path>")
        print("Example: python inference.py configs/config.yaml")
        sys.exit(1)
    load_dotenv()
    config_path = sys.argv[1]
    config = InferenceConfig.from_yaml(config_path)
    main(config)
