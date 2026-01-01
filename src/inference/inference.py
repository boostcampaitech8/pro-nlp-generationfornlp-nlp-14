import os
import sys

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

from chains.builder.retriever import build_retriever as build_base_retriever
from chains.core.logging import tap
from chains.core.state import QuestionState
from chains.core.utils import round_robin_merge
from chains.planning import build_planner
from chains.qa import build_qa_chain
from chains.retrieval import build_retriever
from chains.retrieval.context_builder import build_context
from data.data_processing import load_and_parse_data
from prompts import get_prompt_manager
from prompts.plan.plan import plan_prompt
from schemas.mcq.rows import PredRow, ScoreRow
from schemas.retrieval import RetrievalPlan
from utils import InferenceConfig

MAX_CTX_CHARS = 4_000
QUERY_PLAN_LOG_PATH = "log/query_decompositions.jsonl"


def main(config: InferenceConfig):
    """추론 메인 함수
    Args:
        config: 추론 설정 객체
    """
    # 테스트 데이터 로드 및 전처리
    test_df = load_and_parse_data(config.test_data)
    prompt_manager = get_prompt_manager(config.prompt_style)

    # -------------------------
    # 1) Base retriever 생성
    # -------------------------
    base_retriever = build_base_retriever()

    # -------------------------
    # 2) Planner 생성
    # -------------------------
    planner_llm = ChatOpenAI(
        base_url=os.environ["LLAMA_CPP_SERVER_URL"],
        api_key="API_KEY_NOT_NEED",  # type: ignore
        model="LLama_cpp_model",
        temperature=1.0,
    )
    planner = build_planner(llm=planner_llm, prompt=plan_prompt)

    # -------------------------
    # 3) Retriever chain 생성
    # -------------------------
    retriever_chain = build_retriever(retriever=base_retriever)

    # -------------------------
    # 4) QA chain 생성
    # -------------------------
    qa_chain = build_qa_chain(config=config, prompt_manager=prompt_manager)

    # -------------------------
    # 5) Retrieval step (planner 결과 → context)
    # -------------------------
    def retrieval_step(state: dict) -> dict:
        """
        RetrievalPlan을 받아 검색하고 context를 생성합니다.

        Input: {"data": QuestionState, "plan": RetrievalPlan}
        Output: {"data": QuestionState, "context": str}
        """
        plan: RetrievalPlan = state["plan"]
        data: QuestionState = state["data"]

        # Retriever chain 실행 → list[QueryResult]
        query_results = retriever_chain.invoke(plan)

        # QueryResult의 documents를 round-robin merge
        docs = round_robin_merge(query_results)

        # Documents → context string
        context = build_context(docs, max_chars=MAX_CTX_CHARS)

        return {
            "data": data,
            "context": context,
        }

    retrieval = RunnableLambda(retrieval_step)

    # -------------------------
    # 6) Long path: planner → retrieval → qa
    # -------------------------
    # Query plan 로깅
    query_plan_logger = tap(QUERY_PLAN_LOG_PATH)

    def log_plan(state: dict) -> dict:
        """Plan을 로깅하면서 state 통과"""
        query_plan_logger.invoke(
            {
                "id": state["data"]["id"],
                "plan": state["plan"],
            }
        )
        return state

    long_path = (
        RunnableParallel(
            data=RunnablePassthrough(),  # data 그대로 유지
            plan=planner,  # planner 실행
        )
        | RunnableLambda(log_plan)  # 로깅
        | retrieval  # 검색 및 context 생성
        | qa_chain  # QA 추론
    ).with_config(tags=["planned"], metadata={"path": "planned", "type": "planned"})

    # -------------------------
    # 7) Short path: qa only (no planning, no retrieval)
    # -------------------------
    short_path = (
        RunnableParallel(
            data=RunnablePassthrough(),  # QuestionState 그대로 전달
            context=RunnableLambda(lambda _: ""),  # 빈 context
        )
        | qa_chain
    ).with_config(tags=["no_plan"], metadata={"path": "no_plan", "type": "no_plan"})

    # -------------------------
    # 8) Branch: paragraph가 짧으면 short path, 아니면 long path
    # -------------------------
    MIN_PARA_CHARS_FOR_PLANNER = 600

    whole_chain = RunnableBranch(
        (
            lambda data: len((data["paragraph"] or "").strip()) > MIN_PARA_CHARS_FOR_PLANNER,
            short_path,
        ),
        long_path,  # default
    )

    infer_results: list[tuple[PredRow, ScoreRow]] = []
    for _, row in tqdm(test_df.iterrows(), desc="Inference"):
        # QuestionState (TypedDict) 생성
        data: QuestionState = {
            **row,
            "len_choices": len(row["choices"]),
        }
        outs = whole_chain.invoke(
            data,
            config={"run_name": str(data["id"])},
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
