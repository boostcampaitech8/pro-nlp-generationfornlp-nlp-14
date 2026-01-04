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

from chains.core.logging import tap
from chains.core.state import QuestionState
from chains.planning import build_planner
from chains.qa import build_qa_chain
from chains.reranker import build_reranker, merge_strategies
from chains.retrieval import build_multi_query_retriever, build_tavily_retriever
from data.data_processing import load_and_parse_data
from prompts import get_prompt_manager
from prompts.plan.plan import plan_prompt
from schemas.mcq.rows import PredRow, ScoreRow
from utils import InferenceConfig


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
    base_retriever = build_tavily_retriever()

    # -------------------------
    # 2) Planner 생성
    # -------------------------
    planner_llm = ChatOpenAI(
        base_url=os.environ["LLAMA_CPP_SERVER_URL"],
        api_key="API_KEY_NOT_NEED",  # type: ignore
        model="LLama_cpp_model",
        temperature=config.planner_llm_temperature,
    )
    planner = build_planner(llm=planner_llm, prompt=plan_prompt)

    #  -------------------------
    # Reranker 생성
    # -------------------------
    reranker = build_reranker(base_url=os.environ["RERANKER_URL"])
    merger = merge_strategies(
        strategy_type="query_first",  # 전략 유형: query_first or global_top
        top_n=config.num_retrieved_docs,
        max_chars=config.max_retrieval_context_chars,
    )

    # -------------------------
    # 3) retriever chain 생성
    # -------------------------
    # 리랭커는 원본 데이터(data)와 검색된 문서들(multi_docs)이 모두 필요하므로 구조를 묶어줍니다.
    retrieval_chain = (
        RunnableParallel(
            data=RunnableLambda(lambda x: x["data"]),
            multi_docs=RunnableLambda(lambda x: x["plan"])
            | build_multi_query_retriever(retriever=base_retriever),
        )
        | reranker
        | merger
        | (lambda res: res.context)
    )

    # -------------------------
    # 4) Context chain 생성 -> paragraph 길이에 따라 Branch
    # -------------------------
    query_plan_logger = tap(config.query_plan_log_path)

    def log_plan(state: dict) -> dict:
        """Plan을 로깅하면서 state 통과"""
        query_plan_logger.invoke(
            {
                "id": state["data"]["id"],
                "plan": state["plan"],
            }
        )
        return state

    context_chain = RunnableBranch(
        (
            lambda x: len((x["paragraph"] or "").strip())
            < config.max_paragraph_chars_for_planner,  # paragraph가 짧으면
            (
                RunnableParallel(data=RunnablePassthrough(), plan=planner)
                | RunnableLambda(log_plan)
                | retrieval_chain  # 리랭커가 포함된 retrieval_chain 실행
            ),  # retrieval 실행
        ),
        lambda _: "",  # paragraph가 길면 빈 context
    )

    # -------------------------
    # 5) QA chain 생성
    # -------------------------
    qa_chain = build_qa_chain(config=config, prompt_manager=prompt_manager)

    # -------------------------
    # 6) Whole chain 생성
    # -------------------------
    whole_chain = (
        RunnableParallel(
            data=RunnablePassthrough(),
            context=context_chain,
        )
        | qa_chain
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
