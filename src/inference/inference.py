import os
import sys

import pandas as pd
from dotenv import load_dotenv
from langchain_classic.retrievers.ensemble import EnsembleRetriever
from langchain_core.runnables import (
    RunnableBranch,
    RunnablePassthrough,
)
from langchain_openai import ChatOpenAI
from tqdm import tqdm

from chains.planning import build_planner
from chains.qa.inference.not_think_cot import build_non_think_cot_chain
from chains.reranker import build_reranker, merge_strategies
from chains.retrieval import (
    build_multi_query_retriever,
    contents_quality_filter,
    create_websearch_retriever,
)
from chains.runnables.conditions import all_conditions, constant_check, is_shorter_than
from chains.runnables.logging import tap
from chains.runnables.selectors import constant, selector
from data.data_processing import load_and_parse_data
from prompts import get_prompt_manager
from prompts.plan.plan import plan_prompt
from schemas.mcq.rows import PredRow, ScoreRow
from schemas.question import PreprocessedQuestion
from utils import InferenceConfig
from utils.config_loader import RetrievalConfig


def main(inference_config: InferenceConfig, retrieval_config: RetrievalConfig):
    """추론 메인 함수
    Args:
        config: 추론 설정 객체
    """
    # 테스트 데이터 로드 및 전처리
    test_df = load_and_parse_data(inference_config.test_data)
    prompt_manager = get_prompt_manager(inference_config.prompt_style)

    # -------------------------
    # 1) Planner 생성
    # -------------------------
    planner_llm = ChatOpenAI(
        base_url=os.environ["LLAMA_CPP_SERVER_URL"],
        api_key="API_KEY_NOT_NEED",  # type: ignore
        name="Planner_Model",
        model="LLama_cpp_model",
        temperature=inference_config.planner_llm_temperature,
    )
    planner = build_planner(llm=planner_llm, prompt=plan_prompt)

    #  -------------------------
    # Reranker 생성
    # -------------------------
    reranker = build_reranker(base_url=os.environ["RERANKER_URL"])
    merger = merge_strategies(
        strategy_type="query_first",  # 전략 유형: query_first or global_top
        top_n=inference_config.num_retrieved_docs,
        max_chars=inference_config.max_retrieval_context_chars,
    )

    # -------------------------
    # 2) Retriever 생성 (EnsembleRetriever: Local + Web)
    # -------------------------
    # local_retriever = create_local_retriever(config=retrieval_config)

    websearch_retriever = create_websearch_retriever()

    print(retrieval_config.local_retriever_weight, retrieval_config.web_retriever_weight)
    # EnsembleRetriever로 로컬과 웹 검색 결합 (가중치: local 0.6, web 0.4)
    ensemble_retriever = EnsembleRetriever(
        retrievers=[websearch_retriever],
        weights=[1],
    )

    multi_query_retriever = build_multi_query_retriever(ensemble_retriever)

    # -------------------------
    # 3) Augmentation chain (조건부: paragraph 짧고 RAG 사용 시에만 planning → retrieval → context)
    # -------------------------
    plan_logger = tap(
        inference_config.query_plan_log_path, lambda x: {"id": x["data"]["id"], "plan": x["plan"]}
    )

    augmentation_chain = RunnablePassthrough.assign(
        context=RunnableBranch(
            (
                all_conditions(
                    is_shorter_than(
                        "data",
                        "paragraph",
                        max_chars=inference_config.max_paragraph_chars_for_planner,
                    ),
                    constant_check(inference_config.use_rag),
                ),  # paragraph가 짧고 RAG 사용 시에만 planner + retrieval 실행
                selector("data")
                | RunnablePassthrough.assign(plan=planner)
                | plan_logger  # plan 로그 저장
                | RunnablePassthrough.assign(
                    multi_docs=(
                        selector("plan")
                        | multi_query_retriever
                        | contents_quality_filter(
                            min_length=50,
                            max_length=8000,
                            min_korean_ratio=0.1,
                        )
                    )
                )
                | reranker  # Reranker: list[list[Document]] -> list[list[Document]] (with rerank_score)
                | merger  # Merger: list[list[Document]] -> RetrievalResponse (with context string)
                | (lambda response: response.context),
            ),
            # paragraph가 길거나 RAG 미사용 시 빈 context
            constant(""),
        ),
    )

    # -------------------------
    # 4) QA chain 생성
    # -------------------------
    qa_chain = build_non_think_cot_chain(config=inference_config, prompt_manager=prompt_manager)

    # -------------------------
    # 5) Whole chain 생성
    # -------------------------
    whole_chain = augmentation_chain | qa_chain

    infer_results: list[tuple[PredRow, ScoreRow]] = []
    for _, row in tqdm(test_df.iterrows(), desc="Inference"):
        # PreprocessedQuestion (TypedDict) 생성
        question_data: PreprocessedQuestion = {
            **row,
            "len_choices": len(row["choices"]),
        }
        # PipelineState로 wrap하여 invoke
        outs = whole_chain.invoke(
            {"data": question_data},
            config={"run_name": str(question_data["id"])},
        )
        infer_results.append(outs)

    preds, score = map(list, zip(*infer_results, strict=True))
    # 결과 저장
    result_pred_df = pd.DataFrame(preds)
    result_pred_df.to_csv(inference_config.output_path, index=False)

    # score 저장
    output_socore_path = inference_config.output_path.replace(".csv", ".score.csv")
    result_score_df = pd.DataFrame(score)
    result_score_df.to_csv(output_socore_path, index=False)

    print(f"Inference completed. Results saved to {inference_config.output_path}")
    print(result_pred_df)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python inference.py <config_path>")
        print("Example: python inference.py configs/config.yaml")
        sys.exit(1)
    load_dotenv()
    config_path = sys.argv[1]
    inference_config = InferenceConfig.from_yaml(config_path)
    retrieval_config = RetrievalConfig.from_yaml(config_path)
    main(
        inference_config,
        retrieval_config,
    )
