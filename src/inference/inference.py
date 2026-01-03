import os
import sys

import pandas as pd
from dotenv import load_dotenv
from langchain_core.runnables import (
    RunnableBranch,
    RunnablePassthrough,
)
from langchain_openai import ChatOpenAI
from tqdm import tqdm

from chains.planning import build_planner
from chains.qa import build_qa_chain
from chains.retrieval import build_multi_query_retriever, build_tavily_retriever
from chains.retrieval.context_builder import build_context
from chains.runnables.conditions import is_shorter_than
from chains.runnables.logging import tap
from chains.runnables.selectors import constant, selector
from chains.utils.utils import round_robin_merge
from data.data_processing import load_and_parse_data
from prompts import get_prompt_manager
from prompts.plan.plan import plan_prompt
from schemas.mcq.rows import PredRow, ScoreRow
from schemas.question import PreprocessedQuestion
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
    # 1) Planner 생성
    # -------------------------
    planner_llm = ChatOpenAI(
        base_url=os.environ["LLAMA_CPP_SERVER_URL"],
        api_key="API_KEY_NOT_NEED",  # type: ignore
        name="Planner_Model",
        model="LLama_cpp_model",
        temperature=config.planner_llm_temperature,
    )
    planner = build_planner(llm=planner_llm, prompt=plan_prompt)

    # -------------------------
    # 2) Retriever 생성
    # -------------------------
    retriever = build_multi_query_retriever(build_tavily_retriever())

    # -------------------------
    # 3) Augmentation chain (조건부: paragraph 짧고 RAG 사용 시에만 planning → retrieval → context)
    # -------------------------
    plan_logger = tap(
        config.query_plan_log_path, lambda x: {"id": x["data"]["id"], "plan": x["plan"]}
    )

    augmentation_chain = RunnablePassthrough.assign(
        context=RunnableBranch(
            (
                lambda x: is_shorter_than(
                    "data", "paragraph", max_chars=config.max_paragraph_chars_for_planner
                ).invoke(x)
                and config.use_rag,  # paragraph가 짧고 RAG 사용 시에만 planner + retrieval 실행
                selector("data")
                | RunnablePassthrough.assign(plan=planner)
                | plan_logger  # plan 로그 저장
                | selector("plan")
                | retriever  # 리랭커도입시 | RunnablePassthrough.assign(docs=selector("plan") | retriever)
                # 리랭커 삽입 지점: | reranker
                | round_robin_merge  # will be replaced merge strategy
                | (lambda docs: build_context(docs, max_chars=config.max_retrieval_context_chars)),
            ),
            # paragraph가 길거나 RAG 미사용 시 빈 context
            constant(""),
        ),
    )

    # -------------------------
    # 4) QA chain 생성
    # -------------------------
    qa_chain = build_qa_chain(config=config, prompt_manager=prompt_manager)

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
