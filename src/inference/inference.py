import sys

import pandas as pd
from chains.nodes.mcq_head_nodes import (
    create_choice_scorer,
    create_forward,
    decode_prediction,
    format_rows,
)
from peft import AutoPeftModelForCausalLM
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from data.data_processing import create_test_prompt_messages, load_and_parse_data
from utils import InferenceConfig


def main(config: InferenceConfig):
    """추론 메인 함수

    Args:
        config: 추론 설정 객체
    """
    if config.checkpoint_path[:7] == "outputs":
        # 모델 및 토크나이저 로드
        model = AutoPeftModelForCausalLM.from_pretrained(
            config.checkpoint_path,
            trust_remote_code=True,
            device_map="auto",
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            config.checkpoint_path,
            trust_remote_code=True,
            device_map="auto",
        )
    tokenizer = AutoTokenizer.from_pretrained(
        config.checkpoint_path,
        trust_remote_code=True,
    )

    # 테스트 데이터 로드 및 전처리
    test_df = load_and_parse_data(config.test_data)
    test_dataset = create_test_prompt_messages(test_df)

    # 추론 실행
    infer_results = []

    model.eval()

    # 의존성있는 node들 생성
    llm = create_forward(model, tokenizer)
    compute = create_choice_scorer(tokenizer)
    # create chain
    qa_chain = llm | compute | decode_prediction | format_rows
    for data in tqdm(test_dataset, desc="Inference"):
        outs = qa_chain.invoke(data)
        infer_results.append(outs)

    preds, score = map(list, zip(*infer_results, strict=True))
    # 결과 저장
    result_pred_df = pd.DataFrame(preds)
    result_pred_df.to_csv(config.output_path, index=False)
    result_score_df = pd.DataFrame(score)
    # NOTE config경로 뚫어줘야함
    result_score_df.to_csv("outputs/scores.csv", index=False)

    print(f"Inference completed. Results saved to {config.output_path}")
    print(result_pred_df)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python inference.py <config_path>")
        print("Example: python inference.py configs/config.yaml")
        sys.exit(1)

    config_path = sys.argv[1]
    config = InferenceConfig.from_yaml(config_path)
    main(config)
