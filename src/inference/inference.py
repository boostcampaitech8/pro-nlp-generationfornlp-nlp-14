import sys  # noqa: I001

import pandas as pd
from chains.nodes.mcq_head_nodes import (
    create_choice_scorer,
    create_forward,
    decode_prediction,
    format_rows,
)
from tqdm import tqdm

from data.data_processing import create_test_prompt_messages, load_and_parse_data
from utils import InferenceConfig
from .inference_utils import load_model


def main(config: InferenceConfig):
    """추론 메인 함수

    Args:
        config: 추론 설정 객체
    """
    model, tokenizer = load_model(config.checkpoint_path, config.max_seq_length)

    # 테스트 데이터 로드 및 전처리
    test_df = load_and_parse_data(config.test_data)
    test_dataset = create_test_prompt_messages(test_df)

    # 추론 실행
    infer_results = []

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

    config_path = sys.argv[1]
    config = InferenceConfig.from_yaml(config_path)
    main(config)
