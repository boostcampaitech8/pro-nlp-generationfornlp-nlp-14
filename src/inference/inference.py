import sys  # noqa: I001

import pandas as pd

from tqdm import tqdm

from data.data_processing import create_test_prompt_messages, load_and_parse_data
from utils import InferenceConfig
from chains.mcq_chain import create_local_mcq_chain


def main(config: InferenceConfig):
    """추론 메인 함수

    Args:
        config: 추론 설정 객체
    """

    # 테스트 데이터 로드 및 전처리
    test_df = load_and_parse_data(config.test_data)
    test_dataset = create_test_prompt_messages(test_df)

    # 추론 실행
    infer_results = []

    # create chain
    qa_chain = create_local_mcq_chain(config)
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
