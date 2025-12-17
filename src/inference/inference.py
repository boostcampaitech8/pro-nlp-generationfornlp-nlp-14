import sys

import pandas as pd
import torch
from peft import AutoPeftModelForCausalLM
from tqdm import tqdm
from transformers import AutoTokenizer

from data.data_processing import create_test_prompt_messages, load_and_parse_data
from utils import InferenceConfig, get_choice_token_ids, logits_to_prediction


def main(config: InferenceConfig):
    """추론 메인 함수

    Args:
        config: 추론 설정 객체
    """
    # 모델 및 토크나이저 로드
    model = AutoPeftModelForCausalLM.from_pretrained(
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
    with torch.inference_mode():
        for data in tqdm(test_dataset, desc="Inference"):
            _id = data["id"]
            messages = data["messages"]
            len_choices = data["len_choices"]

            outputs = model(
                tokenizer.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_tensors="pt",
                ).to("cuda")
            )

            logits = outputs.logits[:, -1].flatten().cpu()

            choice_ids = get_choice_token_ids(tokenizer, len_choices)
            target_logits = torch.tensor([logits[idx] for idx in choice_ids])

            predict_value = logits_to_prediction(target_logits, len_choices)
            infer_results.append({"id": _id, "answer": predict_value})

    # 결과 저장
    result_df = pd.DataFrame(infer_results)
    result_df.to_csv(config.output_path, index=False)
    print(f"Inference completed. Results saved to {config.output_path}")
    print(result_df)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python inference.py <config_path>")
        print("Example: python inference.py configs/inference.yaml")
        sys.exit(1)

    config_path = sys.argv[1]
    config = InferenceConfig.from_yaml(config_path)
    main(config)
