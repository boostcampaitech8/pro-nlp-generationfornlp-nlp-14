import argparse

import numpy as np
import pandas as pd
import torch
from peft import AutoPeftModelForCausalLM
from tqdm import tqdm
from transformers import AutoTokenizer

import sys
sys.path.append("src")
from data.data_processing import create_test_prompt_messages, load_and_parse_data


def main(args):
    # 모델 및 토크나이저 로드
    model = AutoPeftModelForCausalLM.from_pretrained(
        args.checkpoint_path,
        trust_remote_code=True,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.checkpoint_path,
        trust_remote_code=True,
    )

    # 테스트 데이터 로드 및 전처리
    test_df = load_and_parse_data(args.test_data)
    test_dataset = create_test_prompt_messages(test_df)

    # 추론 실행
    infer_results = []
    pred_choices_map = {0: "1", 1: "2", 2: "3", 3: "4", 4: "5"}

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

            target_logit_list = [
                logits[tokenizer.vocab[str(i + 1)]] for i in range(len_choices)
            ]

            probs = (
                torch.nn.functional.softmax(
                    torch.tensor(target_logit_list, dtype=torch.float32), dim=-1
                )
                .detach()
                .cpu()
                .numpy()
            )

            predict_value = pred_choices_map[np.argmax(probs, axis=-1)]
            infer_results.append({"id": _id, "answer": predict_value})

    # 결과 저장
    result_df = pd.DataFrame(infer_results)
    result_df.to_csv(args.output_path, index=False)
    print(f"Inference completed. Results saved to {args.output_path}")
    print(result_df)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference with trained model")
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--test_data",
        type=str,
        required=True,
        help="Path to test data CSV",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="output.csv",
        help="Path to save inference results",
    )

    args = parser.parse_args()
    main(args)
