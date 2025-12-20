import sys  # noqa: I001
from unsloth import FastLanguageModel

import pandas as pd
import torch
from peft import AutoPeftModelForCausalLM
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from data.data_processing import create_test_prompt_messages, load_and_parse_data
from utils import InferenceConfig, get_choice_token_ids, logits_to_prediction


def load_model(model_path: str):
    """checkpoint 경로에 따라 모델/토크나이저를 로드한다."""
    # NOTE: config.yaml로 분리할지 논의 필요
    FLM_SUPPORT_MODELS = "unsloth"
    # FIXME: why it return True always
    is_fast_candidate = any(k in model_path.lower() for k in FLM_SUPPORT_MODELS)
    if is_fast_candidate:
        print("--" * 10, "TRY USE UNSLOTH", "--" * 10)
        try:
            model, tokenizer = FastLanguageModel.from_pretrained(
                # NOTE: max_seq_length 설정 필요 기본값 2048
                model_name=model_path,
                load_in_4bit=True,
            )
            FastLanguageModel.for_inference(model)
            return model, tokenizer
        except Exception as e:
            # Fast 경로 실패 시 기본 로더로 폴백
            print(f"FastLanguageModel load failed, fallback to HF loader. reason={e}")

    if model_path[:7] == "outputs":
        model = AutoPeftModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            device_map="auto",
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            device_map="auto",
        )
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
    )
    model.eval()
    return model, tokenizer


def main(config: InferenceConfig):
    """추론 메인 함수

    Args:
        config: 추론 설정 객체
    """
    model, tokenizer = load_model(config.checkpoint_path)

    # 테스트 데이터 로드 및 전처리
    test_df = load_and_parse_data(config.test_data)
    test_dataset = create_test_prompt_messages(test_df)

    # 추론 실행
    infer_results = []

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
        print("Example: python inference.py configs/config.yaml")
        sys.exit(1)

    config_path = sys.argv[1]
    config = InferenceConfig.from_yaml(config_path)
    main(config)
