import sys  # noqa: I001
from unsloth import FastLanguageModel

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
from prompts import get_prompt_manager
from utils import InferenceConfig


def load_model(model_path: str, max_seq_length: int | None = None):
    """checkpoint 경로에 따라 모델/토크나이저를 로드한다."""
    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_path,
            max_seq_length=max_seq_length,
            load_in_4bit=True,
        )
        FastLanguageModel.for_inference(model)
        return model, tokenizer
    except Exception as e:
        # Fast 경로 실패 시 기본 로더로 폴백
        print(f"[WARN]: FastLanguageModel load failed, fallback to HF loader. reason={e}")

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
    model, tokenizer = load_model(config.checkpoint_path, config.max_seq_length)

    # 테스트 데이터 로드 및 전처리
    test_df = load_and_parse_data(config.test_data)
    prompt_manager = get_prompt_manager(config.prompt_style)
    test_dataset = create_test_prompt_messages(test_df, prompt_manager)

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
