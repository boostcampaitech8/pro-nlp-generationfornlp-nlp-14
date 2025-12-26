from peft import AutoPeftModelForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer
from unsloth import FastLanguageModel


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
