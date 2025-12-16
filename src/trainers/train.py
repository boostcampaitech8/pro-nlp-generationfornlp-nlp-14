import argparse

import evaluate
import numpy as np
import torch
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DataCollatorForCompletionOnlyLM, SFTConfig, SFTTrainer

import sys
sys.path.append("src")
from data.data_processing import (
    create_prompt_messages,
    load_and_parse_data,
    set_seed,
    tokenize_dataset,
)


CHAT_TEMPLATE = "{% if messages[0]['role'] == 'system' %}{% set system_message = messages[0]['content'] %}{% endif %}{% if system_message is defined %}{{ system_message }}{% endif %}{% for message in messages %}{% set content = message['content'] %}{% if message['role'] == 'user' %}{{ '<start_of_turn>user\n' + content + '<end_of_turn>\n<start_of_turn>model\n' }}{% elif message['role'] == 'assistant' %}{{ content + '<end_of_turn>\n' }}{% endif %}{% endfor %}"


def get_peft_config(r: int = 6, lora_alpha: int = 8, lora_dropout: float = 0.05):
    """LoRA 설정 반환"""
    return LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=["q_proj", "k_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )


def preprocess_logits_for_metrics(logits, labels, tokenizer):
    """모델의 logits를 조정하여 정답 토큰 부분만 출력"""
    logits = logits if not isinstance(logits, tuple) else logits[0]
    logit_idx = [
        tokenizer.vocab["1"],
        tokenizer.vocab["2"],
        tokenizer.vocab["3"],
        tokenizer.vocab["4"],
        tokenizer.vocab["5"],
    ]
    logits = logits[:, -2, logit_idx]
    return logits


def compute_metrics(evaluation_result, tokenizer, acc_metric):
    """metric 계산 함수"""
    logits, labels = evaluation_result
    int_output_map = {"1": 0, "2": 1, "3": 2, "4": 3, "5": 4}

    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    labels = list(map(lambda x: x.split("<end_of_turn>")[0].strip(), labels))
    labels = list(map(lambda x: int_output_map[x], labels))

    probs = torch.nn.functional.softmax(torch.tensor(logits), dim=-1)
    predictions = np.argmax(probs, axis=-1)

    acc = acc_metric.compute(predictions=predictions, references=labels)
    return acc


def main(args):
    set_seed(args.seed)

    # 모델 및 토크나이저 로드
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=True,
    )
    tokenizer.chat_template = CHAT_TEMPLATE
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"

    # 데이터 로드 및 전처리
    df = load_and_parse_data(args.train_data)
    processed_dataset = create_prompt_messages(df)
    train_dataset, eval_dataset = tokenize_dataset(
        processed_dataset,
        tokenizer,
        max_seq_length=args.max_seq_length,
        test_size=args.eval_ratio,
        seed=args.seed,
    )

    # LoRA 설정
    peft_config = get_peft_config(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    )

    # Data Collator 설정
    response_template = "<start_of_turn>model"
    data_collator = DataCollatorForCompletionOnlyLM(
        response_template=response_template,
        tokenizer=tokenizer,
    )

    # Metric 설정
    acc_metric = evaluate.load("accuracy")

    # SFT Config 설정
    sft_config = SFTConfig(
        do_train=True,
        do_eval=True,
        lr_scheduler_type="cosine",
        max_seq_length=args.max_seq_length,
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        logging_steps=args.logging_steps,
        save_strategy="epoch",
        evaluation_strategy="epoch",
        save_total_limit=2,
        save_only_model=True,
        report_to="none",
    )

    # Trainer 설정
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=lambda x: compute_metrics(x, tokenizer, acc_metric),
        preprocess_logits_for_metrics=lambda logits, labels: preprocess_logits_for_metrics(
            logits, labels, tokenizer
        ),
        peft_config=peft_config,
        args=sft_config,
    )

    # 학습 실행
    trainer.train()
    print(f"Training completed. Model saved to {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model with LoRA")
    parser.add_argument(
        "--model_name", type=str, default="beomi/gemma-ko-2b", help="Model name or path"
    )
    parser.add_argument(
        "--train_data", type=str, required=True, help="Path to training data CSV"
    )
    parser.add_argument(
        "--output_dir", type=str, default="outputs_gemma", help="Output directory"
    )
    parser.add_argument("--max_seq_length", type=int, default=1024, help="Max sequence length")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--logging_steps", type=int, default=1, help="Logging steps")
    parser.add_argument("--eval_ratio", type=float, default=0.1, help="Evaluation ratio")
    parser.add_argument("--lora_r", type=int, default=6, help="LoRA r")
    parser.add_argument("--lora_alpha", type=int, default=8, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()
    main(args)
