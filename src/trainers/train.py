import sys

import evaluate
import numpy as np
import torch
from peft import LoraConfig
from trl import DataCollatorForCompletionOnlyLM, SFTConfig, SFTTrainer

from data.data_processing import (
    create_prompt_messages,
    load_and_parse_data,
    set_seed,
    tokenize_dataset,
)
from models.model_loader import load_model_for_training, load_tokenizer
from utils import TrainConfig, decode_labels, extract_choice_logits, setup_wandb


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
    return extract_choice_logits(logits, tokenizer, position=-2)


def compute_metrics(evaluation_result, tokenizer, acc_metric):
    """metric 계산 함수"""
    logits, labels = evaluation_result

    labels = decode_labels(labels, tokenizer)

    probs = torch.nn.functional.softmax(torch.tensor(logits), dim=-1)
    predictions = np.argmax(probs, axis=-1)

    acc = acc_metric.compute(predictions=predictions, references=labels)
    return acc


def main(config: TrainConfig):
    """학습 메인 함수

    Args:
        config: 학습 설정 객체
    """
    set_seed(config.seed)

    setup_wandb(
        project=config.wandb_project,
        run_name=config.wandb_run_name,
        config=config,
    )

    # 모델 및 토크나이저 로드
    model = load_model_for_training(config.model_name)
    tokenizer = load_tokenizer(config.model_name)

    # 데이터 로드 및 전처리
    df = load_and_parse_data(config.train_data)
    processed_dataset = create_prompt_messages(df)
    train_dataset, eval_dataset = tokenize_dataset(
        processed_dataset,
        tokenizer,
        max_seq_length=config.max_seq_length,
        test_size=config.eval_ratio,
        seed=config.seed,
    )

    # LoRA 설정
    peft_config = get_peft_config(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
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
        max_seq_length=config.max_seq_length,
        output_dir=config.output_dir,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        num_train_epochs=config.epochs,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        logging_steps=config.logging_steps,
        logging_strategy=config.logging_strategy,
        save_strategy="epoch",
        evaluation_strategy="epoch",
        save_total_limit=2,
        save_only_model=True,
        report_to="wandb",
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
    print(f"Training completed. Model saved to {config.output_dir}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python train.py <config_path>")
        print("Example: python train.py configs/config.yaml")
        sys.exit(1)

    config_path = sys.argv[1]
    config = TrainConfig.from_yaml(config_path)
    main(config)
