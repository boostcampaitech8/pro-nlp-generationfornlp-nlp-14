import random
from ast import literal_eval

import numpy as np
import pandas as pd
import torch
from datasets import Dataset

PROMPT_NO_QUESTION_PLUS = """지문:
{paragraph}

질문:
{question}

선택지:
{choices}

1, 2, 3, 4, 5 중에 하나를 정답으로 고르세요.
정답:"""

PROMPT_QUESTION_PLUS = """지문:
{paragraph}

질문:
{question}

<보기>:
{question_plus}

선택지:
{choices}

1, 2, 3, 4, 5 중에 하나를 정답으로 고르세요.
정답:"""


def set_seed(random_seed: int = 42):
    """난수 고정"""
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)


def load_and_parse_data(csv_path: str) -> pd.DataFrame:
    """CSV 파일 로드 및 JSON 파싱"""
    dataset = pd.read_csv(csv_path)

    records = []
    for _, row in dataset.iterrows():
        problems = literal_eval(row["problems"])
        record = {
            "id": row["id"],
            "paragraph": row["paragraph"],
            "question": problems["question"],
            "choices": problems["choices"],
            "answer": problems.get("answer", None),
            "question_plus": problems.get("question_plus", None),
        }
        records.append(record)

    return pd.DataFrame(records)


def create_prompt_messages(df: pd.DataFrame) -> list[dict]:
    """프롬프트 메시지 생성"""
    dataset = Dataset.from_pandas(df)
    processed_dataset = []

    for i in range(len(dataset)):
        choices_string = "\n".join(
            [f"{idx + 1} - {choice}" for idx, choice in enumerate(dataset[i]["choices"])]
        )

        if dataset[i]["question_plus"]:
            user_message = PROMPT_QUESTION_PLUS.format(
                paragraph=dataset[i]["paragraph"],
                question=dataset[i]["question"],
                question_plus=dataset[i]["question_plus"],
                choices=choices_string,
            )
        else:
            user_message = PROMPT_NO_QUESTION_PLUS.format(
                paragraph=dataset[i]["paragraph"],
                question=dataset[i]["question"],
                choices=choices_string,
            )

        processed_dataset.append(
            {
                "id": dataset[i]["id"],
                "messages": [
                    {"role": "system", "content": "지문을 읽고 질문의 답을 구하세요."},
                    {"role": "user", "content": user_message},
                    {"role": "assistant", "content": f"{dataset[i]['answer']}"},
                ],
                "label": dataset[i]["answer"],
            }
        )

    return processed_dataset


def create_test_prompt_messages(df: pd.DataFrame) -> list[dict]:
    """테스트용 프롬프트 메시지 생성 (정답 없음)"""
    test_dataset = []

    for _, row in df.iterrows():
        choices_string = "\n".join(
            [f"{idx + 1} - {choice}" for idx, choice in enumerate(row["choices"])]
        )
        len_choices = len(row["choices"])

        if row["question_plus"]:
            user_message = PROMPT_QUESTION_PLUS.format(
                paragraph=row["paragraph"],
                question=row["question"],
                question_plus=row["question_plus"],
                choices=choices_string,
            )
        else:
            user_message = PROMPT_NO_QUESTION_PLUS.format(
                paragraph=row["paragraph"],
                question=row["question"],
                choices=choices_string,
            )

        test_dataset.append(
            {
                "id": row["id"],
                "messages": [
                    {"role": "system", "content": "지문을 읽고 질문의 답을 구하세요."},
                    {"role": "user", "content": user_message},
                ],
                "label": row["answer"],
                "len_choices": len_choices,
            }
        )

    return test_dataset


def tokenize_dataset(
    processed_dataset: list[dict],
    tokenizer,
    max_seq_length: int = 1024,
    test_size: float = 0.1,
    seed: int = 42,
):
    """토큰화 및 train/eval 분리"""

    def formatting_prompts_func(example):
        output_texts = []
        for i in range(len(example["messages"])):
            output_texts.append(
                tokenizer.apply_chat_template(
                    example["messages"][i],
                    tokenize=False,
                )
            )
        return output_texts

    def tokenize(element):
        outputs = tokenizer(
            formatting_prompts_func(element),
            truncation=False,
            padding=False,
            return_overflowing_tokens=False,
            return_length=False,
        )
        return {
            "input_ids": outputs["input_ids"],
            "attention_mask": outputs["attention_mask"],
        }

    dataset = Dataset.from_pandas(pd.DataFrame(processed_dataset))
    tokenized_dataset = dataset.map(
        tokenize,
        remove_columns=list(dataset.features),
        batched=True,
        num_proc=4,
        load_from_cache_file=True,
        desc="Tokenizing",
    )

    tokenized_dataset = tokenized_dataset.filter(lambda x: len(x["input_ids"]) <= max_seq_length)
    tokenized_dataset = tokenized_dataset.train_test_split(test_size=test_size, seed=seed)

    return tokenized_dataset["train"], tokenized_dataset["test"]
