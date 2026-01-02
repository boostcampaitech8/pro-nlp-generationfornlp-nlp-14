import random
from ast import literal_eval

import numpy as np
import pandas as pd
from datasets import Dataset


def set_seed(random_seed: int = 42):
    import torch

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
            "subject": row.get("subject", "general"),
        }
        records.append(record)

    return pd.DataFrame(records)


def balance_by_answer(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """정답 분포를 균일하게 맞춤 (선택지 순서 변경)

    4지선다와 5지선다를 분리하여 각각 정답 분포를 균일하게 조정.
    선택지의 순서를 변경하여 정답 번호가 균일하게 분포되도록 조정.
    데이터 손실 없이 모든 샘플을 활용.

    Args:
        df: 원본 데이터프레임 (answer, choices 컬럼 필요)
        seed: 랜덤 시드

    Returns:
        균형 잡힌 데이터프레임
    """
    random.seed(seed)
    np.random.seed(seed)

    # 선택지 개수 컬럼 추가
    df = df.copy()
    df["num_choices"] = df["choices"].apply(len)

    print(f"[Balance] 전체 샘플 수: {len(df)}")

    result_dfs = []

    # 4지선다와 5지선다 분리 처리
    for num_choices in sorted(df["num_choices"].unique()):
        subset = df[df["num_choices"] == num_choices].copy()
        print(f"\n[Balance] === {num_choices}지선다 ({len(subset)}개) ===")

        # 원본 정답 분포 확인
        original_counts = subset["answer"].value_counts().sort_index()
        print(f"[Balance] 원본 정답 분포: {original_counts.to_dict()}")

        # 해당 선택지 개수에 맞는 정답 범위 (1 ~ num_choices)
        valid_answers = list(range(1, num_choices + 1))
        target_count = len(subset) // num_choices

        print(f"[Balance] 목표 분포: 각 정답당 약 {target_count}개")

        # 결과 저장
        result_records = []

        # 현재 각 정답별 할당된 개수 추적
        answer_allocated = {ans: 0 for ans in valid_answers}

        # 데이터 셔플 (순서에 따른 편향 방지)
        subset_shuffled = subset.sample(frac=1, random_state=seed).reset_index(drop=True)

        for _, row in subset_shuffled.iterrows():
            choices = row["choices"]
            original_answer = row["answer"]

            # 목표 정답 결정 (가장 부족한 정답 번호로)
            target_answer = min(valid_answers, key=lambda x: answer_allocated.get(x, 0))

            if target_answer == original_answer:
                # 순서 변경 불필요
                new_choices = choices
                new_answer = original_answer
            else:
                # 선택지 순서 변경하여 정답 위치 조정
                correct_choice = choices[original_answer - 1]  # 1-indexed

                # 새 선택지 리스트 생성
                new_choices = choices.copy()
                new_choices.pop(original_answer - 1)
                new_choices.insert(target_answer - 1, correct_choice)
                new_answer = target_answer

            # 레코드 생성
            new_record = row.to_dict()
            new_record["choices"] = new_choices
            new_record["answer"] = new_answer
            result_records.append(new_record)

            answer_allocated[new_answer] = answer_allocated.get(new_answer, 0) + 1

        subset_result = pd.DataFrame(result_records)

        # 결과 분포 출력
        final_counts = subset_result["answer"].value_counts().sort_index()
        print(f"[Balance] 균형 후 정답 분포: {final_counts.to_dict()}")

        result_dfs.append(subset_result)

    # 모든 결과 합치기
    result_df = pd.concat(result_dfs, ignore_index=True)

    # num_choices 컬럼 제거
    result_df = result_df.drop(columns=["num_choices"])

    # 최종 셔플
    result_df = result_df.sample(frac=1, random_state=seed).reset_index(drop=True)

    print(f"\n[Balance] 최종 총 샘플 수: {len(result_df)} (변경 없음)")

    return result_df


def create_prompt_messages(df: pd.DataFrame, prompt_manager) -> list[dict]:
    """프롬프트 메시지 생성"""
    dataset = Dataset.from_pandas(df)
    processed_dataset = []

    for i in range(len(dataset)):
        user_message = prompt_manager.make_user_prompt(dataset[i])
        system_message = prompt_manager.make_system_prompt(dataset[i])

        processed_dataset.append(
            {
                "id": dataset[i]["id"],
                "messages": [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message},
                    {"role": "assistant", "content": f"{dataset[i]['answer']}"},
                ],
                "label": dataset[i]["answer"],
            }
        )

    return processed_dataset


def create_test_prompt_messages(df: pd.DataFrame, prompt_manager) -> list[dict]:
    """테스트용 프롬프트 메시지 생성 (정답 없음)"""
    test_dataset = []

    for _, row in df.iterrows():
        len_choices = len(row["choices"])
        user_message = prompt_manager.make_user_prompt(row)
        system_message = prompt_manager.make_system_prompt(row)

        test_dataset.append(
            {
                "id": row["id"],
                "messages": [
                    {"role": "system", "content": system_message},
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
):
    """토큰화"""

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

    return tokenized_dataset
