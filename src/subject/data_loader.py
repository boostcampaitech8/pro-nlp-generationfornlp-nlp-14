"""Subject 분류용 데이터 로더

train_source_labeled.csv에서 subject 라벨이 있는 데이터를 로드하고 전처리.
"""

import json

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def load_labeled_data(csv_path: str) -> pd.DataFrame:
    """Subject 라벨이 있는 학습 데이터 로드

    Args:
        csv_path: train_source_labeled.csv 경로

    Returns:
        정제된 DataFrame (id, paragraph, question, subject)
    """
    df = pd.read_csv(csv_path)

    records = []
    for _, row in df.iterrows():
        # problems 컬럼이 JSON 문자열인 경우 파싱
        if isinstance(row.get("problems"), str):
            try:
                problems = json.loads(row["problems"])
                question = problems.get("question", row.get("question", ""))
            except json.JSONDecodeError:
                question = row.get("question", "")
        else:
            question = row.get("question", "")

        records.append({
            "id": row["id"],
            "paragraph": row["paragraph"],
            "question": question,
            "subject": row["subject"],
        })

    return pd.DataFrame(records)


def prepare_embedding_texts(df: pd.DataFrame) -> list[str]:
    """임베딩용 텍스트 생성 (지문 + 질문 결합)

    Args:
        df: 데이터프레임

    Returns:
        결합된 텍스트 리스트
    """
    texts = []
    for _, row in df.iterrows():
        combined = f"[지문]\n{row['paragraph']}\n\n[질문]\n{row['question']}"
        texts.append(combined)
    return texts


def split_train_val(
    embeddings: np.ndarray,
    labels: np.ndarray,
    val_ratio: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """학습/검증 데이터 분리 (stratified)

    Args:
        embeddings: 임베딩 벡터 배열
        labels: 라벨 배열
        val_ratio: 검증 데이터 비율
        seed: 랜덤 시드

    Returns:
        (X_train, X_val, y_train, y_val)
    """
    return train_test_split(
        embeddings,
        labels,
        test_size=val_ratio,
        stratify=labels,
        random_state=seed,
    )
