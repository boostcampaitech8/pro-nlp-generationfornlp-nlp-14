import ast
from pathlib import Path

import pandas as pd

from data.preprocess.taggers import tag_klue_mrc, tag_kmmlu, tag_mmmlu
from data.preprocess.tagging_utils import norm_text


def add_source_labels(
    input_path: str | Path,
    output_path: str | Path,
) -> pd.DataFrame:
    """
    train.csv에서 from, subject, num_choices 컬럼을 추가하여 train_source_labeled.csv 생성.

    Args:
        input_path: 원본 train.csv 경로
        output_path: 출력 train_source_labeled.csv 경로

    Returns:
        from, subject, num_choices 컬럼이 추가된 DataFrame
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    print("=== 소스 태깅 시작 ===")

    # 1. 데이터 로드
    train = pd.read_csv(input_path)
    parsed = train["problems"].map(ast.literal_eval)
    train["question"] = parsed.map(lambda x: x["question"])
    train["answer"] = parsed.map(lambda x: x["answer"])
    train["choices"] = parsed.map(lambda x: x["choices"])
    train["paragraph"] = train["paragraph"].astype(str)

    train_origin = train.copy()
    train_origin["para_norm"] = train_origin["paragraph"].map(
        lambda x: norm_text(x, remove_all_space=True)
    )

    # 2. 라벨링을 위한 초기화
    train_labeled = train_origin.copy()
    train_labeled["from"] = pd.NA
    train_labeled["subject"] = pd.NA

    print(f"\n총 샘플 수: {len(train)}")
    print(f"빈 paragraph 수: {train_origin['paragraph'].eq('').sum()}")

    # 3. 각 데이터셋에서 태깅
    train_labeled = tag_kmmlu(train_labeled, train_origin)
    train_labeled = tag_mmmlu(train_labeled, train_origin)
    train_labeled = tag_klue_mrc(train_labeled, train_origin)

    # 4. num_choices 추가
    train_labeled["num_choices"] = train_labeled["choices"].apply(
        lambda x: len(x) if isinstance(x, list) else 0
    )

    print("\n### 선지 개수별 문항 분포 ###")
    print(train_labeled["num_choices"].value_counts().sort_index())

    # 5. 최종 결과 확인
    remaining = train_labeled["from"].isna().sum()
    print(f"\n라벨링되지 않은 샘플: {int(remaining)}")
    print(f"라벨링된 샘플: {int(train_labeled['from'].notna().sum())}")

    print("\n### from 분포 ###")
    print(train_labeled["from"].value_counts(dropna=False))

    # 6. 작업용 컬럼 제거 후 저장
    export_df = train_labeled.drop(
        columns=[
            c
            for c in ["answer", "choices", "para_norm", "question_norm"]
            if c in train_labeled.columns
        ]
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    export_df.to_csv(output_path, index=False)
    print(f"\n저장 완료: {output_path.resolve()}")

    return export_df
