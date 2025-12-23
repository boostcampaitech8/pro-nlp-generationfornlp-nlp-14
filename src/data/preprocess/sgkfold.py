import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from data.preprocess.fold_adjusters import balance_minority_labels, manual_adjust_minority_subjects
from data.preprocess.fold_utils import (
    bin_paragraph_length,
    create_group_id,
    create_stratification_label,
)
from data.preprocess.fold_validators import (
    print_constraint_verification,
    print_fold_summary,
    print_group_leakage_info,
    verify_group_integrity,
)


def create_folds(
    df: pd.DataFrame,
    n_splits: int,
    eval_fold_idx: int,
    random_seed: int,
) -> pd.DataFrame:
    """
    하드/소프트 제약 조건을 만족하는 Stratified Group K-Fold 분할 생성.

    하드 제약 (반드시 충족):
    - 동일한 group_id는 절대 다른 fold로 분산되지 않음
    - 고정된 시드로 fold 생성하여 재현성 보장
    - high_school_geography: 7개 중 eval fold에 3개
    - high_school_government_and_politics: 9개 중 eval fold에 4개

    소프트 제약:
    - 정답 분포가 원본과 유사
    - 과목 분포가 원본과 유사
    - 지문 길이 구간 분포가 원본과 유사

    Args:
        df: 소스 라벨링이 완료된 DataFrame
        n_splits: fold 개수
        eval_fold_idx: 평가용으로 사용할 fold (0-indexed)
        random_seed: 재현성을 위한 랜덤 시드

    Returns:
        'fold' 컬럼이 추가된 DataFrame
    """
    np.random.seed(random_seed)

    print("\n=== Stratified Group K-Fold 분할 시작 ===")
    print(f"샘플 수: {len(df)}")

    # 그룹 및 계층화 레이블 생성
    df["group_id"] = create_group_id(df)
    print(f"고유 그룹 수: {df['group_id'].nunique()}개")

    df["strat_label"] = create_stratification_label(df)
    df["para_len_bin"] = bin_paragraph_length(df)
    df["para_len"] = df["paragraph"].str.len()

    # Fold 할당 초기화
    fold_assignment = np.zeros(len(df), dtype=int)

    # 소수 과목 분리
    minority_subjects = ["high_school_geography", "high_school_government_and_politics"]
    minority_mask = df["subject"].isin(minority_subjects)
    majority_df = df[~minority_mask].copy()

    # StratifiedGroupKFold 적용
    if len(majority_df) > 0:
        sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
        majority_groups = majority_df["group_id"].values
        majority_strat = majority_df["strat_label"].values

        for fold_idx, (_train_idx, val_idx) in enumerate(
            sgkf.split(majority_df, majority_strat, groups=majority_groups)
        ):
            actual_indices = majority_df.iloc[val_idx].index
            fold_assignment[actual_indices] = fold_idx

    # 소수 과목 및 정답 레이블 조정
    fold_assignment = manual_adjust_minority_subjects(
        df, fold_assignment, eval_fold_idx, n_splits, random_seed
    )
    fold_assignment = balance_minority_labels(
        df, fold_assignment, target_label=5, n_splits=n_splits, random_seed=random_seed
    )

    df["fold"] = fold_assignment

    # 검증
    integrity_ok, group_info = verify_group_integrity(df, fold_assignment)
    if not integrity_ok:
        raise ValueError("그룹 무결성 위반! 동일한 group_id가 여러 fold에서 발견됨.")

    # 결과 출력
    print_fold_summary(df, n_splits, eval_fold_idx, group_info)
    print_constraint_verification(df, eval_fold_idx)
    print_group_leakage_info(group_info, integrity_ok)

    return df


def get_train_eval_split(df: pd.DataFrame, eval_fold_idx: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    fold 할당을 기반으로 데이터프레임을 train/eval 세트로 분할.

    Args:
        df: 'fold' 컬럼이 있는 DataFrame
        eval_fold_idx: 평가용으로 사용할 fold

    Returns:
        (train_df, eval_df) 튜플
    """
    if "fold" not in df.columns:
        raise ValueError("DataFrame에 'fold' 컬럼이 필요합니다.")

    train_df = df[df["fold"] != eval_fold_idx].copy()
    eval_df = df[df["fold"] == eval_fold_idx].copy()

    return train_df, eval_df


def get_fold_split(df: pd.DataFrame, fold_idx: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    특정 fold에 대한 train/val 분할 (교차 검증용).

    Args:
        df: 'fold' 컬럼이 있는 DataFrame
        fold_idx: validation으로 사용할 fold

    Returns:
        (train_df, val_df) 튜플
    """
    if "fold" not in df.columns:
        raise ValueError("DataFrame에 'fold' 컬럼이 필요합니다.")

    train_df = df[df["fold"] != fold_idx].copy()
    val_df = df[df["fold"] == fold_idx].copy()

    return train_df, val_df
