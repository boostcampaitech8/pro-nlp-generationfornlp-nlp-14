import ast

import numpy as np
import pandas as pd


def manual_adjust_minority_subjects(
    df: pd.DataFrame,
    fold_assignment: np.ndarray,
    eval_fold_idx: int,
    n_splits: int,
    random_seed: int,
) -> np.ndarray:
    """
    소수 과목에 대해 하드 제약 조건을 만족하도록 fold 할당 수동 조정.

    하드 제약:
    - high_school_geography: 총 7개 중 eval fold에 3개 배치
    - high_school_government_and_politics: 총 9개 중 eval fold에 4개 배치
    """
    fold_assignment = fold_assignment.copy()

    # 마스크 선언
    geo_mask = df["subject"] == "high_school_geography"
    gov_mask = df["subject"] == "high_school_government_and_politics"

    # high_school_geography 처리
    if geo_mask.sum() > 0:
        geo_indices = df[geo_mask].index.tolist()
        np.random.seed(random_seed)
        np.random.shuffle(geo_indices)

        eval_count = 3
        for i, idx in enumerate(geo_indices):
            if i < eval_count:
                fold_assignment[idx] = eval_fold_idx
            else:
                fold_assignment[idx] = (i - eval_count) % n_splits
                if fold_assignment[idx] == eval_fold_idx:
                    fold_assignment[idx] = (fold_assignment[idx] + 1) % n_splits

    # high_school_government_and_politics 처리
    if gov_mask.sum() > 0:
        gov_indices = df[gov_mask].index.tolist()
        np.random.seed(random_seed + 1)
        np.random.shuffle(gov_indices)

        eval_count = 4
        for i, idx in enumerate(gov_indices):
            if i < eval_count:
                fold_assignment[idx] = eval_fold_idx
            else:
                fold_assignment[idx] = (i - eval_count) % n_splits
                if fold_assignment[idx] == eval_fold_idx:
                    fold_assignment[idx] = (fold_assignment[idx] + 1) % n_splits

    return fold_assignment


def balance_minority_labels(
    df: pd.DataFrame,
    fold_assignment: np.ndarray,
    target_label: int,
    n_splits: int,
    random_seed: int,
) -> np.ndarray:
    """
    소수 정답 레이블을 fold 간 균등하게 재배치.
    그룹 무결성을 유지하면서 최대한 균등하게 분산.
    """
    fold_assignment = fold_assignment.copy()

    df_temp = df.copy()
    df_temp["answer"] = df_temp["problems"].apply(
        lambda x: ast.literal_eval(x)["answer"] if isinstance(x, str) else x.get("answer", 0)
    )
    df_temp["fold"] = fold_assignment

    # 마스크 선언
    target_mask = df_temp["answer"] == target_label

    if target_mask.sum() == 0:
        return fold_assignment

    target_groups = df_temp[target_mask]["group_id"].unique()

    group_sizes = {}
    group_indices = {}
    for group_id in target_groups:
        group_mask = df_temp["group_id"] == group_id
        group_sizes[group_id] = group_mask.sum()
        group_indices[group_id] = df_temp[group_mask].index.tolist()

    sorted_groups = sorted(group_sizes.items(), key=lambda x: x[1], reverse=True)

    fold_counts = {i: 0 for i in range(n_splits)}
    for fold_idx in range(n_splits):
        fold_mask = (df_temp["fold"] == fold_idx) & target_mask
        fold_counts[fold_idx] = fold_mask.sum()

    np.random.seed(random_seed + 99)

    for group_id, _group_size in sorted_groups:
        group_target_count = (df_temp["group_id"] == group_id) & target_mask
        group_target_count = group_target_count.sum()

        min_fold = min(fold_counts.items(), key=lambda x: x[1])[0]

        for idx in group_indices[group_id]:
            fold_assignment[idx] = min_fold

        fold_counts[min_fold] += group_target_count

    return fold_assignment
