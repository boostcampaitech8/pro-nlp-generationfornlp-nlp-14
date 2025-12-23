import ast

import numpy as np
import pandas as pd


def verify_group_integrity(
    df: pd.DataFrame, fold_assignment: np.ndarray
) -> tuple[bool, pd.DataFrame]:
    """
    group_id가 여러 fold에 분산되지 않았는지 검증.
    무결성이 유지되면 True, 아니면 False 반환.
    """
    df_temp = df.copy()
    df_temp["fold"] = fold_assignment

    group_fold_counts = df_temp.groupby("group_id")["fold"].nunique()
    violations = group_fold_counts[group_fold_counts > 1]

    group_info = (
        df_temp.groupby("group_id")
        .agg({"fold": lambda x: sorted(x.unique()), "paragraph": "first", "subject": "first"})
        .reset_index()
    )
    group_info["fold_count"] = group_info["fold"].apply(len)
    group_info["sample_count"] = df_temp.groupby("group_id").size().values

    if len(violations) > 0:
        print(f"⚠️ 경고: {len(violations)}개 그룹이 여러 fold에 분산되어 있습니다!")
        return False, group_info

    return True, group_info


def print_fold_summary(
    df: pd.DataFrame, n_splits: int, eval_fold_idx: int, group_info: pd.DataFrame
) -> None:
    """Fold 분할 결과 요약 출력"""
    print(f"\n{'=' * 60}")
    print("FOLD 분할 요약")
    print(f"{'=' * 60}")

    # Fold별 샘플 분포
    print("\nFold별 샘플 분포:")
    for fold_idx in range(n_splits):
        fold_count = (df["fold"] == fold_idx).sum()
        fold_label = "평가" if fold_idx == eval_fold_idx else "학습"
        print(f"  Fold {fold_idx} ({fold_label}): {fold_count}개 샘플")

    # Fold별 과목 분포
    print("\nFold별 과목 분포:")
    subject_dist = df.groupby(["fold", "subject"]).size().unstack(fill_value=0)
    print(subject_dist)

    # Fold별 정답 분포
    df_temp = df.copy()
    df_temp["answer"] = df_temp["problems"].apply(
        lambda x: ast.literal_eval(x)["answer"] if isinstance(x, str) else x.get("answer", 0)
    )
    print("\nFold별 정답 분포:")
    answer_dist = df_temp.groupby(["fold", "answer"]).size().unstack(fill_value=0)
    print(answer_dist)

    # Fold별 지문 길이 구간 분포
    if "para_len_bin" in df.columns:
        print("\nFold별 지문 길이 구간 분포:")
        len_bin_dist = df.groupby(["fold", "para_len_bin"]).size().unstack(fill_value=0)
        print(len_bin_dist)

    # 지문 길이 통계
    if "para_len" in df.columns:
        print("\nFold별 지문 길이 통계:")
        len_stats = df.groupby("fold")["para_len"].agg(["mean", "std", "min", "max"])
        len_stats.columns = ["평균", "표준편차", "최소", "최대"]
        print(len_stats.round(1))


def print_constraint_verification(df: pd.DataFrame, eval_fold_idx: int) -> None:
    """하드 제약 조건 검증 결과 출력"""
    print(f"\n{'=' * 60}")
    print("하드 제약 검증")
    print(f"{'=' * 60}")

    geo_in_eval = ((df["subject"] == "high_school_geography") & (df["fold"] == eval_fold_idx)).sum()
    geo_total = (df["subject"] == "high_school_geography").sum()
    print(f"high_school_geography: {geo_in_eval}/{geo_total}개가 eval fold에 배치 (기대값: 3/7)")

    gov_in_eval = (
        (df["subject"] == "high_school_government_and_politics") & (df["fold"] == eval_fold_idx)
    ).sum()
    gov_total = (df["subject"] == "high_school_government_and_politics").sum()
    print(
        f"high_school_government_and_politics: {gov_in_eval}/{gov_total}개가 eval fold에 배치 (기대값: 4/9)"
    )


def print_group_leakage_info(group_info: pd.DataFrame, integrity_ok: bool) -> None:
    """그룹 누수 검증 정보 출력"""
    print(f"\n{'=' * 60}")
    print("그룹(중복 지문) 누수 검증")
    print(f"{'=' * 60}")

    multi_sample_groups = group_info[group_info["sample_count"] > 1]
    print(f"총 그룹 수: {len(group_info)}개")
    print(f"중복 지문 그룹 수: {len(multi_sample_groups)}개")
    print(f"단일 샘플 그룹 수: {len(group_info) - len(multi_sample_groups)}개")

    if len(multi_sample_groups) > 0:
        print("\n중복 지문 그룹 상위 10개:")
        top_dup = multi_sample_groups.nlargest(10, "sample_count")[
            ["group_id", "sample_count", "fold", "subject"]
        ]
        for _idx, row in top_dup.iterrows():
            print(
                f"  그룹 {row['group_id']}: {row['sample_count']}개 샘플, Fold {row['fold']}, 과목: {row['subject']}"
            )

    leaky_groups = group_info[group_info["fold_count"] > 1]
    if len(leaky_groups) > 0:
        print(f"\n⚠️ 경고: {len(leaky_groups)}개 그룹이 여러 fold에 분산됨!")
        for _idx, row in leaky_groups.iterrows():
            print(f"  그룹 {row['group_id']}: Fold {row['fold']}")
    else:
        print("\n✅ 그룹 누수 없음: 모든 중복 지문이 같은 fold에 배치됨")

    print(f"\n그룹 무결성: {'✓ 통과' if integrity_ok else '✗ 실패'}")
