"""
Stratified Group K-Fold 분할을 위한 데이터 전처리 모듈
"""

import ast
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

RANDOM_SEED = 42
N_SPLITS = 5
EVAL_FOLD = 4  # 0-indexed, 평가용으로 사용할 fold


def create_group_id(df: pd.DataFrame) -> pd.Series:
    """
    데이터 누수를 방지하기 위해 paragraph 기준으로 group_id 생성.
    동일한 지문은 같은 그룹에 속함.
    """
    # 각 고유 지문마다 고유한 그룹 생성 (재현성을 위해 정렬)
    unique_paragraphs = sorted(df["paragraph"].unique())
    paragraph_to_group = {para: idx for idx, para in enumerate(unique_paragraphs)}
    return df["paragraph"].map(paragraph_to_group)


def create_stratification_label(df: pd.DataFrame) -> pd.Series:
    """
    여러 특징을 결합하여 계층화 레이블 생성.
    과목과 정답 분포를 유지하기 위해 사용.
    """
    # 과목과 정답을 결합하여 계층화
    # 과목 및 정답 분포가 모두 보존되도록 보장
    df_temp = df.copy()

    # problems 컬럼에서 answer 파싱
    df_temp["answer"] = df_temp["problems"].apply(
        lambda x: ast.literal_eval(x)["answer"] if isinstance(x, str) else x.get("answer", 0)
    )

    # "과목_정답" 형태로 계층화 레이블 생성
    strat_label = df_temp["subject"].astype(str) + "_" + df_temp["answer"].astype(str)

    return strat_label


def bin_paragraph_length(df: pd.DataFrame, n_bins: int = 5) -> pd.Series:
    """
    지문 길이를 분위수 기반으로 구간화하여 분포 균형 유지.
    """
    df_temp = df.copy()
    df_temp["para_len"] = df_temp["paragraph"].str.len()

    # 분위수 기반 구간 생성
    df_temp["para_len_bin"] = pd.qcut(
        df_temp["para_len"], q=n_bins, labels=False, duplicates="drop"
    )

    return df_temp["para_len_bin"]


def manual_adjust_minority_subjects(
    df: pd.DataFrame, fold_assignment: np.ndarray, eval_fold_idx: int = EVAL_FOLD
) -> np.ndarray:
    """
    소수 과목에 대해 하드 제약 조건을 만족하도록 fold 할당 수동 조정.

    하드 제약:
    - high_school_geography: 총 7개 중 eval fold에 3개 배치
    - high_school_government_and_politics: 총 9개 중 eval fold에 4개 배치
    """
    fold_assignment = fold_assignment.copy()

    # high_school_geography 처리 (총 7개 -> eval에 3개)
    geo_mask = df["subject"] == "high_school_geography"
    if geo_mask.sum() > 0:
        geo_indices = df[geo_mask].index.tolist()
        np.random.seed(RANDOM_SEED)
        np.random.shuffle(geo_indices)

        # eval fold에 3개 할당, 나머지 4개는 다른 fold에 분산
        eval_count = 3
        for i, idx in enumerate(geo_indices):
            if i < eval_count:
                fold_assignment[idx] = eval_fold_idx
            else:
                # 나머지를 다른 fold에 분산
                fold_assignment[idx] = (i - eval_count) % N_SPLITS
                if fold_assignment[idx] == eval_fold_idx:
                    fold_assignment[idx] = (fold_assignment[idx] + 1) % N_SPLITS

    # high_school_government_and_politics 처리 (총 9개 -> eval에 4개)
    gov_mask = df["subject"] == "high_school_government_and_politics"
    if gov_mask.sum() > 0:
        gov_indices = df[gov_mask].index.tolist()
        np.random.seed(RANDOM_SEED + 1)
        np.random.shuffle(gov_indices)

        # eval fold에 4개 할당, 나머지 5개는 다른 fold에 분산
        eval_count = 4
        for i, idx in enumerate(gov_indices):
            if i < eval_count:
                fold_assignment[idx] = eval_fold_idx
            else:
                # 나머지를 다른 fold에 분산
                fold_assignment[idx] = (i - eval_count) % N_SPLITS
                if fold_assignment[idx] == eval_fold_idx:
                    fold_assignment[idx] = (fold_assignment[idx] + 1) % N_SPLITS

    return fold_assignment


def balance_minority_labels(
    df: pd.DataFrame,
    fold_assignment: np.ndarray,
    target_label: int = 5,
    n_splits: int = N_SPLITS,
    random_seed: int = RANDOM_SEED,
) -> np.ndarray:
    """
    소수 정답 레이블(예: 5번)을 fold 간 균등하게 재배치.
    그룹 무결성을 유지하면서 최대한 균등하게 분산.

    Args:
        df: 원본 데이터프레임
        fold_assignment: 현재 fold 할당
        target_label: 균등 배치할 정답 번호
        n_splits: fold 개수
        random_seed: 랜덤 시드

    Returns:
        조정된 fold 할당 배열
    """
    fold_assignment = fold_assignment.copy()

    # 해당 레이블을 가진 샘플 찾기
    df_temp = df.copy()
    df_temp["answer"] = df_temp["problems"].apply(
        lambda x: ast.literal_eval(x)["answer"] if isinstance(x, str) else x.get("answer", 0)
    )
    df_temp["fold"] = fold_assignment

    target_mask = df_temp["answer"] == target_label

    if target_mask.sum() == 0:
        return fold_assignment

    # 해당 레이블을 가진 그룹들 추출
    target_groups = df_temp[target_mask]["group_id"].unique()

    # 각 그룹의 샘플 수 계산
    group_sizes = {}
    group_indices = {}
    for group_id in target_groups:
        group_mask = df_temp["group_id"] == group_id
        group_sizes[group_id] = group_mask.sum()
        group_indices[group_id] = df_temp[group_mask].index.tolist()

    # 그룹을 크기 순으로 정렬 (큰 그룹부터 배치)
    sorted_groups = sorted(group_sizes.items(), key=lambda x: x[1], reverse=True)

    # 각 fold의 현재 타겟 레이블 수 계산
    fold_counts = {i: 0 for i in range(n_splits)}
    for fold_idx in range(n_splits):
        fold_mask = (df_temp["fold"] == fold_idx) & target_mask
        fold_counts[fold_idx] = fold_mask.sum()

    # 그룹을 재배치
    np.random.seed(random_seed + 99)  # 다른 시드 사용

    for group_id, _group_size in sorted_groups:
        # 현재 이 그룹의 타겟 레이블 샘플 수
        group_target_count = (df_temp["group_id"] == group_id) & target_mask
        group_target_count = group_target_count.sum()

        # 가장 적게 가진 fold에 배치
        min_fold = min(fold_counts.items(), key=lambda x: x[1])[0]

        # 해당 그룹의 모든 샘플을 min_fold로 이동
        for idx in group_indices[group_id]:
            fold_assignment[idx] = min_fold

        # fold 카운트 업데이트
        fold_counts[min_fold] += group_target_count

    return fold_assignment


def verify_group_integrity(
    df: pd.DataFrame, fold_assignment: np.ndarray
) -> tuple[bool, pd.DataFrame]:
    """
    group_id가 여러 fold에 분산되지 않았는지 검증.
    무결성이 유지되면 True, 아니면 False 반환.
    """
    df_temp = df.copy()
    df_temp["fold"] = fold_assignment

    # 각 그룹이 여러 fold에 나타나는지 확인
    group_fold_counts = df_temp.groupby("group_id")["fold"].nunique()
    violations = group_fold_counts[group_fold_counts > 1]

    # 그룹별 상세 정보
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


def create_folds(
    data_path: str = "../data/train_source_labeled.csv",
    n_splits: int = N_SPLITS,
    eval_fold_idx: int = EVAL_FOLD,
    random_seed: int = RANDOM_SEED,
    output_path: str | None = None,
) -> pd.DataFrame:
    """
    하드/소프트 제약 조건을 만족하는 Stratified Group K-Fold 분할 생성.

    하드 제약 (반드시 충족):
    - 동일한 group_id는 절대 다른 fold로 분산되지 않음
    - 고정된 시드로 5개 fold 생성하여 재현성 보장
    - high_school_geography: 7개 중 eval fold에 3개
    - high_school_government_and_politics: 9개 중 eval fold에 4개

    소프트 제약 (최선 노력):
    - 정답 분포가 원본과 유사
    - 과목 분포가 원본과 유사 (geography/politics 강제 배치 제외)
    - 지문 길이 구간 분포가 원본과 유사

    Args:
        data_path: train_source_labeled.csv 경로
        n_splits: fold 개수 (기본값: 5)
        eval_fold_idx: 평가용으로 사용할 fold (기본값: 4, 0-indexed)
        random_seed: 재현성을 위한 랜덤 시드
        output_path: fold가 할당된 데이터프레임을 저장할 경로

    Returns:
        'fold' 컬럼이 추가된 DataFrame
    """
    # 재현성을 위한 시드 고정
    np.random.seed(random_seed)

    # 데이터 로드
    df = pd.read_csv(data_path)
    print(f"{data_path}에서 {len(df)}개 샘플 로드")

    # group_id 생성 (하드 제약: 동일 지문 = 동일 그룹)
    df["group_id"] = create_group_id(df)
    print(f"{df['group_id'].nunique()}개의 고유 그룹 생성")

    # 계층화 레이블 생성 (소프트 제약)
    df["strat_label"] = create_stratification_label(df)

    # 지문 길이 구간 추가 (소프트 제약)
    df["para_len_bin"] = bin_paragraph_length(df)
    df["para_len"] = df["paragraph"].str.len()

    # fold 할당 초기화
    fold_assignment = np.zeros(len(df), dtype=int)

    # 수동 처리가 필요한 소수 과목 분리
    minority_subjects = ["high_school_geography", "high_school_government_and_politics"]
    minority_mask = df["subject"].isin(minority_subjects)

    # StratifiedGroupKFold를 적용할 다수 데이터
    majority_df = df[~minority_mask].copy()

    if len(majority_df) > 0:
        # 다수 데이터에 StratifiedGroupKFold 적용
        sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_seed)

        # 다수 데이터에 대해 분할 수행
        majority_groups = majority_df["group_id"].values
        majority_strat = majority_df["strat_label"].values

        for fold_idx, (_train_idx, val_idx) in enumerate(
            sgkf.split(majority_df, majority_strat, groups=majority_groups)
        ):
            actual_indices = majority_df.iloc[val_idx].index
            fold_assignment[actual_indices] = fold_idx

    # 소수 과목 수동 조정 (하드 제약)
    fold_assignment = manual_adjust_minority_subjects(df, fold_assignment, eval_fold_idx)

    # 소수 정답 레이블 균등 배치 (5번 정답)
    fold_assignment = balance_minority_labels(
        df, fold_assignment, target_label=5, n_splits=n_splits, random_seed=random_seed
    )

    # 데이터프레임에 fold 할당 추가
    df["fold"] = fold_assignment

    # 그룹 무결성 검증 (하드 제약 확인)
    integrity_ok, group_info = verify_group_integrity(df, fold_assignment)
    if not integrity_ok:
        raise ValueError("그룹 무결성 위반! 동일한 group_id가 여러 fold에서 발견됨.")

    print(f"\n{'=' * 60}")
    print("FOLD 분할 요약")
    print(f"{'=' * 60}")

    # fold 분포 출력
    print("\nFold별 샘플 분포:")
    for fold_idx in range(n_splits):
        fold_count = (df["fold"] == fold_idx).sum()
        fold_label = "평가" if fold_idx == eval_fold_idx else "학습"
        print(f"  Fold {fold_idx} ({fold_label}): {fold_count}개 샘플")

    # fold별 과목 분포 출력
    print("\nFold별 과목 분포:")
    subject_dist = df.groupby(["fold", "subject"]).size().unstack(fill_value=0)
    print(subject_dist)

    # fold별 정답 분포 출력
    df["answer"] = df["problems"].apply(
        lambda x: ast.literal_eval(x)["answer"] if isinstance(x, str) else x.get("answer", 0)
    )
    print("\nFold별 정답 분포:")
    answer_dist = df.groupby(["fold", "answer"]).size().unstack(fill_value=0)
    print(answer_dist)

    # fold별 지문 길이 구간 분포 출력
    print("\nFold별 지문 길이 구간 분포:")
    len_bin_dist = df.groupby(["fold", "para_len_bin"]).size().unstack(fill_value=0)
    print(len_bin_dist)

    # 지문 길이 통계
    print("\nFold별 지문 길이 통계:")
    len_stats = df.groupby("fold")["para_len"].agg(["mean", "std", "min", "max"])
    len_stats.columns = ["평균", "표준편차", "최소", "최대"]
    print(len_stats.round(1))

    # 소수 과목 제약 검증
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

    # 그룹 누수 검증
    print(f"\n{'=' * 60}")
    print("그룹(중복 지문) 누수 검증")
    print(f"{'=' * 60}")

    # 그룹별 통계
    multi_sample_groups = group_info[group_info["sample_count"] > 1]
    print(f"총 그룹 수: {len(group_info)}개")
    print(f"중복 지문 그룹 수: {len(multi_sample_groups)}개")
    print(f"단일 샘플 그룹 수: {len(group_info) - len(multi_sample_groups)}개")

    # 중복 지문 그룹 상세 정보
    if len(multi_sample_groups) > 0:
        print("\n중복 지문 그룹 상위 10개:")
        top_dup = multi_sample_groups.nlargest(10, "sample_count")[
            ["group_id", "sample_count", "fold", "subject"]
        ]
        for _idx, row in top_dup.iterrows():
            print(
                f"  그룹 {row['group_id']}: {row['sample_count']}개 샘플, Fold {row['fold']}, 과목: {row['subject']}"
            )

    # fold 간 누수 확인
    leaky_groups = group_info[group_info["fold_count"] > 1]
    if len(leaky_groups) > 0:
        print(f"\n⚠️ 경고: {len(leaky_groups)}개 그룹이 여러 fold에 분산됨!")
        for _idx, row in leaky_groups.iterrows():
            print(f"  그룹 {row['group_id']}: Fold {row['fold']}")
    else:
        print("\n✅ 그룹 누수 없음: 모든 중복 지문이 같은 fold에 배치됨")

    print(f"\n그룹 무결성: {'✓ 통과' if integrity_ok else '✗ 실패'}")

    # 저장
    if output_path:
        # 작업용 컬럼 제거
        export_df = df.drop(
            columns=["strat_label", "para_len_bin", "para_len", "answer"], errors="ignore"
        )
        export_df.to_csv(output_path, index=False)
        print(f"\nFold 할당 결과 저장: {output_path}")

    return df


def get_train_eval_split(
    df: pd.DataFrame, eval_fold_idx: int = EVAL_FOLD
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    fold 할당을 기반으로 데이터프레임을 train/eval 세트로 분할.

    Args:
        df: 'fold' 컬럼이 있는 DataFrame
        eval_fold_idx: 평가용으로 사용할 fold

    Returns:
        (train_df, eval_df) 튜플
    """
    if "fold" not in df.columns:
        raise ValueError("DataFrame에 'fold' 컬럼이 필요합니다. create_folds()를 먼저 실행하세요.")

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
        raise ValueError("DataFrame에 'fold' 컬럼이 필요합니다. create_folds()를 먼저 실행하세요.")

    train_df = df[df["fold"] != fold_idx].copy()
    val_df = df[df["fold"] == fold_idx].copy()

    return train_df, val_df


if __name__ == "__main__":
    # 사용 예시
    from pathlib import Path

    # 경로 설정
    data_dir = Path(__file__).parent.parent.parent / "data"
    input_path = data_dir / "train_source_labeled.csv"
    output_path = data_dir / "train_with_folds.csv"

    # Fold 생성
    df_with_folds = create_folds(
        data_path=str(input_path),
        n_splits=N_SPLITS,
        eval_fold_idx=EVAL_FOLD,
        random_seed=RANDOM_SEED,
        output_path=str(output_path),
    )

    # 예시: train/eval 분할
    train_df, eval_df = get_train_eval_split(df_with_folds, eval_fold_idx=EVAL_FOLD)
    print(f"\n최종 학습 데이터 크기: {len(train_df)}")
    print(f"최종 평가 데이터 크기: {len(eval_df)}")
