import ast

import pandas as pd


def create_group_id(df: pd.DataFrame) -> pd.Series:
    """
    데이터 누수를 방지하기 위해 paragraph 기준으로 group_id 생성.
    동일한 지문은 같은 그룹에 속함.
    """
    unique_paragraphs = sorted(df["paragraph"].unique())
    paragraph_to_group = {para: idx for idx, para in enumerate(unique_paragraphs)}
    return df["paragraph"].map(paragraph_to_group)


def create_stratification_label(df: pd.DataFrame) -> pd.Series:
    """
    여러 특징을 결합하여 계층화 레이블 생성.
    과목과 정답 분포를 유지하기 위해 사용.
    """
    df_temp = df.copy()
    df_temp["answer"] = df_temp["problems"].apply(
        lambda x: ast.literal_eval(x)["answer"] if isinstance(x, str) else x.get("answer", 0)
    )
    strat_label = df_temp["subject"].astype(str) + "_" + df_temp["answer"].astype(str)
    return strat_label


def bin_paragraph_length(df: pd.DataFrame, n_bins: int = 5) -> pd.Series:
    """
    지문 길이를 분위수 기반으로 구간화하여 분포 균형 유지.
    """
    df_temp = df.copy()
    df_temp["para_len"] = df_temp["paragraph"].str.len()
    df_temp["para_len_bin"] = pd.qcut(
        df_temp["para_len"], q=n_bins, labels=False, duplicates="drop"
    )
    return df_temp["para_len_bin"]
