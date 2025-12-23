import pandas as pd
from datasets import load_dataset

from data.preprocess.tagging_utils import (
    build_anchor_views,
    build_top_subject_map,
    collect_found_keys,
    make_keys,
    norm_text,
    propagate_labels,
    split_question_paragraph,
)


def tag_kmmlu(train_labeled: pd.DataFrame, train_origin: pd.DataFrame) -> pd.DataFrame:
    """
    KMMLU (Korean History) 데이터셋에서 매칭되는 항목을 태깅.

    Args:
        train_labeled: 태깅 중인 DataFrame (from, subject 컬럼 포함)
        train_origin: 원본 train DataFrame (para_norm 포함)

    Returns:
        업데이트된 train_labeled DataFrame
    """
    print("\n=== KMMLU 태깅 시작 ===")

    # 1) 데이터셋 적재 및 전처리
    ds_kmmlu = load_dataset("HAERAE-HUB/KMMLU", "Korean-History")
    kmmlu_raw = pd.concat([ds_kmmlu[s].to_pandas() for s in ["train", "dev", "test"]])

    kmmlu_proc = kmmlu_raw.copy()
    kmmlu_proc[["question", "paragraph"]] = kmmlu_proc["question"].apply(
        lambda x: pd.Series(split_question_paragraph(x))
    )
    kmmlu_proc["choices"] = kmmlu_proc[["A", "B", "C", "D"]].values.tolist()
    kmmlu_proc = kmmlu_proc[["question", "paragraph", "choices", "answer"]].reset_index(drop=True)
    kmmlu_proc["para_norm"] = kmmlu_proc["paragraph"].map(
        lambda x: norm_text(x, remove_all_space=True)
    )

    # 2) 앵커 기반 매칭 (train -> KMMLU)
    remain = train_labeled["from"].isna()
    train_key = train_origin["para_norm"].astype("string").fillna("")

    ANCHOR_LEN = 50
    train_anchor = train_key.str.slice(0, ANCHOR_LEN)
    LONG_MIN, SHORT_MIN, SHORT_MAX = 30, 8, 29

    long_keys = make_keys(
        train_anchor[remain & (train_key.str.len() >= LONG_MIN)], min_len=LONG_MIN
    )
    short_keys = make_keys(
        train_key[remain & train_key.str.len().between(SHORT_MIN, SHORT_MAX)], min_len=SHORT_MIN
    )

    found_long = collect_found_keys(kmmlu_proc["para_norm"], long_keys, chunk_size=1000)
    found_short = collect_found_keys(kmmlu_proc["para_norm"], short_keys, chunk_size=1000)

    m = remain & (train_anchor.isin(found_long) | train_key.isin(found_short))
    train_labeled.loc[m, "from"] = "KMMLU"
    train_labeled.loc[m, "subject"] = "korean_history"
    train_labeled = propagate_labels(train_labeled)

    print(f"KMMLU(anchor) 새로 라벨링됨: {int(m.sum())}")
    print(f"현재까지 총 라벨링: {int(train_labeled['from'].notna().sum())}")

    # 3) 역방향 포함 검색 (KMMLU 질문/지문 -> train 문단)
    remain_idx = train_labeled.index[train_labeled["from"].isna()]
    train_text = train_origin.loc[remain_idx, "para_norm"].astype("string").fillna("")

    MIN_KEY_LEN = 20
    km_para_keys = kmmlu_proc["para_norm"].astype("string").fillna("").drop_duplicates()
    km_para_keys = km_para_keys[km_para_keys.str.len() >= MIN_KEY_LEN].sort_values(
        key=lambda s: s.str.len(), ascending=False
    )

    km_q_norm = (
        kmmlu_proc["question"]
        .map(lambda x: norm_text(x, remove_all_space=True))
        .astype("string")
        .fillna("")
    )
    km_q_keys = km_q_norm.drop_duplicates()
    km_q_keys = km_q_keys[km_q_keys.str.len() >= MIN_KEY_LEN].sort_values(
        key=lambda s: s.str.len(), ascending=False
    )

    matched = {}
    for idx, txt in zip(remain_idx.tolist(), train_text.tolist(), strict=True):
        for k in km_para_keys:
            if k in txt:
                matched[idx] = k
                break
        if idx in matched:
            continue
        for k in km_q_keys:
            if k in txt:
                matched[idx] = k
                break

    km_mask = train_labeled.index.isin(matched.keys())
    train_labeled.loc[km_mask, "from"] = "KMMLU"
    train_labeled.loc[km_mask, "subject"] = "korean_history"
    train_labeled = propagate_labels(train_labeled)

    print(f"KMMLU(containment) 새로 라벨링됨: {int(km_mask.sum())}")
    print(f"현재까지 총 라벨링: {int(train_labeled['from'].notna().sum())}")

    return train_labeled


def tag_mmmlu(train_labeled: pd.DataFrame, train_origin: pd.DataFrame) -> pd.DataFrame:
    """
    MMMLU 데이터셋에서 매칭되는 항목을 태깅 (화이트리스트 과목만).

    Args:
        train_labeled: 태깅 중인 DataFrame (from, subject 컬럼 포함)
        train_origin: 원본 train DataFrame (para_norm, question 포함)

    Returns:
        업데이트된 train_labeled DataFrame
    """
    print("\n=== MMMLU 태깅 시작 ===")

    ds_mmmlu = load_dataset("openai/MMMLU", "KO_KR")
    mmmlu_raw = pd.concat([ds_mmmlu[s].to_pandas() for s in ds_mmmlu.keys()], ignore_index=True)

    mmmlu_proc = mmmlu_raw.copy()
    mmmlu_proc["q_norm"] = (
        mmmlu_proc["Question"]
        .map(lambda x: norm_text(x, remove_all_space=True))
        .astype("string")
        .fillna("")
    )
    mmmlu_proc["Subject"] = mmmlu_proc["Subject"].astype("string").fillna("")

    WHITE_LIST = {
        "high_school_european_history",
        "high_school_us_history",
        "high_school_world_history",
        "high_school_macroeconomics",
        "high_school_microeconomics",
        "high_school_government_and_politics",
        "high_school_geography",
        "high_school_psychology",
    }
    mmmlu_proc = mmmlu_proc[mmmlu_proc["Subject"].isin(WHITE_LIST)].copy()

    # 1) 문단 기반 앵커 매핑
    remain = train_labeled["from"].isna()
    s = train_origin["para_norm"].astype("string").fillna("")
    a_head, a_mid, a_tail = build_anchor_views(s, anchor_len=40)

    LONG_MIN, SHORT_MIN, SHORT_MAX = 20, 8, 29
    long_keys = make_keys(
        pd.concat(
            [
                a_head[remain & (s.str.len() >= LONG_MIN)],
                a_mid[remain & (s.str.len() >= LONG_MIN)],
                a_tail[remain & (s.str.len() >= LONG_MIN)],
            ],
            ignore_index=True,
        ).drop_duplicates(),
        min_len=LONG_MIN,
    )
    short_keys = make_keys(
        s[remain & s.str.len().between(SHORT_MIN, SHORT_MAX)].drop_duplicates(), min_len=SHORT_MIN
    )

    top_long = build_top_subject_map(
        mmmlu_proc, long_keys, q_col="q_norm", subj_col="Subject", chunk_size=1000
    )
    top_short = build_top_subject_map(
        mmmlu_proc, short_keys, q_col="q_norm", subj_col="Subject", chunk_size=500
    )

    m_long = remain & (a_head.isin(top_long) | a_mid.isin(top_long) | a_tail.isin(top_long))
    m_short = remain & s.isin(top_short)
    m = m_long | m_short

    subj_long = a_head.map(top_long).fillna(a_mid.map(top_long)).fillna(a_tail.map(top_long))
    subj_short = s.map(top_short)

    train_labeled.loc[m, "from"] = "MMMLU"
    train_labeled.loc[m, "subject"] = subj_long.fillna(subj_short)
    train_labeled = propagate_labels(train_labeled)

    print(f"MMMLU(paragraph) 새로 라벨링됨: {int(m.sum())}")
    print(f"현재까지 총 라벨링: {int(train_labeled['from'].notna().sum())}")

    # 2) 질문 텍스트 기반 보정
    remain = train_labeled["from"].isna()
    train_q_norm = (
        train_origin["question"]
        .map(lambda x: norm_text(x, remove_all_space=True))
        .astype("string")
        .fillna("")
    )
    train_labeled["question_norm"] = train_q_norm

    q_head, q_mid, q_tail = build_anchor_views(train_q_norm, anchor_len=40)
    LONG_MIN, SHORT_MIN, SHORT_MAX = 20, 8, 29

    q_long_keys = make_keys(
        pd.concat(
            [
                q_head[remain & (train_q_norm.str.len() >= LONG_MIN)],
                q_mid[remain & (train_q_norm.str.len() >= LONG_MIN)],
                q_tail[remain & (train_q_norm.str.len() >= LONG_MIN)],
            ],
            ignore_index=True,
        ).drop_duplicates(),
        min_len=LONG_MIN,
    )
    q_short_keys = make_keys(
        train_q_norm[
            remain & train_q_norm.str.len().between(SHORT_MIN, SHORT_MAX)
        ].drop_duplicates(),
        min_len=SHORT_MIN,
    )

    top_q_long = build_top_subject_map(
        mmmlu_proc, q_long_keys, q_col="q_norm", subj_col="Subject", chunk_size=1000
    )
    top_q_short = build_top_subject_map(
        mmmlu_proc, q_short_keys, q_col="q_norm", subj_col="Subject", chunk_size=500
    )

    m_long = remain & (q_head.isin(top_q_long) | q_mid.isin(top_q_long) | q_tail.isin(top_q_long))
    m_short = remain & train_q_norm.isin(top_q_short)
    m = m_long | m_short

    subj_long = q_head.map(top_q_long).fillna(q_mid.map(top_q_long)).fillna(q_tail.map(top_q_long))
    subj_short = train_q_norm.map(top_q_short)

    train_labeled.loc[m, "from"] = "MMMLU"
    train_labeled.loc[m, "subject"] = subj_long.fillna(subj_short)
    train_labeled = propagate_labels(train_labeled)

    print(f"MMMLU(question) 새로 라벨링됨: {int(m.sum())}")
    print(f"현재까지 총 라벨링: {int(train_labeled['from'].notna().sum())}")

    return train_labeled


def tag_klue_mrc(train_labeled: pd.DataFrame, train_origin: pd.DataFrame) -> pd.DataFrame:
    """
    KLUE-MRC 데이터셋에서 매칭되는 항목을 태깅.

    Args:
        train_labeled: 태깅 중인 DataFrame (from, subject 컬럼 포함)
        train_origin: 원본 train DataFrame (para_norm 포함)

    Returns:
        업데이트된 train_labeled DataFrame
    """
    print("\n=== KLUE-MRC 태깅 시작 ===")

    ds_klue = load_dataset("klue", "mrc")
    klue_raw = pd.concat([ds_klue[s].to_pandas() for s in ds_klue.keys()], ignore_index=True)

    ALLOWED = {"경제", "교육산업", "국제", "부동산", "사회", "생활", "책마을"}

    klue_proc = klue_raw.copy()
    klue_proc["news_category"] = (
        klue_proc["news_category"]
        .astype("string")
        .str.strip()
        .replace({"null": pd.NA, "NULL": pd.NA, "None": pd.NA, "": pd.NA})
    )
    klue_proc = klue_proc[klue_proc["news_category"].isin(ALLOWED)].copy()
    klue_proc["context_norm"] = (
        klue_proc["context"]
        .map(lambda x: norm_text(x, remove_all_space=True))
        .astype("string")
        .fillna("")
    )

    remain = train_labeled["from"].isna()
    s = train_origin["para_norm"].astype("string").fillna("")
    a_head, a_mid, a_tail = build_anchor_views(s, anchor_len=40)

    LONG_MIN, SHORT_MIN, SHORT_MAX = 20, 8, 29
    long_keys = make_keys(
        pd.concat(
            [
                a_head[remain & (s.str.len() >= LONG_MIN)],
                a_mid[remain & (s.str.len() >= LONG_MIN)],
                a_tail[remain & (s.str.len() >= LONG_MIN)],
            ],
            ignore_index=True,
        ).drop_duplicates(),
        min_len=LONG_MIN,
    )
    short_keys = make_keys(
        s[remain & s.str.len().between(SHORT_MIN, SHORT_MAX)].drop_duplicates(), min_len=SHORT_MIN
    )

    top_long = build_top_subject_map(
        klue_proc, long_keys, q_col="context_norm", subj_col="news_category", chunk_size=1000
    )
    top_short = build_top_subject_map(
        klue_proc, short_keys, q_col="context_norm", subj_col="news_category", chunk_size=500
    )

    m_long = remain & (a_head.isin(top_long) | a_mid.isin(top_long) | a_tail.isin(top_long))
    m_short = remain & s.isin(top_short)
    m = m_long | m_short

    subj_long = a_head.map(top_long).fillna(a_mid.map(top_long)).fillna(a_tail.map(top_long))
    subj_short = s.map(top_short)

    train_labeled.loc[m, "from"] = "klue-mrc"
    train_labeled.loc[m, "subject"] = subj_long.fillna(subj_short)
    train_labeled = propagate_labels(train_labeled)

    print(f"KLUE-MRC 새로 라벨링됨: {int(m.sum())}")
    print(f"현재까지 총 라벨링: {int(train_labeled['from'].notna().sum())}")

    return train_labeled
