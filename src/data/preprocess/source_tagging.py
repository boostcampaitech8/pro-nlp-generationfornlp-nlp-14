"""
데이터 소스 태깅 모듈
train.csv에서 from, subject, num_choices 컬럼을 추가하여 train_source_labeled.csv 생성
"""

import ast
import re
from pathlib import Path

import pandas as pd
from datasets import load_dataset

# 정규 표현식 패턴
_ws = re.compile(r"\s+")
_keep = re.compile(r"[^0-9A-Za-z가-힣\s]")  # 허용: 한글/영문/숫자/공백


def norm_text(x, remove_all_space: bool = True) -> str:
    """Whitespace/특수문자를 정리해 비교용 텍스트를 만든다."""
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return ""
    text = str(x)
    text = _ws.sub(" ", text)
    text = _keep.sub(" ", text)
    text = _ws.sub(" ", text).strip()
    if remove_all_space:
        result: str = re.sub(r"\s+", "", text)
        return result
    return text


def split_question_paragraph(text: str):
    """첫 번째 물음표 이전을 question, 이후를 paragraph로 분리."""
    if not isinstance(text, str):
        return "", ""
    text = text.strip()
    qmark = text.find("?")
    if qmark == -1:
        return text, ""
    return text[: qmark + 1].strip(), text[qmark + 1 :].strip()


def build_anchor_views(series: pd.Series, anchor_len: int):
    """문장을 앞/중간/끝 고정 길이 토막으로 나눠 앵커 세트를 만든다."""
    series = series.astype("string").fillna("")
    head = series.str.slice(0, anchor_len)
    tail = series.str.slice(-anchor_len, None)
    lens = series.str.len().to_numpy()
    mid_starts = (lens // 2) - (anchor_len // 2)
    mid = pd.Series(
        [
            txt[max(0, st) : max(0, st) + anchor_len]
            for txt, st in zip(series.tolist(), mid_starts, strict=True)
        ],
        index=series.index,
        dtype="string",
    )
    return head, mid, tail


def make_keys(s: pd.Series, min_len: int = 2) -> pd.Series:
    """키 후보 생성: 최소 길이 이상이고 중복 제거 후 길이 내림차순 정렬."""
    s = s.astype("string").fillna("")
    s = s[s.str.len() >= min_len].drop_duplicates()
    return s.sort_values(key=lambda x: x.str.len(), ascending=False)


def collect_found_keys(corpus: pd.Series, keys: pd.Series, chunk_size: int = 3000) -> set:
    """코퍼스에서 keys 중 등장하는 것만 모아 검색 범위를 줄인다."""
    corpus = corpus.astype("string").fillna("")
    keys_list = keys.astype("string").fillna("").tolist()
    found = set()
    for i in range(0, len(keys_list), chunk_size):
        sub = keys_list[i : i + chunk_size]
        pat = re.compile("|".join(map(re.escape, sub)))
        for lst in corpus.str.findall(pat):
            if lst:
                found.update(lst)
    return found


def build_top_subject_map(
    df: pd.DataFrame, keys: pd.Series, q_col: str, subj_col: str, chunk_size: int = 2000
) -> dict:
    """질문/지문에서 발견된 key별 최빈 주제를 계산한다."""
    mm = df[[q_col, subj_col]].copy()
    mm[q_col] = mm[q_col].astype("string").fillna("")
    mm[subj_col] = mm[subj_col].astype("string").fillna("")
    counts: dict[str, dict[str, int]] = {}
    keys_list = keys.astype("string").fillna("").tolist()
    for i in range(0, len(keys_list), chunk_size):
        sub = keys_list[i : i + chunk_size]
        pat = re.compile("|".join(map(re.escape, sub)))
        matches = mm[q_col].str.findall(pat)
        for subj, lst in zip(mm[subj_col].tolist(), matches.tolist(), strict=True):
            if lst:
                for k in lst:
                    counts.setdefault(k, {}).setdefault(subj, 0)
                    counts[k][subj] += 1
    return {k: max(subj_count.items(), key=lambda x: x[1])[0] for k, subj_count in counts.items()}


def propagate_labels(df: pd.DataFrame, norm_col: str = "para_norm") -> pd.DataFrame:
    """동일한 정규화 텍스트가 이미 라벨링된 경우 나머지 행에도 전파."""
    labeled = (
        df[df["from"].notna()][[norm_col, "from", "subject"]]
        .drop_duplicates(subset=[norm_col])
        .set_index(norm_col)
    )
    for col in ["from", "subject"]:
        missing = df[col].isna()
        df.loc[missing, col] = df.loc[missing, norm_col].map(labeled[col])
    return df


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


def add_source_labels(
    input_path: str | Path = "../data/train.csv",
    output_path: str | Path = "../data/train_source_labeled.csv",
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

    export_df.to_csv(output_path, index=False)
    print(f"\n저장 완료: {output_path.resolve()}")

    return export_df


if __name__ == "__main__":
    # 사용 예시
    from pathlib import Path

    # 경로 설정
    data_dir = Path(__file__).parent.parent.parent.parent / "data"
    input_path = data_dir / "train.csv"
    output_path = data_dir / "train_source_labeled.csv"

    # 소스 태깅 실행
    df_labeled = add_source_labels(input_path=input_path, output_path=output_path)
