import re

import pandas as pd

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
