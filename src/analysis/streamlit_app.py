import ast
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st


def get_csv_files(directory: str) -> list[str]:
    """지정된 디렉토리에서 CSV 파일 목록을 반환한다."""
    dir_path = Path(directory)
    if not dir_path.exists():
        return []
    csv_files = sorted(dir_path.glob("*.csv"))
    return [str(f) for f in csv_files]


@st.cache_data
def load_data(data_path: str, output_path: str):
    """Load and cache data from CSV files."""
    try:
        train_df = pd.read_csv(data_path)
        output_df = pd.read_csv(output_path)
        return train_df, output_df, None
    except Exception as e:
        return None, None, str(e)


def parse_problem(problem_str: str) -> dict:
    """Parse the problem string into a dictionary."""
    try:
        return ast.literal_eval(problem_str)
    except (ValueError, SyntaxError):
        return {}


def preprocess_data(train_df: pd.DataFrame, output_df: pd.DataFrame) -> pd.DataFrame:
    """Merge and preprocess the dataframes."""
    merged_df = pd.merge(train_df, output_df, on="id", suffixes=("_source", "_pred"))

    if "problems" in merged_df.columns:
        parsed_problems = merged_df["problems"].apply(parse_problem)
        merged_df["question"] = parsed_problems.apply(lambda x: x.get("question"))
        merged_df["choices"] = parsed_problems.apply(lambda x: x.get("choices"))
        merged_df["correct_answer"] = parsed_problems.apply(lambda x: x.get("answer"))

        def check_correct(row):
            try:
                # 모델 예측값은 answer_pred 컬럼에 있음
                pred_answer = (
                    row.get("answer_pred")
                    if "answer_pred" in row
                    else row.get("answer")
                )
                return int(row["correct_answer"]) == int(pred_answer)
            except (ValueError, TypeError):
                return False

        merged_df["is_correct"] = merged_df.apply(check_correct, axis=1)

        # 예측값을 answer 컬럼으로 통일 (나머지 코드 호환성 유지)
        if "answer_pred" in merged_df.columns:
            merged_df["answer"] = merged_df["answer_pred"]

        # Calculate input length
        def calc_len(row):
            p_len = len(str(row.get("paragraph", "")))
            q_len = len(str(row.get("question", "")))
            c_len = sum(len(str(c)) for c in row.get("choices", []))
            return p_len + q_len + c_len

        merged_df["input_length"] = merged_df.apply(calc_len, axis=1)

    return merged_df


def main():
    st.set_page_config(layout="wide", page_title="모델 오답 분석")
    st.title("🎯 모델 오답 분석 대시보드")

    # ==========================================
    # 사이드바: 설정 및 필터
    # ==========================================
    st.sidebar.header("⚙️ 설정 (Configuration)")

    # data/ 디렉토리의 CSV 파일 목록 가져오기
    data_files = get_csv_files("data/fold")
    if data_files:
        data_path = st.sidebar.selectbox(
            "데이터 경로 (CSV)",
            options=data_files,
            index=0,
        )
    else:
        st.sidebar.warning("data/fold/ 디렉토리에 CSV 파일이 없습니다.")
        data_path = st.sidebar.text_input(
            "데이터 경로 (CSV)", "data/fold/train_with_folds.csv"
        )

    # outputs/ 디렉토리의 CSV 파일 목록 가져오기
    output_files = get_csv_files("outputs")
    if output_files:
        output_path = st.sidebar.selectbox(
            "모델 1 출력 경로 (CSV)",
            options=output_files,
            index=0,
        )
    else:
        st.sidebar.warning("outputs/ 디렉토리에 CSV 파일이 없습니다.")
        output_path = st.sidebar.text_input(
            "모델 1 출력 경로 (CSV)", "outputs/output.csv"
        )

    # Multi-model comparison
    with st.sidebar.expander("🔄 모델 비교 (선택사항)"):
        enable_comparison = st.checkbox("다른 모델과 비교")
        if enable_comparison:
            if output_files:
                output_path_2 = st.selectbox(
                    "모델 2 출력 경로 (CSV)",
                    options=output_files,
                    index=min(
                        1, len(output_files) - 1
                    ),  # 두 번째 파일 또는 첫 번째 파일
                    key="output_path_2",
                )
            else:
                output_path_2 = st.text_input(
                    "모델 2 출력 경로 (CSV)", "outputs/model2.csv"
                )
        else:
            output_path_2 = None

    if st.sidebar.button("데이터 로드 (Load Data)", type="primary"):
        st.session_state["load_data"] = True

    if not st.session_state.get("load_data", False):
        st.info("👈 경로를 확인하고 '데이터 로드' 버튼을 눌러주세요.")
        return

    # Load data with caching
    with st.spinner("데이터 로딩 중..."):
        train_df, output_df, error = load_data(data_path, output_path)

    if error:
        st.error(f"파일 로드 오류: {error}")
        return

    if train_df is None or output_df is None:
        st.error("데이터를 로드할 수 없습니다.")
        return

    # Merge and preprocess
    try:
        merged_df = preprocess_data(train_df, output_df)
    except KeyError as e:
        st.error(f"병합 실패: {e}. 두 CSV 파일 모두 'id' 컬럼이 있어야 합니다.")
        return
    except Exception as e:
        st.error(f"전처리 오류: {e}")
        return

    if "is_correct" not in merged_df.columns:
        st.error("데이터 파일에 'problems' 컬럼이 없거나 파싱에 실패했습니다.")
        return

    # Load second model for comparison if enabled
    merged_df_2 = None
    if enable_comparison and output_path_2:
        try:
            output_df_2 = pd.read_csv(output_path_2)
            merged_df_2 = pd.merge(
                train_df, output_df_2, on="id", suffixes=("", "_pred2")
            )
            if "problems" in merged_df_2.columns:
                parsed_problems_2 = merged_df_2["problems"].apply(parse_problem)
                merged_df_2["correct_answer"] = parsed_problems_2.apply(
                    lambda x: x.get("answer")
                )
                merged_df_2["is_correct_2"] = merged_df_2.apply(
                    lambda row: int(row["correct_answer"]) == int(row["answer"])
                    if pd.notna(row["correct_answer"]) and pd.notna(row["answer"])
                    else False,
                    axis=1,
                )
        except Exception as e:
            st.sidebar.warning(f"모델 2 로드 실패: {e}")
            merged_df_2 = None

    # ------------------------------------------
    # 동적 필터 생성 (사이드바)
    # ------------------------------------------
    st.sidebar.header("🔍 필터 (Filters)")

    ignore_cols = [
        "id",
        "paragraph",
        "problems",
        "question_plus",
        "question",
        "choices",
        "correct_answer",
        "answer",
        "is_correct",
        "input_length",
    ]
    potential_cats = [
        col
        for col in merged_df.columns
        if col not in ignore_cols and merged_df[col].nunique() < 50
    ]

    active_filters = {}
    if potential_cats:
        for col in potential_cats:
            options = sorted(merged_df[col].unique().tolist())

            with st.sidebar.expander(f"📁 {col}", expanded=False):
                # 세션 상태 초기화
                if f"filter_{col}" not in st.session_state:
                    st.session_state[f"filter_{col}"] = {opt: True for opt in options}

                # 체크박스 생성
                selected = []
                for opt in options:
                    is_checked = st.checkbox(
                        str(opt),
                        value=st.session_state[f"filter_{col}"].get(opt, True),
                        key=f"cb_{col}_{opt}",
                    )
                    st.session_state[f"filter_{col}"][opt] = is_checked
                    if is_checked:
                        selected.append(opt)

                active_filters[col] = selected

    # 필터 적용
    filtered_df = merged_df.copy()
    for col, selected in active_filters.items():
        filtered_df = filtered_df[filtered_df[col].isin(selected)]

    # 탭 구성
    if enable_comparison and merged_df_2 is not None:
        tab_comprehensive, tab_error_analysis, tab_comparison = st.tabs(
            ["📊 종합 분석", "❌ 모델 오답 분석", "🔄 모델 비교"]
        )
    else:
        tab_comprehensive, tab_error_analysis = st.tabs(
            ["📊 종합 분석", "❌ 모델 오답 분석"]
        )

    # ==========================================
    # 탭 1: 종합 분석
    # ==========================================
    with tab_comprehensive:
        st.header("종합 지표")

        total_count = len(filtered_df)
        correct_count = filtered_df["is_correct"].sum()
        error_count = total_count - correct_count
        accuracy = (correct_count / total_count) * 100 if total_count > 0 else 0

        # Calculate Macro F1
        def calculate_macro_f1(df):
            if "correct_answer" not in df.columns or "answer" not in df.columns:
                return 0.0, {}

            classes = sorted(
                set(df["correct_answer"].unique()) | set(df["answer"].unique())
            )
            class_metrics = {}

            for cls in classes:
                tp = len(df[(df["correct_answer"] == cls) & (df["answer"] == cls)])
                fp = len(df[(df["correct_answer"] != cls) & (df["answer"] == cls)])
                fn = len(df[(df["correct_answer"] == cls) & (df["answer"] != cls)])

                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = (
                    2 * precision * recall / (precision + recall)
                    if (precision + recall) > 0
                    else 0
                )

                class_metrics[cls] = {
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "support": tp + fn,
                }

            macro_f1 = (
                sum(m["f1"] for m in class_metrics.values()) / len(classes)
                if classes
                else 0
            )
            return macro_f1, class_metrics

        macro_f1, class_metrics = calculate_macro_f1(filtered_df)

        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("📝 선택된 문항 수", f"{total_count}개")
        col2.metric("✅ 정답 개수", f"{correct_count}개")
        col3.metric("❌ 오답 개수", f"{error_count}개")
        col4.metric("🎯 정확도", f"{accuracy:.2f}%")
        col5.metric("📊 Macro F1", f"{macro_f1:.4f}")

        # Per-class metrics table (combined with answer distribution)
        st.subheader("📈 선택지별 분포 및 성능 지표")
        if class_metrics and "correct_answer" in filtered_df.columns:
            ans_counts = filtered_df["correct_answer"].value_counts().sort_index()
            ans_acc = filtered_df.groupby("correct_answer")["is_correct"].mean() * 100

            # Build combined table
            all_classes = sorted(set(ans_counts.index) | set(class_metrics.keys()))

            combined_table = pd.DataFrame(
                [
                    {
                        "선택지": cls,
                        "문항 수": int(ans_counts.get(cls, 0)),
                        "비율 (%)": f"{ans_counts.get(cls, 0) / ans_counts.sum() * 100:.1f}%"
                        if ans_counts.sum() > 0
                        else "0.0%",
                        "정답률 (%)": f"{ans_acc.get(cls, 0):.1f}%",
                        "Precision": f"{class_metrics.get(cls, {}).get('precision', 0):.3f}",
                        "Recall": f"{class_metrics.get(cls, {}).get('recall', 0):.3f}",
                        "F1-Score": f"{class_metrics.get(cls, {}).get('f1', 0):.3f}",
                    }
                    for cls in all_classes
                ]
            )
            st.dataframe(combined_table, hide_index=True, use_container_width=True)
        else:
            st.warning("성능 지표를 계산할 수 없습니다.")

        st.divider()

        # ------------------------------------------
        # Position Bias Analysis (위치 편향 분석)
        # ------------------------------------------
        st.subheader("🎲 Position Bias Analysis (위치 편향 분석)")
        st.caption(
            "모델이 특정 번호를 선호하는지 확인합니다. 이상적으로는 예측 분포가 정답 분포와 유사해야 합니다."
        )

        if "correct_answer" in filtered_df.columns and "answer" in filtered_df.columns:
            pred_counts = filtered_df["answer"].value_counts().sort_index()
            true_counts = filtered_df["correct_answer"].value_counts().sort_index()

            # Ensure both series have the same index
            all_answers = sorted(set(pred_counts.index) | set(true_counts.index))
            pred_counts = pred_counts.reindex(all_answers, fill_value=0)
            true_counts = true_counts.reindex(all_answers, fill_value=0)

            # Grouped bar chart using Altair for direct comparison
            st.markdown("**📊 모델 예측 vs 실제 정답 분포 비교**")

            bias_chart_data = pd.DataFrame(
                {
                    "번호": list(all_answers) * 2,
                    "개수": list(pred_counts.values) + list(true_counts.values),
                    "유형": ["모델 예측"] * len(all_answers)
                    + ["실제 정답"] * len(all_answers),
                }
            )

            grouped_chart = (
                alt.Chart(bias_chart_data)
                .mark_bar()
                .encode(
                    x=alt.X("번호:O", title="선택지 번호"),
                    y=alt.Y("개수:Q", title="문항 수"),
                    color=alt.Color(
                        "유형:N",
                        scale=alt.Scale(
                            domain=["모델 예측", "실제 정답"],
                            range=["#f97316", "#3b82f6"],
                        ),
                    ),
                    xOffset="유형:N",
                    tooltip=["번호", "유형", "개수"],
                )
            )

            st.altair_chart(grouped_chart, use_container_width=True)

            # Calculate and display bias metrics
            pred_pct = pred_counts / pred_counts.sum() * 100
            true_pct = true_counts / true_counts.sum() * 100
            bias_diff = pred_pct - true_pct

            st.markdown("**편향 분석 (예측 비율 - 정답 비율)**")
            bias_display = pd.DataFrame(
                {
                    "번호": all_answers,
                    "예측 비율 (%)": [f"{v:.1f}%" for v in pred_pct.values],
                    "정답 비율 (%)": [f"{v:.1f}%" for v in true_pct.values],
                    "편향 (%)": [f"{v:+.1f}%" for v in bias_diff.values],
                }
            )
            st.dataframe(bias_display, hide_index=True)
        else:
            st.warning("위치 편향 분석을 위한 데이터가 없습니다.")

        st.divider()

        # ------------------------------------------
        # Confusion Matrix (혼동 행렬)
        # ------------------------------------------
        st.subheader("🔢 Confusion Matrix (혼동 행렬)")
        st.caption("모델 예측과 실제 정답 간의 관계를 보여줍니다. 대각선이 정답입니다.")

        if "correct_answer" in filtered_df.columns and "answer" in filtered_df.columns:
            confusion_data = (
                filtered_df.groupby(["correct_answer", "answer"])
                .size()
                .reset_index(name="count")
            )

            heatmap = (
                alt.Chart(confusion_data)
                .mark_rect()
                .encode(
                    x=alt.X("answer:O", title="예측 (Predicted)"),
                    y=alt.Y(
                        "correct_answer:O", title="정답 (Actual)", sort="ascending"
                    ),
                    color=alt.Color(
                        "count:Q", scale=alt.Scale(scheme="oranges"), title="문항 수"
                    ),
                    tooltip=[
                        alt.Tooltip("correct_answer:O", title="정답"),
                        alt.Tooltip("answer:O", title="예측"),
                        alt.Tooltip("count:Q", title="문항 수"),
                    ],
                )
                .properties(width=400, height=400, title="Confusion Matrix")
            )

            # Add text labels on heatmap
            text = heatmap.mark_text(baseline="middle").encode(
                text="count:Q",
                color=alt.condition(
                    alt.datum.count > confusion_data["count"].max() / 2,
                    alt.value("white"),
                    alt.value("black"),
                ),
            )

            st.altair_chart(heatmap + text, use_container_width=False)
        else:
            st.warning("Confusion Matrix를 생성하기 위한 데이터가 없습니다.")

        st.divider()

        # ------------------------------------------
        # 입력 길이 구간별 분석
        # ------------------------------------------
        st.subheader("📏 입력 길이 구간별 분석")

        if "input_length" in filtered_df.columns and not filtered_df.empty:
            maxbins = 20  # 히스토그램과 표에서 공통 사용

            # ==============================
            # 1) 정답 / 오답 히스토그램
            # ==============================
            chart_data = filtered_df[["input_length", "is_correct"]].copy()
            chart_data["status"] = chart_data["is_correct"].map(
                {True: "정답 (Correct)", False: "오답 (Incorrect)"}
            )

            hist_chart = (
                alt.Chart(chart_data)
                .mark_bar()
                .encode(
                    x=alt.X(
                        "input_length:Q",
                        bin=alt.Bin(maxbins=maxbins),
                        title="입력 길이 (글자 수)",
                    ),
                    y=alt.Y("count():Q", title="문항 수"),
                    color=alt.Color(
                        "status:N",
                        scale=alt.Scale(
                            domain=["정답 (Correct)", "오답 (Incorrect)"],
                            range=["#3b82f6", "#ef4444"],
                        ),
                        title="정답 여부",
                    ),
                    tooltip=[
                        alt.Tooltip("status:N", title="정답 여부"),
                        alt.Tooltip("count():Q", title="문항 수"),
                    ],
                )
                .properties(title="입력 길이 별 정답/오답 분포")
                .interactive()
            )

            st.altair_chart(hist_chart, use_container_width=True)

            st.caption(
                "※ 히스토그램은 입력 길이 분포를 보여주며, 구간별 정답률의 신뢰도는 "
                "아래 표의 문항 수를 함께 확인해야 합니다."
            )

            # ==============================
            # 2) 입력 길이 구간별 정답률 표 (0~100, 100~200 ...)
            # ==============================
            bin_step = 100

            tmp = filtered_df[["input_length", "is_correct"]].copy()

            # 구간 경계 생성
            min_len = int(tmp["input_length"].min() // bin_step * bin_step)
            max_len = int((tmp["input_length"].max() // bin_step + 1) * bin_step)
            bins = list(range(min_len, max_len + bin_step, bin_step))

            # 구간 나누기
            tmp["length_bin"] = pd.cut(
                tmp["input_length"],
                bins=bins,
                right=False,  # [0,100), [100,200) 형태
            )

            # 집계
            bin_table = (
                tmp.groupby("length_bin", observed=True)
                .agg(
                    문항수=("is_correct", "count"),
                    평균길이=("input_length", "mean"),
                    정답수=("is_correct", "sum"),
                    정답률=("is_correct", "mean"),
                )
                .reset_index()
            )

            # 🔹 구간 라벨을 "0~100" 형식으로 변환
            bin_table["길이 구간"] = bin_table["length_bin"].apply(
                lambda x: f"{int(x.left)}~{int(x.right)}"
            )

            # 포맷 정리
            bin_table["정답률(%)"] = (bin_table["정답률"] * 100).round(2)
            bin_table["평균길이"] = bin_table["평균길이"].round(0).astype(int)

            bin_table = bin_table[
                ["길이 구간", "문항수", "평균길이", "정답수", "정답률(%)"]
            ]

            # 정렬 옵션
            sort_by = st.radio(
                "구간별 정답률 표 정렬 기준",
                ["길이 구간 순", "정답률 높은 순", "정답률 낮은 순"],
                horizontal=True,
            )

            if sort_by == "정답률 높은 순":
                bin_table = bin_table.sort_values("정답률(%)", ascending=False)
            elif sort_by == "정답률 낮은 순":
                bin_table = bin_table.sort_values("정답률(%)", ascending=True)

            st.dataframe(bin_table, use_container_width=True, hide_index=True)

        else:
            st.info("입력 길이 분석을 위한 데이터가 없습니다.")

        st.divider()

        # ------------------------------------------
        # 데이터 라벨 분포 및 정답률
        # ------------------------------------------
        st.subheader("🏷️ 데이터 라벨 분포 및 정답률")

        if potential_cats:
            selected_cat = st.selectbox("분석할 라벨(Feature) 선택", potential_cats)

            if selected_cat:
                row_c1, row_c2 = st.columns(2)

                with row_c1:
                    st.markdown(f"**'{selected_cat}' 분포 (문항 수)**")
                    dist_counts = filtered_df[selected_cat].value_counts().reset_index()
                    dist_counts.columns = [selected_cat, "문항 수"]

                    dist_chart = (
                        alt.Chart(dist_counts)
                        .mark_bar(color="#3b82f6")
                        .encode(
                            x=alt.X(
                                f"{selected_cat}:N",
                                title=None,
                                axis=alt.Axis(labelAngle=0),
                            ),
                            y=alt.Y("문항 수:Q", title="문항 수"),
                            tooltip=[selected_cat, "문항 수"],
                        )
                        .properties(height=300)
                    )
                    st.altair_chart(dist_chart, use_container_width=True)

                with row_c2:
                    st.markdown(f"**'{selected_cat}'별 정답률 (%)**")
                    cat_acc = (
                        filtered_df.groupby(selected_cat)["is_correct"].mean() * 100
                    )
                    cat_acc_df = cat_acc.reset_index()
                    cat_acc_df.columns = [selected_cat, "정답률 (%)"]

                    acc_chart = (
                        alt.Chart(cat_acc_df)
                        .mark_bar(color="#f97316")
                        .encode(
                            x=alt.X(
                                f"{selected_cat}:N",
                                title=None,
                                axis=alt.Axis(labelAngle=0),
                            ),
                            y=alt.Y("정답률 (%):Q", title="정답률 (%)"),
                            tooltip=[
                                selected_cat,
                                alt.Tooltip("정답률 (%):Q", format=".1f"),
                            ],
                        )
                        .properties(height=300)
                    )
                    st.altair_chart(acc_chart, use_container_width=True)
        else:
            st.info(
                "분석할 추가적인 데이터 라벨(카테고리)이 발견되지 않았습니다. (고유값 50개 미만인 컬럼 없음)"
            )

    # ==========================================
    # 탭 2: 모델 오답 분석
    # ==========================================
    with tab_error_analysis:
        st.header("오답 문제 상세 확인")

        # 오답만 필터링
        error_df = filtered_df[~filtered_df["is_correct"]]

        st.markdown(
            f"**현재 필터 기준 오답 문항 수**: {len(error_df)} / {len(filtered_df)}"
        )

        # ------------------------------------------
        # CSV Export
        # ------------------------------------------
        if len(error_df) > 0:
            col_export1, col_export2 = st.columns([1, 4])
            with col_export1:
                csv_data = error_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="📥 오답 데이터 다운로드 (CSV)",
                    data=csv_data,
                    file_name="error_analysis.csv",
                    mime="text/csv",
                )

        if len(error_df) == 0:
            st.success("🎉 해당 조건에서 틀린 문제가 없습니다!")
            return

        st.divider()

        # ------------------------------------------
        # Pagination
        # ------------------------------------------
        st.subheader("📄 오답 목록")

        items_per_page = st.slider(
            "페이지당 문항 수", min_value=5, max_value=50, value=10
        )
        total_pages = max(1, (len(error_df) - 1) // items_per_page + 1)
        page_num = st.number_input(
            "페이지", min_value=1, max_value=total_pages, value=1
        )

        start_idx = (page_num - 1) * items_per_page
        end_idx = start_idx + items_per_page
        paginated_df = error_df.iloc[start_idx:end_idx]

        st.caption(f"페이지 {page_num} / {total_pages} (총 {len(error_df)}개 오답)")

        # 리스트 출력
        for _, row in paginated_df.iterrows():
            with st.expander(f"❌ [오답] ID: {row['id']}"):
                st.markdown("### 지문 (Paragraph)")
                st.info(row["paragraph"])

                st.markdown(f"### 질문: {row['question']}")

                try:
                    choices = row["choices"]
                    correct_idx = int(row["correct_answer"]) - 1
                    pred_idx = int(row["answer"]) - 1
                except (ValueError, TypeError):
                    st.warning("선택지/정답 파싱 오류")
                    continue

                for i, choice in enumerate(choices):
                    prefix = ""
                    color = "black"
                    bg_color = "transparent"

                    if i == correct_idx:
                        prefix += "✅ (정답) "
                        color = "green"
                        bg_color = "#e6ffe6"

                    if i == pred_idx:
                        prefix += "🤖 (예측) "
                        if i != correct_idx:
                            color = "red"
                            bg_color = "#ffe6e6"

                    st.markdown(
                        f"<div style='background-color: {bg_color}; padding: 5px; border-radius: 5px; color: {color};'>"
                        f"{i + 1}. {prefix}{choice}</div>",
                        unsafe_allow_html=True,
                    )

                st.divider()
                st.markdown("**메타데이터**")
                meta_cols = [
                    col
                    for col in row.index
                    if col
                    not in [
                        "paragraph",
                        "problems",
                        "question",
                        "choices",
                        "is_correct",
                        "input_length",
                    ]
                ]
                st.json(row[meta_cols].to_dict())

    # ==========================================
    # 탭 3: 모델 비교 (조건부)
    # ==========================================
    if enable_comparison and merged_df_2 is not None:
        with tab_comparison:
            st.header("🔄 모델 비교 분석")

            # 필터링된 데이터 기준으로 비교
            filtered_ids = set(filtered_df["id"].tolist())
            filtered_df_2_comp = merged_df_2[merged_df_2["id"].isin(filtered_ids)]

            model1_correct = set(filtered_df[filtered_df["is_correct"]]["id"].tolist())
            model2_correct = set(
                filtered_df_2_comp[filtered_df_2_comp["is_correct_2"]]["id"].tolist()
            )
            all_ids = filtered_ids

            # Calculate sets
            both_correct = model1_correct & model2_correct
            only_model1_correct = model1_correct - model2_correct
            only_model2_correct = model2_correct - model1_correct
            both_wrong = all_ids - model1_correct - model2_correct

            # Summary metrics
            st.subheader("📊 비교 요약")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("둘 다 정답", len(both_correct))
            col2.metric("모델 1만 정답", len(only_model1_correct))
            col3.metric("모델 2만 정답", len(only_model2_correct))
            col4.metric("둘 다 오답", len(both_wrong))

            st.divider()

            # Accuracy comparison
            st.subheader("📈 정확도 비교")
            acc_model1 = (
                len(model1_correct) / len(all_ids) * 100 if len(all_ids) > 0 else 0
            )
            acc_model2 = (
                len(model2_correct) / len(all_ids) * 100 if len(all_ids) > 0 else 0
            )

            acc_comparison = pd.DataFrame(
                {
                    "모델": ["모델 1", "모델 2"],
                    "정답 수": [len(model1_correct), len(model2_correct)],
                    "정확도 (%)": [f"{acc_model1:.2f}%", f"{acc_model2:.2f}%"],
                }
            )
            st.dataframe(acc_comparison, hide_index=True, use_container_width=True)

            st.divider()

            # ------------------------------------------
            # 입력 길이별 모델 정답률 비교
            # ------------------------------------------
            st.subheader("📏 입력 길이별 모델 정답률 비교")
            st.caption("입력 길이 구간별로 두 모델의 정답률을 비교합니다.")

            if "input_length" in filtered_df.columns and not filtered_df.empty:
                bin_step = 100

                # 모델 2에 input_length 추가
                filtered_df_2 = merged_df_2[
                    merged_df_2["id"].isin(filtered_df["id"])
                ].copy()
                filtered_df_2["input_length"] = (
                    filtered_df.set_index("id")["input_length"]
                    .reindex(filtered_df_2["id"])
                    .values
                )

                # 구간 경계 생성
                min_len = int(filtered_df["input_length"].min() // bin_step * bin_step)
                max_len = int(
                    (filtered_df["input_length"].max() // bin_step + 1) * bin_step
                )
                bins = list(range(min_len, max_len + bin_step, bin_step))

                # 모델 1 구간별 정답률
                tmp1 = filtered_df[["input_length", "is_correct"]].copy()
                tmp1["length_bin"] = pd.cut(
                    tmp1["input_length"], bins=bins, right=False
                )
                bin_acc_model1 = (
                    tmp1.groupby("length_bin", observed=True)["is_correct"].mean() * 100
                )
                bin_count = tmp1.groupby("length_bin", observed=True).size()

                # 모델 2 구간별 정답률
                tmp2 = filtered_df_2[["input_length", "is_correct_2"]].copy()
                tmp2["length_bin"] = pd.cut(
                    tmp2["input_length"], bins=bins, right=False
                )
                bin_acc_model2 = (
                    tmp2.groupby("length_bin", observed=True)["is_correct_2"].mean()
                    * 100
                )

                # Line Chart with Points
                all_bins = sorted(set(bin_acc_model1.index) | set(bin_acc_model2.index))
                bin_labels = [f"{int(b.left)}~{int(b.right)}" for b in all_bins]
                # x축에 구간 중간값 사용 (정렬용)
                bin_mid = [(b.left + b.right) / 2 for b in all_bins]

                length_chart_data = pd.DataFrame(
                    {
                        "길이 구간": bin_labels * 2,
                        "구간 중간값": bin_mid * 2,
                        "정답률 (%)": list(
                            bin_acc_model1.reindex(all_bins, fill_value=0)
                        )
                        + list(bin_acc_model2.reindex(all_bins, fill_value=0)),
                        "모델": ["모델 1"] * len(all_bins) + ["모델 2"] * len(all_bins),
                    }
                )

                # Line + Point chart
                base = alt.Chart(length_chart_data).encode(
                    x=alt.X(
                        "구간 중간값:Q",
                        title="입력 길이",
                        scale=alt.Scale(domain=[min(bin_mid) - 50, max(bin_mid) + 50]),
                    ),
                    y=alt.Y(
                        "정답률 (%):Q",
                        title="정답률 (%)",
                        scale=alt.Scale(domain=[0, 100]),
                    ),
                    color=alt.Color(
                        "모델:N",
                        scale=alt.Scale(
                            domain=["모델 1", "모델 2"], range=["#f97316", "#9ca3af"]
                        ),
                        legend=alt.Legend(title="모델"),
                    ),
                    tooltip=[
                        "길이 구간",
                        "모델",
                        alt.Tooltip("정답률 (%):Q", format=".1f"),
                    ],
                )

                line = base.mark_line(strokeWidth=3)
                points = base.mark_point(size=100, filled=True)

                length_line_chart = (line + points).properties(
                    title="입력 길이 구간별 모델 정답률 비교",
                    height=350,
                )

                st.altair_chart(length_line_chart, use_container_width=True)

                # 비교 테이블
                st.markdown("**📋 입력 길이별 정답률 비교 테이블**")
                length_comparison_table = pd.DataFrame(
                    {
                        "길이 구간": bin_labels,
                        "문항 수": [bin_count.get(b, 0) for b in all_bins],
                        "모델 1 정답률": [
                            f"{bin_acc_model1.get(b, 0):.1f}%" for b in all_bins
                        ],
                        "모델 2 정답률": [
                            f"{bin_acc_model2.get(b, 0):.1f}%" for b in all_bins
                        ],
                        "차이 (Δ)": [
                            f"{bin_acc_model1.get(b, 0) - bin_acc_model2.get(b, 0):+.1f}%"
                            for b in all_bins
                        ],
                    }
                )
                st.dataframe(
                    length_comparison_table, hide_index=True, use_container_width=True
                )
            else:
                st.info("입력 길이 분석을 위한 데이터가 없습니다.")

            st.divider()

            # ------------------------------------------
            # 라벨별 모델 성능 비교
            # ------------------------------------------
            st.subheader("🏷️ 라벨별 모델 성능 비교")
            st.caption("각 라벨(카테고리)에서 두 모델의 정답률을 비교합니다.")

            if potential_cats:
                selected_cat_comp = st.selectbox(
                    "분석할 라벨(Feature) 선택",
                    potential_cats,
                    key="comparison_cat_select",
                )

                if selected_cat_comp:
                    # 모델 1, 모델 2 정답률 계산
                    cat_acc_model1 = (
                        filtered_df.groupby(selected_cat_comp)["is_correct"].mean()
                        * 100
                    )
                    cat_count = filtered_df.groupby(selected_cat_comp).size()

                    # 모델 2 정답률 계산
                    filtered_df_2 = merged_df_2[
                        merged_df_2["id"].isin(filtered_df["id"])
                    ]
                    cat_acc_model2 = (
                        filtered_df_2.groupby(selected_cat_comp)["is_correct_2"].mean()
                        * 100
                    )

                    # Grouped Bar Chart (Altair)
                    all_labels = sorted(
                        set(cat_acc_model1.index) | set(cat_acc_model2.index)
                    )
                    chart_data = pd.DataFrame(
                        {
                            "라벨": list(all_labels) * 2,
                            "정답률 (%)": list(
                                cat_acc_model1.reindex(all_labels, fill_value=0)
                            )
                            + list(cat_acc_model2.reindex(all_labels, fill_value=0)),
                            "모델": ["모델 1"] * len(all_labels)
                            + ["모델 2"] * len(all_labels),
                        }
                    )

                    grouped_chart = (
                        alt.Chart(chart_data)
                        .mark_bar()
                        .encode(
                            x=alt.X("라벨:N", title=selected_cat_comp, sort=all_labels),
                            y=alt.Y("정답률 (%):Q", title="정답률 (%)"),
                            color=alt.Color(
                                "모델:N",
                                scale=alt.Scale(
                                    domain=["모델 1", "모델 2"],
                                    range=["#f97316", "#9ca3af"],
                                ),
                            ),
                            xOffset="모델:N",
                            tooltip=[
                                "라벨",
                                "모델",
                                alt.Tooltip("정답률 (%):Q", format=".1f"),
                            ],
                        )
                        .properties(title=f"'{selected_cat_comp}'별 모델 정답률 비교")
                    )

                    st.altair_chart(grouped_chart, use_container_width=True)

                    # 비교 테이블
                    st.markdown("**📋 라벨별 정답률 비교 테이블**")
                    comparison_table = pd.DataFrame(
                        {
                            "라벨": all_labels,
                            "문항 수": [cat_count.get(lbl, 0) for lbl in all_labels],
                            "모델 1 정답률": [
                                f"{cat_acc_model1.get(lbl, 0):.1f}%"
                                for lbl in all_labels
                            ],
                            "모델 2 정답률": [
                                f"{cat_acc_model2.get(lbl, 0):.1f}%"
                                for lbl in all_labels
                            ],
                            "차이 (Δ)": [
                                f"{cat_acc_model1.get(lbl, 0) - cat_acc_model2.get(lbl, 0):+.1f}%"
                                for lbl in all_labels
                            ],
                        }
                    )
                    st.dataframe(
                        comparison_table, hide_index=True, use_container_width=True
                    )
            else:
                st.info(
                    "분석할 추가적인 데이터 라벨(카테고리)이 발견되지 않았습니다. (고유값 50개 미만인 컬럼 없음)"
                )

            st.divider()

            # ------------------------------------------
            # 모델 간 차이 분석
            # ------------------------------------------
            st.subheader("🔍 모델 간 차이 분석")

            diff_type = st.selectbox(
                "보기 옵션",
                ["모델 1만 정답인 문제", "모델 2만 정답인 문제", "둘 다 오답인 문제"],
            )

            if diff_type == "모델 1만 정답인 문제":
                diff_ids = list(only_model1_correct)
            elif diff_type == "모델 2만 정답인 문제":
                diff_ids = list(only_model2_correct)
            else:
                diff_ids = list(both_wrong)

            st.markdown(f"**{diff_type}**: {len(diff_ids)}개")

            if diff_ids:
                diff_df = merged_df[merged_df["id"].isin(diff_ids)]

                # Pagination
                items_per_page_diff = st.slider(
                    "페이지당 문항 수 (비교)",
                    min_value=5,
                    max_value=50,
                    value=10,
                    key="diff_per_page",
                )
                total_pages_diff = max(1, (len(diff_df) - 1) // items_per_page_diff + 1)
                page_num_diff = st.number_input(
                    "페이지 (비교)",
                    min_value=1,
                    max_value=total_pages_diff,
                    value=1,
                    key="diff_page",
                )

                start_idx_diff = (page_num_diff - 1) * items_per_page_diff
                end_idx_diff = start_idx_diff + items_per_page_diff
                paginated_diff_df = diff_df.iloc[start_idx_diff:end_idx_diff]

                st.caption(
                    f"페이지 {page_num_diff} / {total_pages_diff} (총 {len(diff_df)}개 차이)"
                )

                for _, row in paginated_diff_df.iterrows():
                    with st.expander(f"ID: {row['id']}"):
                        st.info(
                            row["paragraph"][:200] + "..."
                            if len(str(row["paragraph"])) > 200
                            else row["paragraph"]
                        )
                        st.markdown(f"**질문**: {row['question']}")

                        # Choices display
                        try:
                            choices = row["choices"]
                            correct_idx = int(row["correct_answer"]) - 1
                            pred1_idx = int(row["answer"]) - 1

                            # Model 2 prediction lookup
                            row2 = merged_df_2[merged_df_2["id"] == row["id"]].iloc[0]
                            pred2_idx = int(row2["answer"]) - 1
                        except (ValueError, TypeError, IndexError):
                            st.warning("선택지/정답 파싱 오류")
                            continue

                        st.markdown(
                            f"**정답**: {row['correct_answer']} | **모델 1 예측**: {row['answer']} | **모델 2 예측**: {row2['answer']}"
                        )

                        for i, choice in enumerate(choices):
                            prefix = ""
                            color = "black"
                            bg_color = "transparent"

                            if i == correct_idx:
                                prefix += "✅ (정답) "
                                color = "green"
                                bg_color = "#e6ffe6"

                            pred_info = []
                            if i == pred1_idx:
                                pred_info.append("🤖1")
                            if i == pred2_idx:
                                pred_info.append("🤖2")

                            if pred_info:
                                prefix += f"{' & '.join(pred_info)} (예측) "
                                if i != correct_idx:
                                    color = "red"
                                    bg_color = "#ffe6e6"

                            st.markdown(
                                f"<div style='background-color: {bg_color}; padding: 5px; border-radius: 5px; color: {color};'>"
                                f"{i + 1}. {prefix}{choice}</div>",
                                unsafe_allow_html=True,
                            )


if __name__ == "__main__":
    main()
