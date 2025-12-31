import pandas as pd

try:
    df_csv_basic = pd.read_csv("./data/rag_db/rag_data_basic.csv", index_col=0)

    df_csv_basic_tmp = df_csv_basic[["topic", "content"]]
    print(f"기본 CSV 로드 완료: {len(df_csv_basic_tmp)}행")
except Exception as e:
    print(f"기본 CSV 로드 실패: {e}")
    df_csv_basic_tmp = pd.DataFrame(columns=["topic", "content"])

# 2. JSON 데이터 불러오기 (history_data_final.json)
try:
    # JSON 파일을 데이터프레임으로 변환
    df_json = pd.read_json("./data/rag_db/history_data_final.json", encoding="utf-8")

    # 컬럼명 통일 (title -> topic, context -> content)
    df_json = df_json.rename(columns={"title": "topic", "context": "content"})

    # 필요한 컬럼만 추출
    df_json_tmp = df_json[["topic", "content"]]
    print(f"JSON 데이터 로드 완료: {len(df_json_tmp)}행")
except Exception as e:
    print(f"JSON 파일 로드 실패: {e}")
    df_json_tmp = pd.DataFrame(columns=["topic", "content"])

# 3. 데이터 합치기
df_combined = pd.concat([df_csv_basic_tmp, df_json_tmp], ignore_index=True)

# 4. 최종 결과 저장
output_path = "./data/rag_db/combined_rag_data.csv"
df_combined.to_csv(output_path, index=True, index_label="index", encoding="utf-8-sig")

print("-" * 40)
print("최종 통합 완료")
print(f"총 행 개수: {len(df_combined)}")
print(f"결과 파일: {output_path}")
print("-" * 40)
