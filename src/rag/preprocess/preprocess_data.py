import pandas as pd

# 1. 엑셀 데이터 불러오기 및 가공 (변형 금지 조항을 고려하여 원문 위주 결합)
try:
    df_excel = pd.read_excel("./data/rag_db/rag_data_basic.xlsx", header=0)

    # 결측치 처리
    df_excel["설명"] = df_excel["설명"].fillna("")
    df_excel["내용"] = df_excel["내용"].fillna("")

    # 원문을 훼손하지 않는 범위 내에서 검색용 컬럼 생성
    df_excel["content"] = df_excel["설명"] + " " + df_excel["내용"]
    df_excel["topic"] = df_excel["한글명칭"].fillna("") + " " + df_excel["제목"].fillna("")

    # 필요한 컬럼만 추출
    df_excel_tmp = df_excel[["topic", "content"]]
except Exception as e:
    print(f"엑셀 파일 로드 실패: {e}")
    df_excel_tmp = pd.DataFrame(columns=["topic", "content"])

# 2. JSON 데이터 불러오기
try:
    df_json = pd.read_json("./data/rag_db/history_data_final.json", encoding="utf-8")

    # JSON의 컬럼명(title -> topic, context -> content) 통일
    df_json = df_json.rename(columns={"title": "topic", "context": "content"})

    # 기존 index 열 제외
    df_json_tmp = df_json[["topic", "content"]]
except Exception as e:
    print(f"JSON 파일 로드 실패: {e}")
    df_json_tmp = pd.DataFrame(columns=["topic", "content"])

# 3. 데이터 합치기 (기존 인덱스 무시하고 0부터 새로 생성)
df_combined = pd.concat([df_excel_tmp, df_json_tmp], ignore_index=True)

# 4. 최종 결과 저장
output_path = "./data/rag_db/combined_rag_data.csv"
df_combined.to_csv(output_path, index=True, index_label="index", encoding="utf-8-sig")

print("-" * 30)
print("'topic'과 'content'로 컬럼 통일 및 합치기 완료!")
print(f"총 행 개수: {len(df_combined)}")
print(f"저장 파일: {output_path}")
print("-" * 30)
