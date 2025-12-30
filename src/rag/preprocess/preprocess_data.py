import pandas as pd

# 1. 엑셀 데이터 불러오기 및 가공
df_excel = pd.read_excel("./data/rag_db/rag_data_basic.xlsx", header=0)

# 결측치 처리 및 컬럼 합치기
df_excel["설명"] = df_excel["설명"].fillna("")
df_excel["content"] = df_excel["설명"] + " " + df_excel["내용"]
df_excel["topic"] = df_excel["한글명칭"] + " " + df_excel["제목"]  # topic 열 생성

# 필요한 컬럼만 추출
df_excel_tmp = df_excel[["topic", "content"]]

# 2. CSV 데이터 불러오기 (index, title, context 열 존재)
df_csv = pd.read_csv("./data/rag_db/refined_history_data.csv")

# [변경사항] CSV의 title은 topic으로, context는 content로 이름을 변경
df_csv = df_csv.rename(columns={"title": "topic", "context": "content"})

# 필요한 컬럼만 추출 (기존의 구형 index는 제외)
df_csv_tmp = df_csv[["topic", "content"]]

# 3. 데이터 합치기 (기존 인덱스 무시하고 0부터 새로 생성)
df_combined = pd.concat([df_excel_tmp, df_csv_tmp], ignore_index=True)

# 4. 최종 결과 저장
# index=True, index_label='index'를 통해 통합된 데이터에 새로운 순번 부여
output_path = "./data/rag_db/combined_rag_data.csv"
df_combined.to_csv(output_path, index=True, index_label="index", encoding="utf-8-sig")

print("'topic'과 'content'로 컬럼 통일 및 합치기 완료!")
print(f"총 행 개수: {len(df_combined)}")
print(f"저장 파일: {output_path}")
