import pandas as pd

df = pd.read_excel("./data/rag_db/rag_data_basic.xlsx", header=0)

df["설명"] = df["설명"].fillna("")

df["content"] = df["설명"] + " " + df["내용"]

df["topic"] = df["한글명칭"] + " " + df["제목"]

df_tmp = df[["topic", "content"]]

df_tmp.to_csv("./data/rag_db/rag_data_basic.csv", encoding="utf-8-sig")
print("완료")
