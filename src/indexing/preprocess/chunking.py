import json

import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 경로 설정
INPUT_PATH = "./data/rag_db/combined_rag_data.csv"
PARENT_OUTPUT = "./data/rag_db/parent_documents.jsonl"
CHILD_OUTPUT = "./data/rag_db/child_documents.jsonl"

# 1. 부모 청킹 설정 (토큰 제한 4000에 맞춤 - 약 2500자 권장)
parent_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2500,
    chunk_overlap=200,  # 부모 조각 간의 문맥 유지를 위해 오버랩 부여
    length_function=len,
    separators=["\n\n", "\n", ". "],
)

# 2. 자식 청킹 설정 (검색용 - 350자)
child_splitter = RecursiveCharacterTextSplitter(
    chunk_size=350,
    chunk_overlap=50,
    length_function=len,
    separators=["\n\n", "\n", ". ", "? ", "! ", " ", ""],
    keep_separator="end",
    strip_whitespace=True,
)


def save_to_jsonl(data_list, file_path):
    with open(file_path, "w", encoding="utf-8") as f:
        for entry in data_list:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def main():
    try:
        df = pd.read_csv(INPUT_PATH)
    except FileNotFoundError:
        print(f"파일을 찾을 수 없습니다: {INPUT_PATH}")
        return

    parent_documents = []
    child_documents = []

    print(f"총 {len(df)}개의 원본 데이터 처리 시작...")

    for _, row in df.iterrows():
        original_index = int(row["index"])
        topic = str(row["topic"])
        content = str(row["content"])

        # [단계 1] 원본 본문을 임베딩 가능한 크기의 '부모 청크'들로 분할
        parent_chunks = parent_splitter.split_text(content)

        for p_idx, p_content in enumerate(parent_chunks):
            # 새로운 부모 ID 생성 (원본ID_부모순번)
            # 나중에 리랭커에서 이 ID를 기준으로 원문을 가져옵니다.
            parent_id_str = f"{original_index}_{p_idx}"
            parent_doc = {
                "parent_id": parent_id_str,  # 이제 문자열 ID를 사용하거나 정수형 변환 필요
                "topic": topic,
                "content": p_content,
            }
            parent_documents.append(parent_doc)

            # [단계 2] 각 부모 청크를 다시 '자식 청크'들로 분할
            child_chunks = child_splitter.split_text(p_content)

            for c_idx, c_content in enumerate(child_chunks):
                child_doc = {
                    "id": f"{parent_id_str}_{c_idx}",  # 고유 ID (부모ID_순번_자식순번)
                    "parent_id": parent_id_str,
                    "topic": topic,
                    "chunked_content": c_content,
                    "topic_vector": None,
                    "chunked_content_vector": None,
                }
                child_documents.append(child_doc)

    save_to_jsonl(parent_documents, PARENT_OUTPUT)
    save_to_jsonl(child_documents, CHILD_OUTPUT)

    print("-" * 40)
    print("계층적 청킹 완료")
    print(f"부모 문서(임베딩 가능): {len(parent_documents)}개")
    print(f"자식 청크(검색용): {len(child_documents)}개")
    print("-" * 40)


if __name__ == "__main__":
    main()
