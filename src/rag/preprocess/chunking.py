import json

import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter

"""
Child_Document{
    id: str,
    parent_id: int,
    topic: str,
    topic_vector: float[],
    chunked_content: str,
    chunked_content_vector: float[],
}

Parent_Document{
    parent_id: int,
    topic: str,
    content: str,
}
"""


# 경로 및 청킹 파라미터 설정
INPUT_PATH = "./data/rag_db/combined_rag_data.csv"
PARENT_OUTPUT = "./data/rag_db/parent_documents.jsonl"
CHILD_OUTPUT = "./data/rag_db/child_documents.jsonl"

# 청킹 설정 (자식 조각의 크기)
# PDR에서는 작은 조각(300~500자)이 검색에 유리합니다.
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=350,
    chunk_overlap=50,
    length_function=len,
    separators=["\n\n", "\n", ". ", "? ", "! ", " ", ""],
    keep_separator="end",  # separator를 청크 끝에 포함
    strip_whitespace=True,  # 앞뒤 공백 제거
)


def save_to_jsonl(data_list, file_path):
    """리스트 데이터를 JSONL 형식으로 저장"""
    with open(file_path, "w", encoding="utf-8") as f:
        for entry in data_list:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def main():
    # 1. 통합 데이터 불러오기
    try:
        df = pd.read_csv(INPUT_PATH)
    except FileNotFoundError:
        print(f"파일을 찾을 수 없습니다: {INPUT_PATH}")
        return

    parent_documents = []
    child_documents = []

    print(f"총 {len(df)}개의 데이터 청킹 시작...")

    # 2. 데이터 순회 및 스키마 적용
    for _, row in df.iterrows():
        parent_id = int(row["index"])
        topic = str(row["topic"])
        content = str(row["content"])

        # [Parent_Document 스키마]
        parent_doc = {"parent_id": parent_id, "topic": topic, "content": content}
        parent_documents.append(parent_doc)

        # 청킹 수행 및 Child_Document 스키마 생성
        chunks = text_splitter.split_text(content)

        for c_idx, chunk in enumerate(chunks):
            child_doc = {
                "id": f"{parent_id}_{c_idx}",  # 고유 ID (부모ID_순번)
                "parent_id": parent_id,
                "topic": topic,
                "chunked_content": chunk,
                # 벡터 값은 이후 인덱싱 단계에서 API를 통해 채울 예정
                "topic_vector": None,
                "chunked_content_vector": None,
            }
            child_documents.append(child_doc)

    # 3. 중간 결과 저장
    save_to_jsonl(parent_documents, PARENT_OUTPUT)
    save_to_jsonl(child_documents, CHILD_OUTPUT)

    print("-" * 40)
    print("청킹 완료 및 파일 저장 완료")
    print(f"부모 문서: {len(parent_documents)}개 -> {PARENT_OUTPUT}")
    print(f"자식 청크: {len(child_documents)}개 -> {CHILD_OUTPUT}")
    print("-" * 40)


if __name__ == "__main__":
    main()
