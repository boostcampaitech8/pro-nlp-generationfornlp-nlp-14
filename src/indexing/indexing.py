import json
import os

import dotenv
from openai import OpenAI

# 프로젝트 구조에 맞춘 임포트 (경로를 명확히 지정)
from vectorstore import (
    ChunkDoc,
    ESConfig,
    ParentDoc,
    WriteThroughStore,
    check_connection,
    create_es_client,
)

# 1. 환경 변수 및 설정 로드
dotenv.load_dotenv()
config = ESConfig()
es = create_es_client(config)
store = WriteThroughStore(es, config)

client = OpenAI(api_key=os.getenv("SOLAR_PRO2_API_KEY"), base_url="https://api.upstage.ai/v1")

EMB_MODEL = "embedding-passage"
DIMENSIONS = config.embedding_dims  # Solar Pro2: 4096


# 2. 배치 임베딩 생성 함수
def get_solar_embeddings_batch(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []

    # 토큰 제한을 피하기 위한 최소한의 전처리 (글자수 절단)
    clean_texts = [t.replace("\n", " ")[:2500] for t in texts]

    try:
        response = client.embeddings.create(input=clean_texts, model=EMB_MODEL)
        return [item.embedding for item in response.data]
    except Exception as e:
        print(f"임베딩 API 오류: {e}")
        return [[0.0] * DIMENSIONS for _ in texts]


# 3. 메인 인덱싱 로직
def run_indexing_with_batch():
    # --- [A] 부모 문서 인덱싱 ---
    print("부모 문서 인덱싱 시작...")
    with open("./data/rag_db/parent_documents.jsonl", encoding="utf-8") as f:
        parent_lines = f.readlines()

    for i in range(0, len(parent_lines), 20):
        batch_lines = parent_lines[i : i + 20]
        batch_data = [json.loads(line) for line in batch_lines]

        t_vecs = get_solar_embeddings_batch([d["topic"] for d in batch_data])
        p_vecs = get_solar_embeddings_batch([d["content"] for d in batch_data])

        parent_docs = []
        for d, t_vec, p_vec in zip(batch_data, t_vecs, p_vecs, strict=True):
            parent_docs.append(
                ParentDoc(
                    doc_id=str(d["parent_id"]),
                    subject="history",
                    topic=d["topic"],
                    parent_text=d["content"],
                    topic_vector=t_vec,
                    parent_vector=p_vec,
                    created_at=None,  # 시간 함수 없이 None 처리
                )
            )

        store.bulk_upsert_parents(parent_docs)
        print(f"부모 배치 완료 ({min(i + 20, len(parent_lines))}/{len(parent_lines)})")

    # --- [B] 자식 문서 인덱싱 ---
    print("자식 문서 인덱싱 시작...")
    with open("./data/rag_db/child_documents.jsonl", encoding="utf-8") as f:
        child_lines = f.readlines()

    for i in range(0, len(child_lines), 100):
        batch_lines = child_lines[i : i + 100]
        batch_data = [json.loads(line) for line in batch_lines]

        c_vecs = get_solar_embeddings_batch([d["chunked_content"] for d in batch_data])

        child_docs = []
        for d, c_vec in zip(batch_data, c_vecs, strict=True):
            child_docs.append(
                ChunkDoc(
                    chunk_id=d["id"],
                    doc_id=str(d["parent_id"]),
                    subject="history",
                    topic=d["topic"],
                    chunk_idx=int(d["id"].split("_")[-1]) if "_" in d["id"] else 0,
                    chunk_text=d["chunked_content"],
                    chunk_vector=c_vec,
                    created_at=None,  # 시간 함수 없이 None 처리
                )
            )

        store.bulk_upsert_chunks(child_docs)
        if (i // 100) % 5 == 0:
            print(f"자식 배치 업로드 중... ({min(i + 100, len(child_lines))}/{len(child_lines)})")


if __name__ == "__main__":
    if check_connection(es):
        print(f"연결 성공: {config.es_url}")
        run_indexing_with_batch()
        print("인덱싱 완료!")
    else:
        print("ES 연결 실패")
