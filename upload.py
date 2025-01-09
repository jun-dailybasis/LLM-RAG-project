import pickle
import requests
import json
import pandas as pd
from config import *
import pdb
from langchain.docstore.document import Document
import re
import os

# ------------------------------------------------------------------
# [1] 임베딩 캐시 로드
# ------------------------------------------------------------------
EMBEDDING_CACHE_FILE = "./embeddings_cache.pkl"

def load_local_embeddings(file_path=EMBEDDING_CACHE_FILE):
    if not os.path.exists(file_path):
        print("⚠️ Embedding cache file not found.")
        return {}

    with open(file_path, "rb") as f:
        cache = pickle.load(f)

    print(f"✅ Loaded embeddings cache with {len(cache)} entries.")
    return cache

# ------------------------------------------------------------------
# [2] 엑셀 데이터 로드 및 문서 변환
# ------------------------------------------------------------------
def load_and_prepare_data(file_path):
    df = pd.read_excel(file_path, engine="openpyxl")

    # Body 생성
    df['body'] = (
    "제품명은 " + df['제품명'].fillna('') + "입니다. "
    "제조업체는 " + df['업체명'].fillna('') + "입니다. "
    "주성분은 " + df['주성분'].fillna('') + "입니다. "
    "이 약의 효능은 다음과 같습니다. " + df['이 약의 효능은 무엇입니까?'].fillna('') + ". "
    "이 약의 복용법은 다음과 같습니다. " + df['이 약은 어떻게 사용합니까?'].fillna('') + ". "
    "사용 전 주의사항은 다음과 같습니다. " + df['이 약을 사용하기 전에 반드시 알아야 할 내용은 무엇입니가?'].fillna('') + ". "
    "주의 사항은 다음과 같습니다. " + df['이 약의 사용상 주의사항은 무엇입니까?'].fillna('') + ". "
    "주의해야 할 약이나 음식은 " + df['이 약을 사용하는 동안 주의해야 할 약 또는 음식은 무엇입니까?'].fillna('') + "입니다. "
    "이 약은 다음과 같은 부작용이 있을 수 있습니다. " + df['이 약은 어떤 이상반응이 나타날 수 있습니까?'].fillna('') + ". "
    "보관방법은 다음과 같습니다. " + df['이 약은 어떻게 보관해야 합니까?'].fillna('') + "."
)

    # Document 객체로 변환
    documents = [
        Document(
            page_content=row['body'],
            metadata={
                "product_name": row['제품명'] if pd.notnull(row['제품명']) else '정보 없음',
                "company_name": row['업체명'] if pd.notnull(row['업체명']) else '정보 없음',
                "main_ingredient": row['주성분'] if pd.notnull(row['주성분']) else '정보 없음'
            }
        )
        for _, row in df.iterrows()
    ]
    return documents

def preprocess_text(text):
    # 줄 바꿈과 탭 제거 (엔터 제거 포함)
    text = text.replace("\n", " ").replace("\t", " ")
    
    # 연속된 공백 제거
    text = re.sub(r'\s+', ' ', text)

    # 불필요한 마침표 제거
    # 연속된 마침표 처리: `...` -> `.`
    text = re.sub(r'([.!?])\1+', r'\1', text)
    # 마침표가 단독으로 존재하는 경우 제거
    text = re.sub(r'\s+\. ', ' ', text)
    text = re.sub(r'\.\s+\.', '.', text)

    # 문장 기호와 공백 정리
    # 문장 기호 앞의 불필요한 공백 제거
    text = re.sub(r'\s+([.,!?])', r'\1', text)
    # 문장 기호 뒤의 불필요한 공백 조정 (단일 공백 유지)
    text = re.sub(r'([.,!?])\s+', r'\1 ', text)

    # 문자열 양 끝의 불필요한 공백 제거
    text = text.strip()

    return text

# ------------------------------------------------------------------
# [3] OpenSearch에 임베딩 업로드
# ------------------------------------------------------------------
def upload_embeddings_to_opensearch(documents, embeddings_cache, index_name="embeddings-index"):
    for doc in documents:
        content = doc.page_content
        content = preprocess_text(content)
        embedding = embeddings_cache.get(content)
        # pdb.set_trace()
        if embedding is None:
            print(f"❌ Embedding not found for: {content[:30]}...")
            continue

        # 메타데이터 가져오기
        metadata = doc.metadata

        # 업로드할 문서 구조
        doc_to_upload = {
            "content": content,
            "embedding": embedding,
            "metadata": metadata
        }

        # OpenSearch에 업로드 요청
        response = requests.post(
            url=f"{OPENSEARCH_URL}/{index_name}/_doc",
            data=json.dumps(doc_to_upload),
            headers=OPENSEARCH_HEADERS,
            auth=OPENSEARCH_AUTH
        )

        if response.status_code // 100 != 2:
            print(f"❌ Failed to upload document: {response.text}")
        else:
            print(f"✅ Uploaded document: {content[:30]}...")
            # pdb.set_trace()

# ------------------------------------------------------------------
# [4] 메인 실행부
# ------------------------------------------------------------------
if __name__ == "__main__":
    # 엑셀 파일 경로
    file_path = "edruginfo.xlsx"

    print(os.path.exists("./embeddings_cache2.pkl"))

    # 데이터 로드
    documents = load_and_prepare_data(file_path)
    
    # pdb.set_trace()
    # 임베딩 캐시 로드
    embeddings_cache = load_local_embeddings()
    # pdb.set_trace()
    # OpenSearch에 업로드
    upload_embeddings_to_opensearch(documents, embeddings_cache)
