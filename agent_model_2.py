import os
import pickle
import numpy as np
from langchain_openai import ChatOpenAI
from langchain.docstore.document import Document
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.schema import SystemMessage, HumanMessage
from sklearn.metrics.pairwise import cosine_similarity

import pdb
import re
# 임베딩 캐시 파일 경로
EMBEDDING_CACHE_FILE = "./embeddings_cache.pkl"

# 캐시 로드/저장 함수
def load_embeddings_cache():
    if os.path.exists(EMBEDDING_CACHE_FILE):
        with open(EMBEDDING_CACHE_FILE, "rb") as f:
            return pickle.load(f)
    return {}

def save_embeddings_cache(cache):
    with open(EMBEDDING_CACHE_FILE, "wb") as f:
        pickle.dump(cache, f)

# 데이터 로드 및 문서 변환
def load_and_prepare_data(file_path):
    import pandas as pd
    df = pd.read_excel(file_path, engine="openpyxl")
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
    documents = [
        Document(page_content=row['body'], metadata={"product_name": row['제품명'], "company_name": row['업체명']})
        for _, row in df.iterrows()
    ]
    return documents

# 벡터 스토어 생성 (캐시와 OpenAI API 혼용)

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


# 벡터 스토어 생성 (캐시와 OpenAI API 혼용)
def build_vector_store(documents):
    embedding_model = OpenAIEmbeddings(model='text-embedding-3-large')
    embedding_cache = load_embeddings_cache()
    new_embeddings = {}

    vectors = []
    filtered_documents = []

    for doc in documents:
        content = doc.page_content
        content = preprocess_text(content)
        # pdb.set_trace()

        if content in embedding_cache:
            print(f"✅ Using cached embedding for: {content[:30]}...")
            vectors.append(embedding_cache[content])
        else:
            print(f"🚨 Generating new embedding for: {content[:30]}...")
            embedding = embedding_model.embed_query(content)
            new_embeddings[content] = embedding
            vectors.append(embedding)

        filtered_documents.append(doc)

    # 캐시 업데이트 및 저장
    embedding_cache.update(new_embeddings)
    save_embeddings_cache(embedding_cache)

    return np.array(vectors), filtered_documents

# 유사도 검색 함수
def similarity_search(question, vectors, documents, embedding_model, top_k=5):
    question_embedding = embedding_model.embed_query(question)
    similarities = cosine_similarity([question_embedding], vectors)[0]
    top_indices = np.argsort(similarities)[::-1][:top_k]

    return [documents[i] for i in top_indices]

# 키워드 검색 함수
def keyword_search(question, documents):
    keywords = question.split()  # 질문에서 키워드 추출
    return [
        doc for doc in documents
        if any(keyword in doc.page_content for keyword in keywords)
    ]

# RAG 쿼리 함수
def query_rag(question, vectors, documents, model, embedding_model, k=5):
    # Step 1: 유사도 검색
    search_results = similarity_search(question, vectors, documents, embedding_model, top_k=k)
    
    # Step 2: 키워드 검색
    keyword_results = keyword_search(question, documents)
    
    # Step 3: 검색 결과 병합 (중복 제거)
    combined_results = {doc.page_content: doc for doc in search_results + keyword_results}.values()
    
    if not combined_results:
        return "관련된 정보를 찾을 수 없습니다."

    # 검색된 문서 내용 합치기
    context = "\n\n".join([result.page_content for result in combined_results])
    system_prompt = (
        "다음 정보를 참고해 사용자 질문에 답해주세요. "
        "만약 해당 내용이 부족하거나 확신이 없다면, 모른다고 답해주세요.\n\n"
        f"{context}"
    )

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=question),
    ]
    response = model.invoke(messages)
    return response.content

# 메인 실행
if __name__ == "__main__":
    file_path = "./edruginfo.xlsx"
    documents = load_and_prepare_data(file_path)
    vectors, filtered_documents = build_vector_store(documents)
    llm_model = ChatOpenAI(model='gpt-4o-mini', temperature=0)
    embedding_model = OpenAIEmbeddings(model='text-embedding-3-large')

    while True:
        question = input("질문을 입력해주세요 (종료하려면 'exit' 입력): ")
        if question.lower() == 'exit':
            break
        answer = query_rag(question, vectors, filtered_documents, llm_model, embedding_model, k=5)
        print("\nA:", answer)
