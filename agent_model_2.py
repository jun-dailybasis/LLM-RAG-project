import os
import pandas as pd
import numpy as np
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain.docstore.document import Document
from langchain.schema import SystemMessage, HumanMessage

# 데이터 로드 및 body 생성
def load_and_prepare_data(file_path):
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
    return df

# 임베딩 벡터 생성
def get_embedding_vectors(df, column, batch_size=50):
    import time
    from openai.embeddings_utils import get_embedding

    embeddings = []
    texts = df[column].fillna('').tolist()

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        try:
            print(f"Processing batch {i // batch_size + 1}...")
            batch_embeddings = get_embedding(batch_texts, model="text-embedding-3-large")
            embeddings.extend(batch_embeddings)
            time.sleep(1)
        except Exception as e:
            print(f"Error processing batch {i // batch_size + 1}: {e}")
            break

    return np.array(embeddings)

# Document 변환
def convert_to_documents(df):
    documents = []
    for idx, row in df.iterrows():
        content = str(row['body'])
        metadata = {"product_name": row['제품명'], "company_name": row['업체명'], "row_index": idx}
        documents.append(Document(page_content=content, metadata=metadata))
    return documents

# 벡터스토어 생성
def build_vector_store(documents):
    embeddings = OpenAIEmbeddings(model='text-embedding-3-large')
    return DocArrayInMemorySearch.from_documents(documents, embeddings)

# RAG 쿼리
def query_rag(question, vector_store, model, k=5):
    search_results = vector_store.similarity_search(question, k=k)
    if not search_results:
        return "관련된 정보를 찾을 수 없습니다."

    context = "\n\n".join([result.page_content for result in search_results])
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
    df = load_and_prepare_data(file_path)
    df['body'] = df['body'].apply(lambda x: x[:1000] if len(x) > 1000 else x)
    documents = convert_to_documents(df)
    vector_store = build_vector_store(documents)
    llm_model = ChatOpenAI(model='gpt-4o-mini', temperature=0)

    while True:
        question = input("질문을 입력해주세요 (종료하려면 'exit' 입력): ")
        if question.lower() == 'exit':
            break
        answer = query_rag(question, vector_store, llm_model, k=5)
        print("\nA:", answer)
