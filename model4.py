import os
import pickle
import numpy as np
import pandas as pd
from langchain_openai import ChatOpenAI
from langchain.docstore.document import Document
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.schema import SystemMessage, HumanMessage
from sklearn.metrics.pairwise import cosine_similarity

# ------------------------------------------------------------------
# [1] 임베딩 캐시 로드/저장
# ------------------------------------------------------------------
EMBEDDING_CACHE_FILE = "./embeddings_cache3.pkl"

def load_embeddings_cache():
    """ 로컬 캐시 파일에서 임베딩 정보를 로드 """
    if os.path.exists(EMBEDDING_CACHE_FILE):
        with open(EMBEDDING_CACHE_FILE, "rb") as f:
            return pickle.load(f)
    return {}

def save_embeddings_cache(cache):
    """ 로컬 캐시에 임베딩 정보를 저장 """
    with open(EMBEDDING_CACHE_FILE, "wb") as f:
        pickle.dump(cache, f)

# ------------------------------------------------------------------
# [2] 엑셀 로드 → (body 생성) → 문서 리스트로 변환
# ------------------------------------------------------------------
def load_and_prepare_data(file_path):
    df = pd.read_excel(file_path, engine="openpyxl")

    # 1) Body 정의
    df['body'] = df.apply(
        lambda row: (
            f"제품명: {row['제품명'] if pd.notnull(row['제품명']) else '정보 없음'}\n"
            f"제조업체: {row['업체명'] if pd.notnull(row['업체명']) else '정보 없음'}\n"
            f"주성분: {row['주성분'] if pd.notnull(row['주성분']) else '정보 없음'}\n"
            f"효능: {row['이 약의 효능은 무엇입니까?'] if pd.notnull(row['이 약의 효능은 무엇입니까?']) else '정보 없음'}\n"
            f"복용법: {row['이 약은 어떻게 사용합니까?'] if pd.notnull(row['이 약은 어떻게 사용합니까?']) else '정보 없음'}\n"
            f"사용 전 주의사항: {row['이 약을 사용하기 전에 반드시 알아야 할 내용은 무엇입니가?'] if pd.notnull(row['이 약을 사용하기 전에 반드시 알아야 할 내용은 무엇입니가?']) else '정보 없음'}\n"
            f"사용상 주의사항: {row['이 약의 사용상 주의사항은 무엇입니까?'] if pd.notnull(row['이 약의 사용상 주의사항은 무엇입니까?']) else '정보 없음'}\n"
            f"주의해야 할 약/음식: {row['이 약을 사용하는 동안 주의해야 할 약 또는 음식은 무엇입니까?'] if pd.notnull(row['이 약을 사용하는 동안 주의해야 할 약 또는 음식은 무엇입니까?']) else '정보 없음'}\n"
            f"부작용: {row['이 약은 어떤 이상반응이 나타날 수 있습니까?'] if pd.notnull(row['이 약은 어떤 이상반응이 나타날 수 있습니까?']) else '정보 없음'}\n"
            f"보관방법: {row['이 약은 어떻게 보관해야 합니까?'] if pd.notnull(row['이 약은 어떻게 보관해야 합니까?']) else '정보 없음'}\n"
        ),
        axis=1
    )

    # 2) Document 객체로 변환
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

# ------------------------------------------------------------------
# [3] 임베딩 & 벡터 스토어 생성
# ------------------------------------------------------------------
def build_vector_store(documents):
    """
    전체 documents 리스트에 대해 임베딩을 생성/로드하고
    numpy array(vectors)와 문서 목록(filtered_documents)을 반환.
    """
    embedding_model = OpenAIEmbeddings(model='text-embedding-3-large')
    embedding_cache = load_embeddings_cache()
    new_embeddings = {}

    vectors = []
    filtered_documents = []

    for doc in documents:
        content = doc.page_content
        if content in embedding_cache:
            print(f"✅ Using cached embedding for: {content[:30]}...")
            vectors.append(embedding_cache[content])
        else:
            print(f"🚨 Generating new embedding for: {content[:30]}...")
            embedding = embedding_model.embed_query(content)
            new_embeddings[content] = embedding
            vectors.append(embedding)

        filtered_documents.append(doc)

    # 새로 생성된 임베딩을 캐시에 업데이트하고 저장
    embedding_cache.update(new_embeddings)
    save_embeddings_cache(embedding_cache)

    return np.array(vectors), filtered_documents

# ------------------------------------------------------------------
# [4] 유사도 검색 함수
# ------------------------------------------------------------------
def similarity_search(question, vectors, documents, embedding_model, top_k=5):
    """
    질문을 임베딩해, 전체 벡터 중 상위 top_k개를 골라 그 문서를 반환
    """
    question_embedding = embedding_model.embed_query(question)
    similarities = cosine_similarity([question_embedding], vectors)[0]
    top_indices = np.argsort(similarities)[::-1][:top_k]
    return [documents[i] for i in top_indices]

# ------------------------------------------------------------------
# [5] RAG 기반 질의 응답
# ------------------------------------------------------------------
def query_rag(question, vectors, documents, model, embedding_model, k=5):
    # 1) 유사도 검색
    search_results = similarity_search(question, vectors, documents, embedding_model, top_k=k)

    if not search_results:
        return "관련된 정보를 찾을 수 없습니다."

    # 2) 검색된 문서 내용 합치기
    context = "\n\n".join([doc.page_content for doc in search_results])

    # 3) Few-shot 예시 정의 (문자열로 바로 정의)
    few_shot = """
예시 1:
질문: 타이레놀의 효능은 무엇인가요?
답변: 타이레놀은 두통 및 발열 완화에 효과적입니다.

예시 2:
질문: 이부프로펜의 복용법은 무엇인가요?
답변: 이부프로펜은 성인의 경우 1일 3회, 식사 후 복용하는 것이 권장됩니다.

예시 3:
질문: 판피린의 부작용은 무엇인가요?
답변: 판피린은 졸음, 구강 건조, 소화불량 등의 부작용이 나타날 수 있습니다.

예시 4:
질문: 아스피린과 타이레놀의 차이점은 무엇인가요?
답변: 아스피린은 항염증 효과가 있으며, 타이레놀은 주로 진통과 해열에 사용됩니다.

예시 5:
질문: 부루펜의 사용상 주의사항은 무엇인가요?
답변: 부루펜은 위장 장애를 일으킬 수 있으므로 식사 후 복용하는 것이 좋습니다. 또한, 장기 복용은 피하세요.
"""

    # 4) 개선된 시스템 프롬프트
    system_prompt = f"""
너는 전문 약사(Pharmacist) 역할을 맡고 있다.
사용자의 의약품 관련 질문에 대해, 아래 제공된 정보를 참고하여
전문적이면서도 이해하기 쉽게 답변해라.

다음은 질문과 답변 예시이다:
{few_shot}

- 제공된 정보에 없는 내용이나 확신이 없는 부분에 대해서는 '잘 모르겠습니다' 또는 '해당 정보로는 답하기 어렵습니다'라고 명확하게 답하라.
- 반드시 제공된 정보만을 활용하여 답변하라.
- 답변은 간결하면서도 구체적으로 작성하라.

참고할 수 있는 정보:
{context}
"""

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=question),
    ]

    # 5) 모델 질의
    response = model.invoke(messages)
    return response.content

# ------------------------------------------------------------------
# [6] 메인 실행부
# ------------------------------------------------------------------
if __name__ == "__main__":
    file_path = "./edruginfo.xlsx"

    # 데이터 로드
    documents = load_and_prepare_data(file_path)

    # 임베딩 + 캐싱
    vectors, filtered_documents = build_vector_store(documents)

    # LLM & Embedding 모델 준비
    llm_model = ChatOpenAI(model='gpt-4o-mini', temperature=0)
    embedding_model = OpenAIEmbeddings(model='text-embedding-3-large')

    while True:
        question = input("질문을 입력해주세요 (종료하려면 'exit' 입력): ")
        if question.lower() == 'exit':
            break

        answer = query_rag(question, vectors, filtered_documents, llm_model, embedding_model, k=5)
        print("\nA:", answer)
