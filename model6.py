import json
import requests
from langchain_openai import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.schema import SystemMessage, HumanMessage
from config import *

# ------------------------------------------------------------------
# [1] OpenSearch k-NN 쿼리로 유사 문서 검색
# ------------------------------------------------------------------
def knn_search(query_vector, index_name="embeddings-index", top_k=5):
    knn_query = {
        "size": top_k,
        "query": {
            "knn": {
                "embedding": {
                    "vector": query_vector,
                    "k": top_k
                }
            }
        }
    }

    response = requests.post(
        url=f"{OPENSEARCH_URL}/{index_name}/_search",
        data=json.dumps(knn_query),
        headers=OPENSEARCH_HEADERS,
        auth=OPENSEARCH_AUTH
    )

    assert response.status_code == 200, f"Failed to perform k-NN search: {response.text}"
    hits = response.json()["hits"]["hits"]
    return [hit["_source"]["content"] for hit in hits]

# ------------------------------------------------------------------
# [2] RAG 기반 질의 응답
# ------------------------------------------------------------------
def query_rag_with_knn(question, model, embedding_model, top_k=5):
    # 1) 질문 임베딩 생성
    question_embedding = embedding_model.embed_query(question)

    # 2) OpenSearch k-NN 쿼리로 유사 문서 검색
    search_results = knn_search(question_embedding, top_k=top_k)

    if not search_results:
        # 검색된 문서가 없을 경우 기본 프롬프트 사용
        fallback_prompt = f"""
너는 친절하고 전문적인 약사(Pharmacist) 역할을 맡고 있다.
사용자가 의약품 관련 질문을 했으나, 참고할 정보가 충분하지 않다.

- 가능한 일반적인 의약품 관련 지식을 바탕으로 질문에 답변하라.
- 명확한 답변이 불가능하면, '이 질문에 대한 정확한 정보를 찾을 수 없습니다. 하지만 일반적으로...'라는 형식으로 답하라.
- 답변은 사용자가 이해하기 쉽고 따뜻한 어조로 작성하라.

사용자 질문:
{question}
"""
        messages = [
            SystemMessage(content=fallback_prompt),
            HumanMessage(content=question),
        ]
        response = model.invoke(messages)
        return response.content

    # 3) 검색된 문서 내용을 합쳐 컨텍스트 생성
    context = "\n\n".join(search_results)

    # 4) LLM 기반 답변 생성
    system_prompt = f"""
너는 사용자가 약사에게 묻는 것처럼 질문에 친절하고 전문적으로 답변하는 약사이다.
다음은 답변 작성 시의 주의사항이다:
- 사용자의 걱정과 궁금증을 공감하며 답하라.
- 제공된 정보에서만 근거를 찾되, 이해하기 쉽고 친절한 어조를 유지하라.
- 정보가 부족할 경우에는 '제공된 정보로는 정확히 알 수 없습니다만, 일반적으로...'라는 형식으로 답하라.

다음은 질문과 답변 예시이다:
예시 1:
질문: 타이레놀의 효능은 무엇인가요?
답변: 타이레놀은 두통 및 발열을 완화하는 데 효과적이에요. 특히 감기 증상을 완화하는 데 자주 사용됩니다.

예시 2:
질문: 이부프로펜의 복용법은 무엇인가요?
답변: 이부프로펜은 성인의 경우 1일 3회, 식사 후 복용하는 것이 권장돼요. 식사와 함께 복용하면 위장 장애를 줄일 수 있어요.

참고할 수 있는 정보:
{context}
"""
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=question),
    ]

    response = model.invoke(messages)
    return response.content

# ------------------------------------------------------------------
# [3] 메인 실행부
# ------------------------------------------------------------------
if __name__ == "__main__":
    # LLM & Embedding 모델 준비
    llm_model = ChatOpenAI(model='gpt-4o-mini', temperature=0)
    embedding_model = OpenAIEmbeddings(model='text-embedding-3-large')

    while True:
        question = input("질문을 입력해주세요 (종료하려면 'exit' 입력): ")
        if question.lower() == 'exit':
            break

        answer = query_rag_with_knn(question, llm_model, embedding_model, top_k=5)
        print("\nA:", answer)

