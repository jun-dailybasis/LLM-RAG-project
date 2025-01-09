import json
import requests
import shortuuid
import datetime as dt
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
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
# [2] RAG 기반 답변 생성
# ------------------------------------------------------------------
def query_rag_with_knn(question, model, embedding_model, top_k=5):
    print(f"query_rag_with_knn...................... ing")

    print(f"[question]......................{question}")

    # 1) 질문 임베딩 생성
    question_embedding = embedding_model.embed_query(question)

    # 2) OpenSearch k-NN 쿼리로 유사 문서 검색
    search_results = knn_search(question_embedding, top_k=top_k)
    print(f"[search_results]  {search_results}")
          
    if not search_results:
        # 검색된 문서가 없을 경우 기본 프롬프트 사용
        print(f"유사한 검색 결과가 없어요. ")
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

    # 4) 개선된 시스템 프롬프트
    system_prompt = f"""
너는 사용자가 신뢰할 수 있는 약사로서 의약품 관련 질문에 답변한다.
다음은 답변 작성 시의 지침이다:
1. **사용자의 걱정과 궁금증에 공감하라**: 답변 시작 전에 사용자의 상황을 이해하는 어조를 사용하라.
2. **구체적이고 명확한 정보 제공**: 문서에서 제공된 정보를 바탕으로 복용법, 효능, 부작용, 주의사항 등을 설명하라.
3. **부족한 정보는 일반 지식으로 보완**: 문서에 없는 경우, '제공된 정보로는 정확히 알 수 없지만, 일반적으로...'라는 형식으로 보충하라.
4. **친절하고 이해하기 쉬운 어조 유지**: 약학 용어는 설명하고, 사용자 친화적인 언어를 사용하라.
5. ** 참고 문서 내용을 요약하여, 사용자 질문에 답변하라.

사용자 질문:
{question}

참고 문서 내용:
{context}
"""
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=question),
    ]

    response = model.invoke(messages)
    return response.content

# ------------------------------------------------------------------
# [3] 챗 기록 업로드
# ------------------------------------------------------------------
def upload_chat_history(user_id, role, text):
    doc = {
        'user_id': user_id,
        'role': role,
        'text': text,
        'timestamp': dt.datetime.now().isoformat(),
    }
    doc_id = shortuuid.uuid()

    response = requests.put(
        url=f"{OPENSEARCH_URL}/chat-history2/_doc/{doc_id}",
        data=json.dumps(doc),
        headers=OPENSEARCH_HEADERS,
        auth=OPENSEARCH_AUTH
    )

    assert response.status_code // 100 == 2, f"Failed to upload chat history: {response.text}"

# ------------------------------------------------------------------
# [4] 챗봇 응답 생성
# ------------------------------------------------------------------
def generate_chat_talk(user_id, utterance, llm_model, embedding_model):
    answer = query_rag_with_knn(utterance, llm_model, embedding_model, top_k=5)
    upload_chat_history(user_id, 'user', utterance)
    upload_chat_history(user_id, 'assistant', answer)

    body = {
        'version': '2.0',
        'template': {
            'outputs': [
                {
                    'simpleText': {
                        'text': answer,
                    }
                }
            ],
        },
    }

    return body

# ------------------------------------------------------------------
# [5] 약사 챗봇 콜백 핸들러
# ------------------------------------------------------------------
def main(event, context):
    body = json.loads(event['body'])

    user_id = body['userRequest']['user']['id']
    utterance = body['userRequest']['utterance']

    print(f"[Medicine Callback] User ID: {user_id}")
    print(f"[Medicine Callback] Utterance: {utterance}")

    llm_model = ChatOpenAI(model='gpt-4o-mini', temperature=0)
    embedding_model = OpenAIEmbeddings(model='text-embedding-3-large')

    response_body = generate_chat_talk(user_id, utterance, llm_model, embedding_model)
    return {"statusCode": 200, "body": json.dumps(response_body)}


