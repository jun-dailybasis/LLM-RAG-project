import json
import requests
import shortuuid
import datetime as dt
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain.schema import SystemMessage, HumanMessage
from config import *
from openai import OpenAI
import threading
import time
import queue as q

#사용자별 응답 큐를 저장할 전역 딕셔너리
response_map = {}
# ------------------------------------------------------------------
# [1] OpenSearch k-NN 쿼리로 유사 문서 검색
# ------------------------------------------------------------------
def knn_search(query_vector, index_name="embeddings-index", top_k=3):
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
def query_rag_with_knn(question, embedding_model, top_k=3):
    print(f"query_rag_with_knn...................... ing")

    print(f"[question]......................{question}")

    client = OpenAI()

    # 1) 질문 임베딩 생성
    question_embedding = embedding_model.embed_query(question)

    # 2) OpenSearch k-NN 쿼리로 유사 문서 검색
    search_results = knn_search(question_embedding, top_k=top_k)
    print(f"[search_results]  {search_results}")
          
    if not search_results:
        # 검색된 문서가 없을 경우 기본 프롬프트 사용
        print(f"[X] 유사한 검색 결과가 없어요. ")
        fallback_prompt = f"""
        아래 내용을 요약해 답변해줘.
        - 너는 친절한 약사(Pharmacist)이다.
        - 명확한 답변이 불가능하면, '이 질문에 대한 정확한 정보를 찾을 수 없습니다. 하지만 일반적으로...'라는 형식으로 답하라.
        - 답변은 사용자가 이해하기 쉽고, 생기발랄한 어조로 작성하라.

        사용자 질문:
        {question}
        """
        messages = [
            {"role": "system", "content": fallback_prompt},
            {"role": "user", "content": question},
        ]
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
        )
        return resp.choices[0].message.content.strip()

    context = "\n\n".join(search_results[:3])
    # 4) 개선된 시스템 프롬프트
    system_prompt = f"""
너는 사용자가 신뢰할 수 있는 약사로서 의약품 관련 질문에 답변한다.
다음은 답변 작성 시의 지침이다: 물어본 질문에 대한 답변은 2 문장 내로 줄인다. 
1. 문서에서 제공된 정보를 바탕으로 복용법, 효능, 부작용, 주의사항를 설명하라.
2. 부족한 정보는 일반 지식으로 보완: 문서에 없는 경우, '제공된 정보로는 정확히 알 수 없지만, 일반적으로...'라는 형식으로 보충하라.

사용자 질문:
{question}

참고 문서 내용:
{context}
"""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
    )
    return resp.choices[0].message.content.strip()

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
def generate_chat_talk(user_id, utterance, embedding_model, response_queue):
    # OpenAI 호출 및 RAG 생성
    try:
        answer = query_rag_with_knn(utterance, embedding_model, top_k=3)
        response_map[user_id].put({
            'version': '2.0',
            'template': {
                'outputs': [
                    {'simpleText': {'text': answer}}
                ]
            }
        })
    except Exception as e:
        response_map[user_id].put({
            'version': '2.0',
            'template': {
                'outputs': [
                    {'simpleText': {'text': "처리 중 오류가 발생했습니다. 다시 시도해주세요."}}
                ]
            }
        })
        print(f"Error in generate_chat_talk: {e}")


# 응답시간 초과시 먼저 답변
def timeover():
    response = {
        'version': '2.0',
        'template': {
            'outputs': [
                {
                    'simpleText': {
                        'text': "아직 제가 생각이 끝나지 않았어요\n잠시 후 아래 말풍선을 눌러주세요."
                    }
                }
            ],
            'quickReplies': [
                {
                    'action': 'message',
                    'label': '답변 확인하기',
                    'messageText': '생각답변 확인하기'
                }
            ]
        }
    }
    return response


# ------------------------------------------------------------------
# [5] 약사 챗봇 콜백 핸들러
# ------------------------------------------------------------------
def main(event, context):
    start_time = time.time()  # 시작 시간 기록
    response_queue = q.Queue()  # 비동기 응답을 저장할 큐

    # 요청 데이터 처리
    body = json.loads(event['body'])
    user_id = body['userRequest']['user']['id']
    utterance = body['userRequest']['utterance']

    print(f"[Medicine Callback] User ID: {user_id}")
    print(f"[Medicine Callback] Utterance: {utterance}")

    # "생각답변 확인하기" 요청 처리
    if utterance == "생각답변 확인하기":
        if user_id in response_map and not response_map[user_id].empty():
            response = response_map[user_id].get()
        else:
            response = {
                'version': '2.0',
                'template': {
                    'outputs': [
                        {'simpleText': {'text': '아직 답변이 준비되지 않았습니다. 잠시 후 다시 시도해주세요.'}}
                    ],
                    'quickReplies': [
                        {'action': 'message', 'label': '답변 확인하기', 'messageText': '생각답변 확인하기'}
                    ]
                }
            }
        return {"statusCode": 200, "body": json.dumps(response)}
    
    # LLM 및 임베딩 모델 초기화
    embedding_model = OpenAIEmbeddings(model='text-embedding-3-large')
    response_map[user_id] = q.Queue()

    # 비동기 처리 시작
    chat_thread = threading.Thread(
        target=generate_chat_talk,
        args=(user_id, utterance, embedding_model, response_queue)
    )
    chat_thread.start()

    print(f"시간차 >>>>>>> {time.time() - start_time}")
    # 3.5초 안에 응답 확인
    while time.time() - start_time < 3.5:
        if not response_map[user_id].empty():
            response = response_map[user_id].get()
            break
        time.sleep(0.01)
    else:
        # 타임아웃 응답 반환
        response = timeover()
        
    # 최종 응답 반환
    return {"statusCode": 200, "body": json.dumps(response)}

