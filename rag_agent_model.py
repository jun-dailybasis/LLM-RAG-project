import os
import json
import pandas as pd
from typing import List

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain.docstore.document import Document
from langchain.schema import SystemMessage, HumanMessage

##############################################################################
# 1) CSV 및 JSON 로드 함수
##############################################################################
def load_csv_file(file_path, text_column="combined_text") -> List[Document]:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    df = pd.read_csv(file_path, encoding='utf-8')

    documents = []
    for idx, row in df.iterrows():
        content = str(row.get(text_column, ""))
        metadata = {
            "source_file": os.path.basename(file_path),
            "row_index": idx
        }
        doc = Document(page_content=content, metadata=metadata)
        documents.append(doc)
    return documents

def load_json_file(json_path) -> List[Document]:
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"File not found: {json_path}")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    documents = []
    for idx, item in enumerate(data):
        product_name = item.get("product_name", "")
        company_name = item.get("company_name", "")
        ingredients = item.get("active_ingredient_array", [])

        ingredients_str = ", ".join(ingredients)
        content = (
            f"제품명: {product_name}\n"
            f"업체명: {company_name}\n"
            f"주성분: {ingredients_str}"
        )

        metadata = {
            "source_file": os.path.basename(json_path),
            "row_index": idx
        }
        doc = Document(page_content=content, metadata=metadata)
        documents.append(doc)
    return documents

##############################################################################
# 2) 벡터스토어 빌드 함수
##############################################################################
def build_topic_vector_stores():
    embeddings = OpenAIEmbeddings(model='text-embedding-3-large')

    topic1_docs = load_csv_file("./drug_clusters_topic_1.csv")
    topic2_docs = load_csv_file("./drug_clusters_topic_2.csv")
    topic3_docs = load_csv_file("./drug_clusters_topic_3.csv")
    json_docs = load_json_file("./edruginfo_extracted.json")

    topic1_store = DocArrayInMemorySearch.from_documents(topic1_docs, embeddings)
    topic2_store = DocArrayInMemorySearch.from_documents(topic2_docs, embeddings)
    topic3_store = DocArrayInMemorySearch.from_documents(topic3_docs, embeddings)
    json_store = DocArrayInMemorySearch.from_documents(json_docs, embeddings)

    return {
        "topic1": topic1_store,
        "topic2": topic2_store,
        "topic3": topic3_store,
        "json":   json_store
    }

##############################################################################
# 3) LLM 모델 생성
##############################################################################
def create_llm_model():
    model = ChatOpenAI(
        model='gpt-4o-mini',
        temperature=0
    )
    return model

##############################################################################
# 4) RAG 쿼리 함수
##############################################################################
def query_rag(question, topic_stores: dict, model, k=2, top_n=3):
    # Step 1: JSON에서 제품 정보 검색
    json_results = topic_stores["json"].similarity_search(question, k=k)
    if not json_results:
        return "관련된 제품 정보를 찾을 수 없습니다."

    # Step 2: 관련 주성분 추출
    primary_doc = json_results[0]  # 가장 유사한 문서
    ingredients = primary_doc.page_content.split("\n")[-1].replace("주성분: ", "").split(", ")

    # Step 3: Topic 1, 2, 3에서 주성분 기반 추가 검색
    context_list = []
    for topic, store in topic_stores.items():
        if topic == "json":
            continue  # JSON은 이미 검색 완료
        for ingredient in ingredients:
            results = store.similarity_search(ingredient, k=1)
            if results:
                context_list.append(f"[Topic: {topic}]\n{results[0].page_content}\n")

    # Step 4: LLM에 전달할 컨텍스트 병합
    merged_context = "\n".join(context_list[:top_n])
    system_prompt = (
        "다음 정보를 참고해 사용자 질문에 답해주세요. "
        "만약 해당 내용이 부족하거나 확신이 없다면, 모른다고 답해주세요.\n\n"
        f"{merged_context}"
    )

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=question),
    ]
    response = model.invoke(messages)
    return response.content

##############################################################################
# 5) 메인 실행
##############################################################################
if __name__ == "__main__":
    topic_stores = build_topic_vector_stores()
    llm_model = create_llm_model()

    while True:
        question = input("질문을 입력해주세요 (종료하려면 'exit' 입력): ")
        if question.lower() == 'exit':
            print("종료합니다.")
            break
        print("\n==================================")
        print("Q:", question)
        answer = query_rag(question, topic_stores, llm_model, k=2, top_n=5)
        print("A:", answer)

 