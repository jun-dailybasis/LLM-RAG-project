import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from openai import OpenAI
import os

# Load the dataset
def load_dataset(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    df = pd.read_excel(file_path)

    # Rename columns for consistency
    df.rename(columns={
        "품목일련번호": "product_serial_number",
        "제품명": "product_name",
        "업체명": "company_name",
        "주성분": "active_ingredient",
        "이 약의 효능은 무엇입니까?": "drug_effect",
        "이 약은 어떻게 사용합니까?": "usage_instructions",
        "이 약을 사용하기 전에 반드시 알아야 할 내용은 무엇입니가?": "precautions_before_use",
        "이 약의 사용상 주의사항은 무엇입니까?": "usage_warnings",
        "이 약을 사용하는 동안 주의해야 할 약 또는 음식은 무엇입니까?": "drug_or_food_interactions",
        "이 약은 어떤 이상반응이 나타날 수 있습니까?": "potential_side_effects",
        "이 약은 어떻게 보관해야 합니까?": "storage_instructions",
        "공개일자": "release_date",
        "수정일자": "last_modified_date"
    }, inplace=True)
    return df

# Generate embeddings using OpenAI
def get_embedding_vectors(df, column, page_size=100):
    print()
    print(f'Get embedding vectors for {column} via OpenAI...')

    client = OpenAI()
    embeddings = []

    texts = df[column].fillna('').tolist()

    for i in range(0, len(df), page_size):
        print(f'- index {i}')

        resp = client.embeddings.create(
            model="text-embedding-3-large",
            input=texts[i: i + page_size],
        )

        embeddings += [x.embedding for x in resp.data]

    embeddings = np.array(embeddings)

    print(f" - Embedding Shape : {embeddings.shape}")

    return embeddings

# (선택) 최적 k 찾기
def find_optimal_clusters(data, max_k):
    """
    data: (예) 임베딩 벡터 리스트 혹은 numpy 배열
    max_k: 시도할 최대 k 값
    """
    silhouette_scores = []
    for k in range(2, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
        cluster_labels = kmeans.fit_predict(data)
        silhouette_avg = silhouette_score(data, cluster_labels)
        silhouette_scores.append((k, silhouette_avg))
        print(f"Clusters: {k}, Silhouette Score: {silhouette_avg:.4f}")
    return silhouette_scores

def cluster_data(df, embeds, k=5):
    """
    K-Means 군집화를 수행하여 df에 'cluster' 컬럼을 추가하고,
    각 문서와 그 군집의 중심(center) 간 dot product를 'similarity'로 계산해 저장.

    Parameters
    ----------
    df : pd.DataFrame
        문서(또는 아이템) 정보가 담긴 DataFrame (embeds와 인덱스 동일 가정)
    embeds : np.ndarray
        각 문서(또는 아이템)의 임베딩 벡터 배열 (df와 행 인덱스 일치)
    k : int
        K-Means로 군집화할 클러스터 개수

    Returns
    -------
    df : pd.DataFrame
        cluster와 similarity 컬럼이 추가된 DataFrame
    """
    print(f"\n[cluster_data] Clustering into {k} clusters...")
    clusterer = KMeans(n_clusters=k, random_state=42, n_init='auto')
    labels = clusterer.fit_predict(embeds)

    # df에 cluster 레이블 저장
    df['cluster'] = labels

    print("[cluster_data] Calculating similarities (dot product) with cluster centers...")
    similarities = []
    for i in range(len(df)):
        label = labels[i]
        center = clusterer.cluster_centers_[label]
        # dot product를 similarity로 사용
        sim = np.dot(embeds[i], center)
        similarities.append(sim)

    df['similarity'] = similarities
    return df

def generate_cluster_summaries(df, text_column, max_tokens=10000):
    """
    각 클러스터(cluster)에서 similarity가 가장 높은 문서를 찾아, 
    text_column 내용을 OpenAI로 요약하는 함수.

    Parameters
    ----------
    df : pd.DataFrame
        'cluster'와 'similarity' 컬럼이 이미 있는 DataFrame
    text_column : str
        요약할 텍스트가 들어 있는 컬럼 이름
    max_tokens : int
        대표 문서가 너무 길 경우, 앞부분만 사용하기 위한 커트 기준 (문자 수)

    Returns
    -------
    None
        (화면에 요약 결과를 출력만 하고, df에는 저장하지 않음)
    """
    client = OpenAI()  # OpenAI 클라이언트 초기화

    print("\n[generate_cluster_summaries] Generating summaries from representative document in each cluster...\n")
    for cluster_id in sorted(df['cluster'].unique()):
        # 이 클러스터에 속한 문서들만 필터링
        cluster_data = df[df['cluster'] == cluster_id]

        # 해당 클러스터에서 similarity가 가장 큰 문서(대표 문서) 찾기
        cluster_data = cluster_data.sort_values('similarity', ascending=False)
        top_doc = cluster_data.iloc[0]

        # 대표 문서 텍스트가 너무 길면 일정 길이까지만 사용
        doc_text = str(top_doc[text_column]) if pd.notnull(top_doc[text_column]) else ''
        if len(doc_text) > max_tokens:
            doc_text = doc_text[:max_tokens]

        print(f"Cluster {cluster_id}: {len(cluster_data)} items")
        # 'title' 컬럼이 없을 수도 있으니, get() 사용
        print(f"- Representative Document Title: {top_doc.get('title', '(no title)')}")

        # 요약 요청
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "다음 텍스트를 500자 이내로 요약해줘."},
                {"role": "user", "content": doc_text}
            ]
        )
        summary = response.choices[0].message.content.strip()

        print(f"Summary:\n{summary}\n")

# ------------------------------------------------------------
# 사용 예시
if __name__ == '__main__':
    # 1) 데이터 불러오기
    file_path = "./edruginfo.xlsx"
    df = load_dataset(file_path)

    # 여러 topic 조합을 시도하기
    clustering_topics = {
        "topic_1": "drug_effect active_ingredient",
        "topic_2": "usage_warnings active_ingredient",
        "topic_3": "potential_side_effects active_ingredient"
    }

    for topic_name, combined_columns in clustering_topics.items():
        print(f"Processing {topic_name}...")

        # 2) 요약할 텍스트(= 여러 컬럼 결합)를 'combined_text'에 저장
        df['combined_text'] = df[[col.strip() for col in combined_columns.split()]].fillna('').agg(' '.join, axis=1)

        # 3) 임베딩 생성
        embeds = get_embedding_vectors(df, 'combined_text')

        # 4) 최적 k 찾기 (optional)
        optimal_clusters = find_optimal_clusters(embeds, 10)
        best_k = max(optimal_clusters, key=lambda x: x[1])[0]
        print(f"Best number of clusters for {topic_name}: {best_k}")

        # 5) 최적 k로 클러스터링
        df = cluster_data(df, embeds, k=best_k)

        # 6) 대표 문서만 골라 요약
        generate_cluster_summaries(df, 'combined_text')

        # 7) 결과 저장
        output_file = f"./drug_clusters_{topic_name}.csv"
        df.to_csv(output_file, index=False)
        print(f"Clustering results for {topic_name} saved to {output_file}.\n")