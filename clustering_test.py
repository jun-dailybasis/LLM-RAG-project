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

# Determine the optimal number of clusters
def find_optimal_clusters(data, max_k):
    silhouette_scores = []
    for k in range(2, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        cluster_labels = kmeans.fit_predict(data)
        silhouette_avg = silhouette_score(data, cluster_labels)
        silhouette_scores.append((k, silhouette_avg))
        print(f"Clusters: {k}, Silhouette Score: {silhouette_avg:.4f}")
    return silhouette_scores

# Summarize clusters using OpenAI
def generate_cluster_summaries(df, column, max_tokens=10000):
    client = OpenAI()
    for cluster_id in sorted(df['cluster'].unique()):
        cluster_data = df[df['cluster'] == cluster_id]
        combined_text = ' '.join(cluster_data[column].fillna('').tolist())

        # Limit the length of the combined text
        if len(combined_text) > max_tokens:
            combined_text = combined_text[:max_tokens]

        print(f"Cluster {cluster_id}: {len(cluster_data)} items")

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Summarize the following drug information in under 500 characters."},
                {"role": "user", "content": combined_text}
            ]
        )

        summary = response.choices[0].message.content.strip()
        print(f"Summary for Cluster {cluster_id}: {summary}\n")

if __name__ == '__main__':
    file_path = "./edruginfo.xlsx"
    df = load_dataset(file_path)

    clustering_topics = {
        "topic_1": "drug_effect active_ingredient",
        "topic_2": "usage_warnings active_ingredient",
        "topic_3": "potential_side_effects active_ingredient",
        "topic_4": "product_name active_ingredient"
    }

    for topic_name, combined_columns in clustering_topics.items():
        print(f"Processing {topic_name}...")

        # Combine relevant columns
        df['combined_text'] = df[[col.strip() for col in combined_columns.split()]].fillna('').agg(' '.join, axis=1)

        # Generate embeddings
        embeds = get_embedding_vectors(df, 'combined_text')

        # Find optimal clusters
        optimal_clusters = find_optimal_clusters(embeds, 10)

        # Choose the best number of clusters
        best_k = max(optimal_clusters, key=lambda x: x[1])[0]
        print(f"Best number of clusters for {topic_name}: {best_k}")

        # Perform clustering
        kmeans = KMeans(n_clusters=best_k, random_state=42)
        df['cluster'] = kmeans.fit_predict(embeds)

        # Generate summaries for each cluster
        generate_cluster_summaries(df, 'combined_text')

        # Save cluster-specific results
        output_file = f"./drug_clusters_{topic_name}.csv"
        df.to_csv(output_file, index=False)
        print(f"Clustering results for {topic_name} saved to {output_file}.")
