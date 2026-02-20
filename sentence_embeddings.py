import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
from collections import defaultdict

if __name__ == "__main__":
    # 1. Load your JSON results
    with open('clustered_neurons.json', 'r') as f:
        data = json.load(f)

    # Extract just the specific niches (keeping track of their original neuron IDs)
    niches = []
    neuron_mapping = []

    for item in data:
        for topic in item['topics']:
            niches.append(topic['specific_niche'])
            neuron_mapping.append(item['neuron_id'])

    # 2. Load a lightweight, fast embedding model
    print("Loading embedding model...")
    model = SentenceTransformer('all-MiniLM-L6-v2') 

    # 3. Convert the text phrases into numerical vectors
    print("Generating embeddings...")
    embeddings = model.encode(niches)

    # 4. Cluster the embeddings
    # distance_threshold: Lower = more strict/granular clusters. Higher = broader clusters.
    print("Clustering...")
    clustering_model = AgglomerativeClustering(
        n_clusters=None, 
        distance_threshold=1.5, # Tweak this value between 1.0 and 2.0
        linkage='ward'
    )
    clustering_model.fit(embeddings)
    cluster_labels = clustering_model.labels_

    # 5. Group and display the results
    clusters = defaultdict(list)
    for niche, label in zip(niches, cluster_labels):
        clusters[label].append(niche)

    # Print a sample of the clustered groups
    for cluster_id, items in list(clusters.items())[:5]:
        print(f"\n--- Cluster {cluster_id} (Size: {len(items)}) ---")
        # Print unique items in the cluster
        for item in list(set(items))[:5]: 
            print(f" - {item}")