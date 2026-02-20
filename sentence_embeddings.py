import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
from collections import defaultdict
import pandas as pd

LAYER = 16
EXPERT = 1
W = 3

if __name__ == "__main__":
    # 1. Load your JSON results
    neuron_topics_path = f'results/clustered_neurons_layer{LAYER}_expert{EXPERT}_w{W}.json'
    neurons_top20_titles_path = f'results/neuron_top20_titles_layer{LAYER}_expert{EXPERT}_w{W}.csv'

    df = pd.read_csv(neurons_top20_titles_path)

    with open(neuron_topics_path, 'r') as f:
        data = json.load(f)

    # Extract just the specific niches (keeping track of their original neuron IDs)
    niches = []
    neuron_mapping = []
    seen_neurons = set()

    for item in data:
        for topic in item['topics']:
            if item['neuron_id'] not in seen_neurons:
                niches.append(topic['specific_niche'])
                neuron_mapping.append(item['neuron_id'])
                seen_neurons.add(item['neuron_id'])

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
    for niche, neuron_id, label in zip(niches, neuron_mapping, cluster_labels):
        clusters[label].append((neuron_id, niche))

    # Sort clusters by size (largest first) and print top 5
    sorted_clusters = sorted(clusters.items(), key=lambda x: len(x[1]), reverse=True)
    
    for cluster_id, items in sorted_clusters[:5]:
        print(f"\n--- Cluster {cluster_id} (Size: {len(items)}) ---")
        # Print unique items in the cluster with neuron IDs
        unique_items = list({(neuron_id, niche) for neuron_id, niche in items})[:10]
        for neuron_id, niche in unique_items:
            row = df[df['neuron_id'] == neuron_id] # get the row with this neuron_id
            output = f" - {neuron_id}: {niche} \t ({row['top1'].values[0]}, {row['top2'].values[0]}, {row['top3'].values[0]})"
            print(output[:100])