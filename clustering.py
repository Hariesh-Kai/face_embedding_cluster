import os
import pickle
import numpy as np
import hdbscan
from tqdm import tqdm
from dotenv import load_dotenv

# Load env variables from .env (if any)
load_dotenv()

# Paths for data files (fallback to default values if not set)
EMBEDDINGS_PATH = os.getenv("EMBEDDINGS_FILE", "D:/Job/Placement/tango/pkl/embeddings.pkl")
CLUSTERS_OUTPUT = os.getenv("CLUSTERS_FILE", "D:/Job/Placement/tango/pkl/clusters.pkl")

def load_embedding_data(path_to_embeddings):
    """
    Load the embedding data from a pickle file.
    Expected format: { image_path: embedding_vector }
    """
    with open(path_to_embeddings, 'rb') as f:
        data = pickle.load(f)
    return data

def run_hdbscan_clustering(embedding_dict, min_cluster_size=3):
    """
    Perform HDBSCAN clustering on the embeddings.
    Returns both the predicted cluster labels and the corresponding image paths.
    """
    img_paths = list(embedding_dict.keys())
    feature_matrix = np.array(list(embedding_dict.values()))

    # Initialize HDBSCAN model
    hdb = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, metric="euclidean")

    print("Fitting HDBSCAN on embeddings...")
    predicted_labels = hdb.fit_predict(feature_matrix)

    return predicted_labels, img_paths

def group_images_by_cluster(labels, paths):
    """
    Organize image paths by their assigned cluster labels.
    """
    cluster_map = {}
    for label, path in zip(labels, paths):
        if label not in cluster_map:
            cluster_map[label] = []
        cluster_map[label].append(path)
    return cluster_map

def save_cluster_mapping(clusters_dict, output_file):
    """
    Save the cluster mapping (label → image paths) to a pickle file.
    """
    with open(output_file, 'wb') as f:
        pickle.dump(clusters_dict, f)
    print(f"✅ Clusters saved at: {output_file}")

def main():
    print("\n📦 Loading embeddings...")
    embeddings_data = load_embedding_data(EMBEDDINGS_PATH)

    print("🔍 Performing clustering...")
    cluster_labels, image_paths = run_hdbscan_clustering(embeddings_data)

    print("🔗 Grouping images by clusters...")
    clusters = group_images_by_cluster(cluster_labels, image_paths)

    print("\n📊 Cluster Summary:")
    for cid, images in clusters.items():
        print(f"Cluster {cid}: {len(images)} image(s)")

    print("\n💾 Saving clusters...")
    save_cluster_mapping(clusters, CLUSTERS_OUTPUT)

if __name__ == '__main__':
    main()
