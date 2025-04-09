import os
import pickle
import numpy as np
from dotenv import load_dotenv
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import umap
import random

# Load environment variables
load_dotenv()

# Paths
EMBEDDINGS_FILE = os.getenv("EMBEDDINGS_FILE", "D:/Job/Placement/tango/pkl/embeddings.pkl")
CLUSTERS_FILE = os.getenv("CLUSTERS_FILE", "D:/Job/Placement/tango/pkl/clusters.pkl")

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def prepare_data():
    embeddings = load_pickle(EMBEDDINGS_FILE)
    clusters = load_pickle(CLUSTERS_FILE)

    X = []
    labels = []

    print("🔍 Gathering embeddings per cluster...")
    for cluster_id, image_paths in tqdm(clusters.items(), desc="Clusters"):
        for img_path in image_paths:
            emb = embeddings.get(img_path)
            if emb is not None:
                X.append(emb)
                labels.append(str(cluster_id))  # Make labels string for plotting
            else:
                print(f"⚠️ Embedding not found for: {img_path}")

    return np.array(X), labels

def plot_clusters_2d(X, labels, title="2D UMAP Clusters", save_path=None):
    print("🧠 Running UMAP dimensionality reduction...")
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='euclidean', random_state=42)
    embedding_2d = reducer.fit_transform(X)

    plt.figure(figsize=(12, 8))

    num_clusters = len(set(labels))
    palette = sns.color_palette("tab10") if num_clusters <= 10 else sns.color_palette("husl", num_clusters)

    sns.scatterplot(
        x=embedding_2d[:, 0], y=embedding_2d[:, 1],
        hue=labels,
        palette=palette,
        s=15,
        linewidth=0,
        alpha=0.8
    )

    plt.title(title)
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=1, fontsize='small', title="Cluster ID")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"📸 Plot saved to: {save_path}")
    else:
        plt.show()

def main():
    print("📊 Preparing data for visualization...")
    X, labels = prepare_data()

    if len(X) == 0:
        print("❌ No embeddings available for plotting.")
        return

    print(f"✅ Ready to visualize {len(X)} points across {len(set(labels))} clusters.")
    plot_clusters_2d(X, labels)

if __name__ == "__main__":
    main()
