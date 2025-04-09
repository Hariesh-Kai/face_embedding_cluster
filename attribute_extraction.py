import os
import cv2
import pickle
import numpy as np
from dotenv import load_dotenv
from tqdm import tqdm
from collections import defaultdict, Counter

# Load environment variables from a .env file (if available)
load_dotenv()

# File path for saved cluster data (adjusted via environment variables or default path)
CLUSTERS_PATH = os.getenv("CLUSTERS_FILE", "D:/Job/Placement/tango/pkl/clusters.pkl")

# Define some common clothing color references (BGR format)
BGR_COLOR_LABELS = {
    "black": np.array([0, 0, 0]),
    "white": np.array([255, 255, 255]),
    "red": np.array([0, 0, 255]),
    "green": np.array([0, 255, 0]),
    "blue": np.array([255, 0, 0]),
    "yellow": np.array([0, 255, 255]),
    "cyan": np.array([255, 255, 0]),
    "magenta": np.array([255, 0, 255]),
    "gray": np.array([128, 128, 128]),
    "brown": np.array([42, 42, 165])
}

def get_nearest_color_name(color_bgr):
    """
    Find the closest basic color name for a given BGR color.
    """
    closest_name = "unknown"
    smallest_distance = float("inf")

    for label, ref_bgr in BGR_COLOR_LABELS.items():
        distance = np.linalg.norm(color_bgr - ref_bgr)
        if distance < smallest_distance:
            smallest_distance = distance
            closest_name = label

    return closest_name

def find_main_color(image, num_colors=1):
    """
    Run k-means clustering to determine the dominant color in an image.
    """
    flat_pixels = image.reshape(-1, 3).astype(np.float32)

    # Set criteria: stop after max 100 iterations or move by epsilon
    kmeans_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1.0)

    _, labels, centers = cv2.kmeans(
        flat_pixels,
        num_colors,
        None,
        kmeans_criteria,
        10,
        cv2.KMEANS_RANDOM_CENTERS
    )

    label_counts = np.bincount(labels.flatten())
    dominant = centers[np.argmax(label_counts)]

    return dominant.astype(int)

def detect_clothing_color(image_file_path):
    """
    Load an image and extract dominant clothing color from the torso area.
    """
    image = cv2.imread(image_file_path)
    if image is None:
        return "unknown"  # Possibly a corrupted or missing file

    height, width = image.shape[:2]
    
    # Focus on the center region where clothing usually appears
    torso_region = image[height // 3 : 2 * height // 3, :]

    top_color = find_main_color(torso_region)
    return get_nearest_color_name(top_color)

def analyze_cluster_colors(cluster_data_path):
    """
    For each cluster of images, determine the most common dominant clothing color.
    """
    with open(cluster_data_path, "rb") as file:
        cluster_data = pickle.load(file)

    cluster_to_color = {}

    for cluster_id, img_list in tqdm(cluster_data.items(), desc="Processing clusters"):
        color_results = []

        for img_path in img_list:
            detected_color = detect_clothing_color(img_path)
            if detected_color != "unknown":
                color_results.append(detected_color)

        most_frequent_color = Counter(color_results).most_common(1)
        if most_frequent_color:
            cluster_to_color[cluster_id] = most_frequent_color[0][0]
        else:
            cluster_to_color[cluster_id] = "unknown"

    return cluster_to_color

def main():
    print("\n🧥 Identifying dominant clothing color for each detected person (by cluster)...")

    color_results = analyze_cluster_colors(CLUSTERS_PATH)

    print("\n🎯 Final Results:\n")
    for cid, color in color_results.items():
        print(f"Cluster {cid}: {color}")

if __name__ == "__main__":
    main()
