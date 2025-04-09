import os
import cv2
import pickle
import argparse
import numpy as np
import torch
from torchvision import transforms
from facenet_pytorch import InceptionResnetV1
from dotenv import load_dotenv
from tqdm import tqdm
import faiss

# Load environment variables (.env)
load_dotenv()

# === Paths from .env or fallback defaults ===
PROCESSED_DIR = os.getenv("PROCESSED_DIR", "D:/Job/Placement/tango/dataset/processed_images")
EMBEDDINGS_FILE = os.getenv("EMBEDDINGS_FILE", "D:/Job/Placement/tango/pkl/embeddings.pkl")
CLUSTERS_FILE = os.getenv("CLUSTERS_FILE", "D:/Job/Placement/tango/pkl/clusters.pkl")

# === Set up torch device ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Load the FaceNet model ===
model = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# === Preprocessing pipeline ===
preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

def extract_embedding(image_path):
    """
    Extract 512-d embedding from a given image using FaceNet.
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"⚠️ Could not read image: {image_path}")
        return None
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    tensor = preprocess(img_rgb).unsqueeze(0).to(device)
    
    with torch.no_grad():
        embedding = model(tensor)
    
    return embedding.cpu().numpy().flatten()

def load_pickle(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def build_faiss_index(embeddings_dict):
    """
    Build a FAISS index from the embeddings.
    Returns the FAISS index and the image paths.
    """
    image_paths = list(embeddings_dict.keys())
    X = np.array(list(embeddings_dict.values()), dtype=np.float32)
    
    index = faiss.IndexFlatL2(X.shape[1])
    index.add(X)
    return index, image_paths

def find_nearest_image(query_embedding, embeddings_dict):
    """
    Use FAISS to find the closest image in the dataset.
    """
    index, image_paths = build_faiss_index(embeddings_dict)
    query_embedding = np.expand_dims(query_embedding.astype('float32'), axis=0)
    
    _, indices = index.search(query_embedding, 1)
    return image_paths[indices[0][0]]

def retrieve_cluster_images(query_image_path):
    """
    Extracts cluster members of the closest match to the given query image.
    """
    # Step 1: Extract embedding from query image
    query_embedding = extract_embedding(query_image_path)
    if query_embedding is None:
        return []

    # Step 2: Load embeddings and cluster mappings
    embeddings = load_pickle(EMBEDDINGS_FILE)
    clusters = load_pickle(CLUSTERS_FILE)

    # Step 3: Find nearest image in the dataset
    nearest_image = find_nearest_image(query_embedding, embeddings)

    # Step 4: Identify which cluster this nearest image belongs to
    for cluster_id, image_list in clusters.items():
        if nearest_image in image_list:
            return image_list

    print("⚠️ No cluster found for the nearest image.")
    return []

def main():
    parser = argparse.ArgumentParser(description="Retrieve similar face images based on query image.")
    parser.add_argument("--query", type=str, required=True, help="Path to the query image")
    args = parser.parse_args()

    print(f"🔍 Query image: {args.query}")
    results = retrieve_cluster_images(args.query)

    if results:
        print("\n✅ Images found in the same cluster:")
        for path in results:
            print(f"📌 {path}")
    else:
        print("\n❌ No similar images found.")

if __name__ == "__main__":
    main()
