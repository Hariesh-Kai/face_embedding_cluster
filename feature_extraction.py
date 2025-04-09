import os
import cv2
import glob2
import pickle
import numpy as np
from tqdm import tqdm
from dotenv import load_dotenv

import torch
from torchvision import transforms
from facenet_pytorch import InceptionResnetV1

# Load environment variables (like paths)
load_dotenv()

# Directory where preprocessed images are stored
PROCESSED_DIR = os.getenv("PROCESSED_DIR", "D:/Job/Placement/tango/dataset/processed_images")

# Path where embeddings will be saved
EMBEDDINGS_FILE = os.getenv("EMBEDDINGS_FILE", "D:/Job/Placement/tango/pkl/embeddings.pkl")

# Use GPU if available, otherwise fall back to CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load pretrained face recognition model (VGGFace2 variant)
model = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# Image preprocessing: resize, normalize, convert to tensor
preprocess_pipeline = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

def get_face_embedding(image_path):
    """
    Given an image path, returns its 512-d embedding using InceptionResnetV1.
    """
    img = cv2.imread(image_path)
    if img is None:
        return None
    
    # Convert from BGR (OpenCV) to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Apply transformation pipeline
    tensor = preprocess_pipeline(img_rgb).unsqueeze(0).to(device)

    # Extract embedding without tracking gradients
    with torch.no_grad():
        embedding = model(tensor)

    return embedding.cpu().numpy().flatten()

def main():
    print(f"\n🔍 Scanning for processed images in: {PROCESSED_DIR}")
    image_files = glob2.glob(os.path.join(PROCESSED_DIR, "**/*.jpg"), recursive=True)
    
    print(f"📈 Found {len(image_files)} images. Starting embedding extraction...\n")
    embedding_map = {}

    for path in tqdm(image_files, desc="Extracting embeddings"):
        emb = get_face_embedding(path)
        if emb is not None:
            embedding_map[path] = emb

    print(f"\n💾 Saving {len(embedding_map)} embeddings to file...")
    with open(EMBEDDINGS_FILE, 'wb') as f:
        pickle.dump(embedding_map, f)

    print(f"\n✅ Embeddings saved to: {EMBEDDINGS_FILE}\n")

if __name__ == "__main__":
    main()
