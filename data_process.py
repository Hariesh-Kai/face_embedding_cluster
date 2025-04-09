import os
import cv2
import glob2
import numpy as np
from PIL import Image
from tqdm import tqdm
from dotenv import load_dotenv
from facenet_pytorch import MTCNN

# Load any environment configs (like custom paths)
load_dotenv()

# Fallback paths in case env vars are missing
RAW_IMAGES_DIR = os.getenv("DATASET_DIR", "D:/Job/Placement/tango/dataset/tango-cv-assessment-dataset")
PROCESSED_IMAGES_DIR = os.getenv("PROCESSED_DIR", "D:/Job/Placement/tango/dataset/processed_images")

# Load all image files (recursively) and resize them
def fetch_and_resize_images(root_dir, target_size=(224, 224)):
    all_paths = glob2.glob(os.path.join(root_dir, "**/*.jpg"))
    loaded_images = []

    print(f"\n🔍 Loading images from: {root_dir}")
    for img_path in tqdm(all_paths, desc="Processing images"):
        img = cv2.imread(img_path)
        if img is not None:
            resized = cv2.resize(img, target_size)
            loaded_images.append((img_path, resized))
    return loaded_images

# Set up face detector (MTCNN from facenet-pytorch)
face_detector = MTCNN(keep_all=False)

def detect_face(tensor_image):
    """
    Detect a single face in the image (if any).
    """
    return face_detector(tensor_image)

def write_images_to_disk(image_list, destination_dir):
    """
    Save all processed images to a specified directory.
    """
    os.makedirs(destination_dir, exist_ok=True)
    print(f"💾 Saving processed images to: {destination_dir}")

    for img_path, img in image_list:
        filename = os.path.basename(img_path)
        save_path = os.path.join(destination_dir, filename)
        cv2.imwrite(save_path, img)

if __name__ == "__main__":
    print("📦 Starting image processing pipeline...\n")
    dataset = fetch_and_resize_images(RAW_IMAGES_DIR)
    write_images_to_disk(dataset, PROCESSED_IMAGES_DIR)
    print("\n✅ Image processing complete.")
