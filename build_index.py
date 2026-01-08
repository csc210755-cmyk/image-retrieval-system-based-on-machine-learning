import os
import numpy as np
import faiss
from feature_extractor import FeatureExtractor
import tensorflow as tf
import gc

# Suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
tf.get_logger().setLevel('ERROR')

def build_index(dataset_path, index_path):
    """
    Extract features from all images in the dataset and build a FAISS index.
    """
    image_files = sorted([os.path.join(dataset_path, f) for f in os.listdir(dataset_path) 
                   if f.endswith(('.jpg', '.jpeg', '.png', '.jfif'))])

    # Extract features for each image
    features_list = []
    valid_image_files = []
    extractor = FeatureExtractor()

    for idx, img_path in enumerate(image_files, 1):
        try:
            features = extractor.extract(img_path)
            features_list.append(features)
            valid_image_files.append(img_path)
            if idx % 10 == 0:
                print(f"Processing images... {idx}/{len(image_files)}")
                gc.collect()  # Force garbage collection
        except Exception as e:
            print(f"Error processing {img_path}: {e}")

    # Convert to numpy array
    features_array = np.array(features_list).astype('float32')

    # Build FAISS index
    dimension = features_array.shape[1]
    index = faiss.IndexFlatL2(dimension)  # Using L2 distance (Euclidean)
    index.add(features_array)

    # Save the index and the list of image files
    os.makedirs(os.path.dirname(index_path), exist_ok=True)
    faiss.write_index(index, index_path)
    np.save(os.path.join(os.path.dirname(index_path), 'image_files.npy'), valid_image_files)

    print(f"Index built with {len(valid_image_files)} images.")

if __name__ == '__main__':
    DATASET_PATH = 'images'  # Path to images directory
    INDEX_PATH = 'data/indexes/faiss_index.index'
    build_index(DATASET_PATH, INDEX_PATH)