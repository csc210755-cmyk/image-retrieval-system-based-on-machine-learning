import faiss
import numpy as np
import pickle
import os

class FAISSIndex:
    def __init__(self, index_path='data/indexes/faiss_index.index', 
                 image_paths_path='data/indexes/image_files.npy'):
        self.index_path = index_path
        self.image_paths_path = image_paths_path
        
        if os.path.exists(index_path):
            self.index = faiss.read_index(index_path)
            # Try to load image paths from .npy file first, then .pkl
            if os.path.exists(image_paths_path):
                self.image_paths = list(np.load(image_paths_path, allow_pickle=True))
            elif os.path.exists(image_paths_path.replace('.npy', '.pkl')):
                with open(image_paths_path.replace('.npy', '.pkl'), 'rb') as f:
                    self.image_paths = pickle.load(f)
            else:
                self.image_paths = []
        else:
            self.index = None
            self.image_paths = []
    
    def build_index(self, features_list, image_paths):
        """Build FAISS index from extracted features"""
        # Convert to numpy array
        features_array = np.array(features_list).astype('float32')
        
        # Create FAISS index
        dimension = features_array.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(features_array)
        
        # Save index and image paths
        self.image_paths = image_paths
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        faiss.write_index(self.index, self.index_path)
        
        with open(self.image_paths_path, 'wb') as f:
            pickle.dump(image_paths, f)
    
    def search(self, query_features, k=10):
        """Search for similar images"""
        if self.index is None:
            raise ValueError("Index not built. Please build index first.")
        
        # Reshape and normalize query features
        query_features = query_features.reshape(1, -1).astype('float32')
        
        # Search
        distances, indices = self.index.search(query_features, k)
        
        # Return image paths and distances
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.image_paths):
                results.append({
                    'path': self.image_paths[idx],
                    'distance': float(distances[0][i])
                })
        
        return results