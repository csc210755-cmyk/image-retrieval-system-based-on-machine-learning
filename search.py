import numpy as np
import faiss

def search_similar_images(query_vector, index_path, dataset_path, top_k=10):
    """
    Search for similar images in the FAISS index.
    """
    # Load the index and the image files list
    index = faiss.read_index(index_path)
    image_files = np.load('image_files.npy', allow_pickle=True)

    # Ensure the query vector is normalized (if not already)
    norm = np.linalg.norm(query_vector)
    if norm != 0:
        query_vector = query_vector / norm

    # Reshape the query vector to 2D (1, dimension)
    query_vector = query_vector.reshape(1, -1).astype('float32')

    # Search in the index
    distances, indices = index.search(query_vector, top_k)

    # Get the paths of the similar images
    similar_images = []
    for idx in indices[0]:
        if idx < len(image_files):
            similar_images.append(image_files[idx])

    return similar_images