import tensorflow as tf
from tensorflow.keras.applications.vgg19 import VGG19, preprocess_input
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import os

# Suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')

class FeatureExtractor:
    def __init__(self):
        # Load VGG19 model without top layers
        base_model = VGG19(weights='imagenet', include_top=False)
        self.model = tf.keras.Model(
            inputs=base_model.input,
            outputs=base_model.get_layer('block5_pool').output
        )
        self.input_size = (224, 224)
    
    def preprocess(self, img_path):
        # Read and resize image using PIL
        img = Image.open(img_path).convert('RGB')
        img = img.resize(self.input_size)
        
        # Convert to numpy array and normalize
        img = np.array(img).astype('float32') / 255.0
        
        # Expand dimensions for model input
        img = np.expand_dims(img, axis=0)
        return img
    
    def extract(self, img_path):
        # Preprocess image
        img = self.preprocess(img_path)
        
        # Extract features with batch_size=1 to avoid Keras dataset issues
        features = self.model.predict(img, verbose=0, batch_size=1)
        
        # Flatten and normalize (L2 normalization)
        features = features.flatten()
        norm = np.linalg.norm(features)
        if norm != 0:
            features = features / norm
        
        return features