from flask import Flask, render_template, request, jsonify, send_from_directory
import os
from werkzeug.utils import secure_filename
from feature_extractor import FeatureExtractor
from faiss_index import FAISSIndex
import threading
import time

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB limit

# Lazy-loaded components
feature_extractor = None
faiss_index = None
index_mtime = None  # Track index modification time for real-time updates

def get_feature_extractor():
    global feature_extractor
    if feature_extractor is None:
        feature_extractor = FeatureExtractor()
    return feature_extractor

def get_faiss_index():
    global faiss_index, index_mtime
    index_path = 'data/indexes/faiss_index.index'
    
    # Check if index file has been modified (real-time reload)
    if os.path.exists(index_path):
        current_mtime = os.path.getmtime(index_path)
        if index_mtime != current_mtime:
            faiss_index = FAISSIndex(index_path=index_path)
            index_mtime = current_mtime
    
    if faiss_index is None:
        faiss_index = FAISSIndex(index_path=index_path)
        if os.path.exists(index_path):
            index_mtime = os.path.getmtime(index_path)
    
    return faiss_index

# Ensure directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No image selected'}), 400
    
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            # Extract features (lazy load)
            fe = get_feature_extractor()
            features = fe.extract(filepath)
            
            # Search for similar images (with real-time reload check)
            fi = get_faiss_index()
            similar_images = fi.search(features, k=10)
            
            return jsonify({
                'success': True,
                'similar_images': similar_images
            })
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    return jsonify({'error': 'Error processing image'}), 500

@app.route('/static/images/<path:filename>')
def serve_image(filename):
    return send_from_directory('images', filename)

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)