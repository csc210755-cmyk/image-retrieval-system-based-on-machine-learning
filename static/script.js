document.addEventListener('DOMContentLoaded', function() {
    const uploadArea = document.getElementById('uploadArea');
    const imageInput = document.getElementById('imageInput');
    const previewContainer = document.getElementById('previewContainer');
    const previewImage = document.getElementById('previewImage');
    const resultsGrid = document.getElementById('resultsGrid');
    
    // Drag and drop functionality
    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.style.background = '#f0f0f0';
    });
    
    uploadArea.addEventListener('dragleave', () => {
        uploadArea.style.background = '';
    });
    
    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.style.background = '';
        
        if (e.dataTransfer.files.length) {
            handleImageUpload(e.dataTransfer.files[0]);
        }
    });
    
    // File input change
    imageInput.addEventListener('change', (e) => {
        if (e.target.files.length) {
            handleImageUpload(e.target.files[0]);
        }
    });
    
    function handleImageUpload(file) {
        if (!file.type.match('image.*')) {
            alert('Please upload an image file');
            return;
        }
        
        const reader = new FileReader();
        reader.onload = function(e) {
            previewImage.src = e.target.result;
            previewContainer.style.display = 'block';
        };
        reader.readAsDataURL(file);
    }
});

async function searchSimilar() {
    const fileInput = document.getElementById('imageInput');
    const resultsGrid = document.getElementById('resultsGrid');
    
    if (!fileInput.files.length) {
        alert('Please select an image first');
        return;
    }
    
    const formData = new FormData();
    formData.append('image', fileInput.files[0]);
    
    try {
        resultsGrid.innerHTML = '<p class="loading">Searching for similar images...</p>';
        
        const response = await fetch('/upload', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (data.success) {
            displayResults(data.similar_images);
        } else {
            resultsGrid.innerHTML = `<p class="error">Error: ${data.error}</p>`;
        }
    } catch (error) {
        resultsGrid.innerHTML = `<p class="error">Network error: ${error.message}</p>`;
    }
}

function displayResults(images) {
    const resultsGrid = document.getElementById('resultsGrid');
    
    if (!images.length) {
        resultsGrid.innerHTML = '<p class="no-results">No similar images found</p>';
        return;
    }
    
    resultsGrid.innerHTML = images.map(img => `
        <div class="result-item">
            <img src="/static/images/${img.path.split('/').pop()}" 
                 alt="Similar image"
                 onerror="this.src='https://via.placeholder.com/200x200?text=Image+Not+Found'">
            <div class="result-info">
                <span class="similarity">Similarity: ${(1 - img.distance).toFixed(3)}</span>
            </div>
        </div>
    `).join('');
}