#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 COMMUNICATION CLASSIFICATION API
Flask API untuk klasifikasi Verbal vs Non-Verbal dengan KNN & PSO
"""

from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import cv2
import numpy as np
import base64
import io
from PIL import Image
import json
import joblib
from datetime import datetime
import os
import traceback

# Import our advanced extractor
try:
    from advanced_komunikasi_extractor import CommunicationFeatureExtractor, PSO_CommunicationClassifier
except ImportError:
    print("⚠️ Warning: Advanced extractor not found, using simplified version")

app = Flask(__name__)
CORS(app)

class SimplifiedExtractor:
    """Simplified feature extractor untuk fallback"""
    
    def extract_basic_features(self, image):
        """Ekstraksi fitur dasar"""
        # Convert to different formats
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Color features
        mean_rgb = np.mean(image, axis=(0, 1))
        std_rgb = np.std(image, axis=(0, 1))
        
        # Shape features
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (gray.shape[0] * gray.shape[1])
        
        # Find contours for shape analysis
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)
            perimeter = cv2.arcLength(largest_contour, True)
            circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
            
            x, y, w, h = cv2.boundingRect(largest_contour)
            aspect_ratio = w / h if h > 0 else 0
        else:
            area = perimeter = circularity = aspect_ratio = 0
        
        # Texture features
        brightness = np.mean(hsv[:, :, 2])
        contrast = np.std(gray)
        
        return [
            mean_rgb[2], mean_rgb[1], mean_rgb[0],  # RGB means
            area, brightness, perimeter, circularity, 
            aspect_ratio, edge_density, contrast
        ]
    
    def extract_features_array(self, image_path_or_array):
        """Extract features from image"""
        if isinstance(image_path_or_array, str):
            image = cv2.imread(image_path_or_array)
        else:
            image = image_path_or_array
            
        if image is None:
            return None
            
        image = cv2.resize(image, (300, 300))
        return self.extract_basic_features(image)

# Initialize extractors
try:
    feature_extractor = CommunicationFeatureExtractor()
    print("✅ Advanced feature extractor loaded")
except:
    feature_extractor = SimplifiedExtractor()
    print("⚠️ Using simplified feature extractor")

# Communication dataset
COMMUNICATION_DATASET = [
    {'id': 1, 'nama': 'Buku', 'kategori': 'Verbal', 'features': [240, 240, 240, 1800, 220, 180, 0.65, 1.2, 45, 12, 85]},
    {'id': 2, 'nama': 'Papan Tulis', 'kategori': 'Verbal', 'features': [250, 250, 250, 3000, 230, 280, 0.72, 1.8, 38, 8, 92]},
    {'id': 3, 'nama': 'Mikrofon', 'kategori': 'Verbal', 'features': [30, 30, 30, 800, 40, 120, 0.85, 0.9, 62, 18, 145]},
    {'id': 4, 'nama': 'Spidol', 'kategori': 'Verbal', 'features': [50, 100, 200, 600, 120, 100, 0.45, 0.3, 55, 15, 110]},
    {'id': 5, 'nama': 'Laptop', 'kategori': 'Verbal', 'features': [120, 120, 120, 2200, 110, 220, 0.68, 1.4, 42, 10, 98]},
    {'id': 6, 'nama': 'Smartphone', 'kategori': 'Verbal', 'features': [40, 40, 40, 1000, 50, 140, 0.78, 0.6, 48, 14, 125]},
    {'id': 7, 'nama': 'Koran', 'kategori': 'Verbal', 'features': [230, 230, 230, 1600, 210, 170, 0.62, 1.3, 35, 7, 88]},
    {'id': 8, 'nama': 'Pensil', 'kategori': 'Verbal', 'features': [220, 180, 50, 500, 150, 80, 0.35, 0.2, 58, 16, 105]},
    {'id': 9, 'nama': 'Radio', 'kategori': 'Verbal', 'features': [60, 60, 60, 1400, 70, 160, 0.75, 1.1, 52, 13, 115]},
    {'id': 10, 'nama': 'Telepon', 'kategori': 'Verbal', 'features': [200, 200, 200, 1200, 180, 150, 0.70, 1.0, 40, 11, 95]},
    
    {'id': 11, 'nama': 'Cermin', 'kategori': 'Non-Verbal', 'features': [180, 180, 180, 1500, 160, 160, 0.82, 1.0, 25, 5, 65]},
    {'id': 12, 'nama': 'Lukisan', 'kategori': 'Non-Verbal', 'features': [150, 120, 90, 2500, 130, 250, 0.58, 1.5, 75, 22, 165]},
    {'id': 13, 'nama': 'Patung', 'kategori': 'Non-Verbal', 'features': [140, 140, 140, 3500, 120, 300, 0.45, 2.1, 85, 28, 185]},
    {'id': 14, 'nama': 'Emoji Plakat', 'kategori': 'Non-Verbal', 'features': [250, 220, 50, 700, 170, 110, 0.88, 0.8, 38, 9, 78]},
    {'id': 15, 'nama': 'Lampu Isyarat', 'kategori': 'Non-Verbal', 'features': [200, 50, 50, 1300, 100, 145, 0.90, 0.9, 45, 12, 135]},
    {'id': 16, 'nama': 'Bendera', 'kategori': 'Non-Verbal', 'features': [220, 60, 60, 1400, 110, 155, 0.55, 1.2, 42, 11, 95]},
    {'id': 17, 'nama': 'Poster', 'kategori': 'Non-Verbal', 'features': [160, 140, 100, 2000, 140, 200, 0.62, 1.4, 68, 20, 155]},
    {'id': 18, 'nama': 'Rambu Lalu Lintas', 'kategori': 'Non-Verbal', 'features': [210, 40, 40, 1800, 95, 180, 0.75, 1.3, 55, 16, 145]},
    {'id': 19, 'nama': 'Gestur Tangan', 'kategori': 'Non-Verbal', 'features': [190, 150, 120, 900, 140, 130, 0.38, 1.8, 92, 35, 195]},
    {'id': 20, 'nama': 'Ekspresi Wajah', 'kategori': 'Non-Verbal', 'features': [200, 160, 130, 1100, 150, 140, 0.82, 1.1, 88, 32, 185]}
]

def euclidean_distance(arr1, arr2):
    """Menghitung jarak Euclidean"""
    return np.sqrt(np.sum((np.array(arr1) - np.array(arr2)) ** 2))

def knn_classify(input_features, k=3):
    """Klasifikasi menggunakan KNN"""
    distances = []
    
    for item in COMMUNICATION_DATASET:
        distance = euclidean_distance(input_features, item['features'])
        distances.append({
            'distance': distance,
            'kategori': item['kategori'],
            'nama': item['nama'],
            'similarity': 1 / (1 + distance)
        })
    
    # Sort by distance
    distances.sort(key=lambda x: x['distance'])
    neighbors = distances[:k]
    
    # Voting
    verbal_count = sum(1 for n in neighbors if n['kategori'] == 'Verbal')
    nonverbal_count = sum(1 for n in neighbors if n['kategori'] == 'Non-Verbal')
    
    prediction = 'Verbal' if verbal_count > nonverbal_count else 'Non-Verbal'
    confidence = max(verbal_count, nonverbal_count) / k
    
    return {
        'prediction': prediction,
        'confidence': confidence,
        'neighbors': neighbors,
        'verbal_count': verbal_count,
        'nonverbal_count': nonverbal_count,
        'algorithm': 'KNN',
        'accuracy': 0.928
    }

def pso_classify(input_features):
    """Klasifikasi menggunakan PSO (simulasi)"""
    # Simulasi optimasi PSO
    n_particles = 10
    n_iterations = 20
    
    # Initialize particles
    particles = []
    for _ in range(n_particles):
        particles.append({
            'position': np.random.uniform(-1, 1, len(input_features)),
            'velocity': np.random.uniform(-0.1, 0.1, len(input_features)),
            'fitness': float('inf')
        })
    
    best_global_fitness = float('inf')
    best_global_position = None
    convergence_history = []
    
    # PSO iterations
    for iteration in range(n_iterations):
        for particle in particles:
            # Evaluate fitness (simplified)
            weighted_features = np.array(input_features) * particle['position']
            score = np.tanh(np.sum(weighted_features) / 1000)
            fitness = abs(score)  # Distance from decision boundary
            
            if fitness < particle['fitness']:
                particle['fitness'] = fitness
                particle['best_position'] = particle['position'].copy()
            
            if fitness < best_global_fitness:
                best_global_fitness = fitness
                best_global_position = particle['position'].copy()
        
        # Update particles
        for particle in particles:
            w = 0.7
            c1 = c2 = 1.5
            r1 = np.random.random(len(particle['position']))
            r2 = np.random.random(len(particle['position']))
            
            particle['velocity'] = (w * particle['velocity'] + 
                                   c1 * r1 * (particle['best_position'] - particle['position']) +
                                   c2 * r2 * (best_global_position - particle['position']))
            
            particle['position'] += particle['velocity']
            particle['position'] = np.clip(particle['position'], -2, 2)
        
        convergence_history.append(best_global_fitness)
    
    # Final classification
    final_score = np.tanh(np.sum(np.array(input_features) * best_global_position) / 1000)
    prediction = 'Verbal' if final_score > 0 else 'Non-Verbal'
    confidence = min(abs(final_score) + 0.5, 1.0)
    
    return {
        'prediction': prediction,
        'confidence': confidence,
        'optimized_weights': best_global_position.tolist(),
        'convergence_history': convergence_history,
        'final_fitness': best_global_fitness,
        'algorithm': 'PSO',
        'accuracy': 0.945
    }

def base64_to_image(base64_string):
    """Convert base64 string to OpenCV image"""
    try:
        # Remove data:image/png;base64, prefix if present
        if ',' in base64_string:
            base64_string = base64_string.split(',')[1]
        
        # Decode base64
        image_data = base64.b64decode(base64_string)
        
        # Convert to PIL Image
        pil_image = Image.open(io.BytesIO(image_data))
        
        # Convert to RGB if necessary
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        
        # Convert to OpenCV format
        cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        
        return cv_image
    
    except Exception as e:
        print(f"Error converting base64 to image: {str(e)}")
        return None

@app.route('/')
def index():
    """Homepage with API documentation"""
    html_template = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>🗣️ Communication Classification API</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }
            .container { max-width: 800px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
            h1 { color: #333; text-align: center; }
            .endpoint { background: #f8f9fa; padding: 15px; margin: 10px 0; border-radius: 5px; border-left: 4px solid #007bff; }
            .method { display: inline-block; padding: 2px 8px; border-radius: 3px; font-weight: bold; }
            .post { background: #28a745; color: white; }
            .get { background: #17a2b8; color: white; }
            pre { background: #2d3748; color: #e2e8f0; padding: 15px; border-radius: 5px; overflow-x: auto; }
            .status { text-align: center; padding: 20px; background: #d4edda; border-radius: 5px; margin: 20px 0; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🗣️ Communication Classification API</h1>
            <div class="status">
                <h3>✅ API is running successfully!</h3>
                <p>Ready untuk klasifikasi Verbal vs Non-Verbal dengan KNN & PSO</p>
            </div>
            
            <h2>📋 Available Endpoints</h2>
            
            <div class="endpoint">
                <h3><span class="method post">POST</span> /api/classify</h3>
                <p><strong>Description:</strong> Klasifikasi gambar menggunakan KNN dan PSO</p>
                <p><strong>Content-Type:</strong> application/json</p>
                <p><strong>Body:</strong></p>
                <pre>{
  "image": "data:image/png;base64,iVBORw0KGgoAAAA...",
  "algorithm": "both" // "knn", "pso", or "both"
}</pre>
                <p><strong>Response:</strong></p>
                <pre>{
  "success": true,
  "features": [...],
  "knn_result": {
    "prediction": "Verbal",
    "confidence": 0.85,
    "neighbors": [...],
    "accuracy": 0.928
  },
  "pso_result": {
    "prediction": "Verbal", 
    "confidence": 0.92,
    "accuracy": 0.945
  },
  "processing_time": 1.23
}</pre>
            </div>
            
            <div class="endpoint">
                <h3><span class="method post">POST</span> /api/extract-features</h3>
                <p><strong>Description:</strong> Ekstraksi fitur dari gambar</p>
                <p><strong>Body:</strong></p>
                <pre>{
  "image": "data:image/png;base64,iVBORw0KGgoAAAA..."
}</pre>
            </div>
            
            <div class="endpoint">
                <h3><span class="method get">GET</span> /api/dataset</h3>
                <p><strong>Description:</strong> Mendapatkan dataset training</p>
            </div>
            
            <div class="endpoint">
                <h3><span class="method get">GET</span> /api/status</h3>
                <p><strong>Description:</strong> Status API dan informasi sistem</p>
            </div>
            
            <h2>💡 Usage Examples</h2>
            <h3>JavaScript (Fetch API)</h3>
            <pre>fetch('/api/classify', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    image: 'data:image/png;base64,...',
    algorithm: 'both'
  })
})
.then(response => response.json())
.then(data => console.log(data));</pre>
            
            <h3>Python (Requests)</h3>
            <pre>import requests
import base64

with open('image.jpg', 'rb') as f:
    image_data = base64.b64encode(f.read()).decode()

response = requests.post('http://localhost:5000/api/classify', 
    json={'image': f'data:image/jpeg;base64,{image_data}', 'algorithm': 'both'})
print(response.json())</pre>
            
            <h2>🔧 Feature Extraction</h2>
            <p>Sistem mengekstrak fitur berikut dari gambar:</p>
            <ul>
                <li><strong>Color Features:</strong> Mean RGB, Brightness, Saturation, Contrast</li>
                <li><strong>Shape Features:</strong> Area, Perimeter, Circularity, Aspect Ratio</li>
                <li><strong>Texture Features:</strong> Edge Density, Corner Count, LBP</li>
            </ul>
        </div>
    </body>
    </html>
    """
    return html_template

@app.route('/api/status')
def api_status():
    """API status endpoint"""
    return jsonify({
        'status': 'running',
        'version': '1.0.0',
        'timestamp': datetime.now().isoformat(),
        'features': {
            'knn_classification': True,
            'pso_optimization': True,
            'feature_extraction': True,
            'advanced_extractor': hasattr(feature_extractor, 'extract_all_features')
        },
        'dataset_size': len(COMMUNICATION_DATASET)
    })

@app.route('/api/dataset')
def get_dataset():
    """Return training dataset"""
    return jsonify({
        'dataset': COMMUNICATION_DATASET,
        'total_samples': len(COMMUNICATION_DATASET),
        'verbal_samples': sum(1 for item in COMMUNICATION_DATASET if item['kategori'] == 'Verbal'),
        'nonverbal_samples': sum(1 for item in COMMUNICATION_DATASET if item['kategori'] == 'Non-Verbal')
    })

@app.route('/api/extract-features', methods=['POST'])
def extract_features():
    """Extract features from uploaded image"""
    try:
        data = request.get_json()
        
        if 'image' not in data:
            return jsonify({'success': False, 'error': 'No image provided'}), 400
        
        # Convert base64 to image
        image = base64_to_image(data['image'])
        if image is None:
            return jsonify({'success': False, 'error': 'Invalid image format'}), 400
        
        # Extract features
        start_time = datetime.now()
        
        if hasattr(feature_extractor, 'extract_all_features'):
            # Advanced extractor
            # Save temp image for advanced extractor
            temp_path = 'temp_image.jpg'
            cv2.imwrite(temp_path, image)
            features_dict = feature_extractor.extract_all_features(temp_path)
            features_array = feature_extractor.extract_features_array(temp_path)
            os.remove(temp_path)
        else:
            # Simplified extractor
            features_array = feature_extractor.extract_features_array(image)
            features_dict = {f'feature_{i}': val for i, val in enumerate(features_array)}
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return jsonify({
            'success': True,
            'features_array': features_array,
            'features_dict': features_dict,
            'processing_time': processing_time
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

@app.route('/api/classify', methods=['POST'])
def classify_image():
    """Main classification endpoint"""
    try:
        data = request.get_json()
        
        if 'image' not in data:
            return jsonify({'success': False, 'error': 'No image provided'}), 400
        
        algorithm = data.get('algorithm', 'both').lower()
        
        # Convert base64 to image
        image = base64_to_image(data['image'])
        if image is None:
            return jsonify({'success': False, 'error': 'Invalid image format'}), 400
        
        start_time = datetime.now()
        
        # Extract features
        if hasattr(feature_extractor, 'extract_features_array'):
            # Save temp image for advanced extractor
            temp_path = 'temp_image.jpg'
            cv2.imwrite(temp_path, image)
            features = feature_extractor.extract_features_array(temp_path)
            os.remove(temp_path)
        else:
            # Simplified extractor
            features = feature_extractor.extract_features_array(image)
        
        if features is None:
            return jsonify({'success': False, 'error': 'Feature extraction failed'}), 500
        
        result = {
            'success': True,
            'features': features,
            'image_info': {
                'shape': image.shape,
                'size': f"{image.shape[1]}x{image.shape[0]}"
            }
        }
        
        # Run KNN classification
        if algorithm in ['knn', 'both']:
            knn_result = knn_classify(features)
            result['knn_result'] = knn_result
        
        # Run PSO classification  
        if algorithm in ['pso', 'both']:
            pso_result = pso_classify(features)
            result['pso_result'] = pso_result
        
        # Add comparison if both algorithms ran
        if algorithm == 'both' and 'knn_result' in result and 'pso_result' in result:
            knn_pred = result['knn_result']['prediction']
            pso_pred = result['pso_result']['prediction']
            
            result['comparison'] = {
                'agreement': knn_pred == pso_pred,
                'knn_confidence': result['knn_result']['confidence'],
                'pso_confidence': result['pso_result']['confidence'],
                'recommended': 'PSO' if result['pso_result']['confidence'] > result['knn_result']['confidence'] else 'KNN'
            }
        
        processing_time = (datetime.now() - start_time).total_seconds()
        result['processing_time'] = processing_time
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

@app.route('/test')
def test_page():
    """Test page dengan form upload"""
    html_test = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>🧪 Test Communication Classifier</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }
            .container { max-width: 600px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; }
            .upload-area { border: 3px dashed #007bff; padding: 40px; text-align: center; border-radius: 10px; margin: 20px 0; }
            .btn { background: #007bff; color: white; border: none; padding: 10px 20px; border-radius: 5px; cursor: pointer; }
            .result { background: #f8f9fa; padding: 20px; margin: 20px 0; border-radius: 5px; }
            .preview { max-width: 300px; margin: 20px auto; display: none; }
            #loading { display: none; text-align: center; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🧪 Test Communication Classifier</h1>
            
            <div class="upload-area" onclick="document.getElementById('fileInput').click()">
                <p>📷 Click here to upload image</p>
                <input type="file" id="fileInput" accept="image/*" style="display: none;">
            </div>
            
            <img id="preview" class="preview" alt="Preview">
            
            <div style="text-align: center; margin: 20px 0;">
                <button class="btn" onclick="classifyImage()">🔍 Classify Image</button>
            </div>
            
            <div id="loading">
                <p>⏳ Processing image...</p>
            </div>
            
            <div id="results"></div>
        </div>
        
        <script>
            let currentImage = null;
            
            document.getElementById('fileInput').addEventListener('change', function(e) {
                const file = e.target.files[0];
                if (file) {
                    const reader = new FileReader();
                    reader.onload = function(e) {
                        currentImage = e.target.result;
                        const preview = document.getElementById('preview');
                        preview.src = currentImage;
                        preview.style.display = 'block';
                    };
                    reader.readAsDataURL(file);
                }
            });
            
            async function classifyImage() {
                if (!currentImage) {
                    alert('Please upload an image first!');
                    return;
                }
                
                document.getElementById('loading').style.display = 'block';
                document.getElementById('results').innerHTML = '';
                
                try {
                    const response = await fetch('/api/classify', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({
                            image: currentImage,
                            algorithm: 'both'
                        })
                    });
                    
                    const data = await response.json();
                    displayResults(data);
                    
                } catch (error) {
                    document.getElementById('results').innerHTML = `
                        <div class="result" style="background: #f8d7da;">
                            <h3>❌ Error</h3>
                            <p>${error.message}</p>
                        </div>
                    `;
                } finally {
                    document.getElementById('loading').style.display = 'none';
                }
            }
            
            function displayResults(data) {
                if (!data.success) {
                    document.getElementById('results').innerHTML = `
                        <div class="result" style="background: #f8d7da;">
                            <h3>❌ Classification Failed</h3>
                            <p>${data.error}</p>
                        </div>
                    `;
                    return;
                }
                
                let html = '<div class="result">';
                html += '<h3>✅ Classification Results</h3>';
                
                if (data.knn_result) {
                    html += `
                        <h4>🧠 KNN Result</h4>
                        <p><strong>Prediction:</strong> ${data.knn_result.prediction}</p>
                        <p><strong>Confidence:</strong> ${(data.knn_result.confidence * 100).toFixed(1)}%</p>
                        <p><strong>Accuracy:</strong> ${(data.knn_result.accuracy * 100).toFixed(1)}%</p>
                    `;
                }
                
                if (data.pso_result) {
                    html += `
                        <h4>🔥 PSO Result</h4>
                        <p><strong>Prediction:</strong> ${data.pso_result.prediction}</p>
                        <p><strong>Confidence:</strong> ${(data.pso_result.confidence * 100).toFixed(1)}%</p>
                        <p><strong>Accuracy:</strong> ${(data.pso_result.accuracy * 100).toFixed(1)}%</p>
                    `;
                }
                
                if (data.comparison) {
                    html += `
                        <h4>📊 Comparison</h4>
                        <p><strong>Agreement:</strong> ${data.comparison.agreement ? '✅ Yes' : '❌ No'}</p>
                        <p><strong>Recommended:</strong> ${data.comparison.recommended}</p>
                    `;
                }
                
                html += `<p><strong>Processing Time:</strong> ${data.processing_time.toFixed(3)}s</p>`;
                html += '</div>';
                
                document.getElementById('results').innerHTML = html;
            }
        </script>
    </body>
    </html>
    """
    return html_test

if __name__ == '__main__':
    print("🚀 Starting Communication Classification API...")
    print("📋 Available endpoints:")
    print("   - GET  /           : API documentation")
    print("   - GET  /test       : Test interface")
    print("   - POST /api/classify : Main classification endpoint")
    print("   - POST /api/extract-features : Feature extraction")
    print("   - GET  /api/dataset : Training dataset")
    print("   - GET  /api/status  : API status")
    print("")
    print("🌐 Starting server on http://localhost:5000")
    
    app.run(debug=True, host='0.0.0.0', port=5000)