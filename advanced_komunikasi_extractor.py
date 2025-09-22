#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🎯 ADVANCED COMMUNICATION FEATURE EXTRACTOR
Ekstraksi fitur canggih untuk klasifikasi Verbal vs Non-Verbal
dengan integrasi OpenCV, Deep Learning, dan Computer Vision
"""

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance
import matplotlib.pyplot as plt
import seaborn as sns
from skimage import feature, measure, filters
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import json
import os
from datetime import datetime

class CommunicationFeatureExtractor:
    def __init__(self):
        self.feature_names = [
            'mean_red', 'mean_green', 'mean_blue',
            'std_red', 'std_green', 'std_blue',
            'area', 'perimeter', 'circularity', 'aspect_ratio',
            'brightness', 'contrast', 'saturation',
            'edge_density', 'corner_count', 'texture_energy',
            'lbp_mean', 'lbp_std', 'glcm_contrast', 'glcm_homogeneity',
            'shape_complexity', 'symmetry_score'
        ]
        
    def extract_color_features(self, image):
        """Ekstraksi fitur warna RGB dan HSV"""
        # Convert to different color spaces
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        
        # RGB statistics
        mean_rgb = np.mean(image, axis=(0, 1))
        std_rgb = np.std(image, axis=(0, 1))
        
        # HSV statistics
        brightness = np.mean(hsv[:, :, 2])  # V channel
        saturation = np.mean(hsv[:, :, 1])  # S channel
        
        # Contrast using LAB
        contrast = np.std(lab[:, :, 0])  # L channel standard deviation
        
        return {
            'mean_red': mean_rgb[2],
            'mean_green': mean_rgb[1], 
            'mean_blue': mean_rgb[0],
            'std_red': std_rgb[2],
            'std_green': std_rgb[1],
            'std_blue': std_rgb[0],
            'brightness': brightness,
            'saturation': saturation,
            'contrast': contrast
        }
    
    def extract_shape_features(self, image):
        """Ekstraksi fitur bentuk dan geometri"""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Edge detection
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (gray.shape[0] * gray.shape[1])
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Get largest contour
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Calculate shape properties
            area = cv2.contourArea(largest_contour)
            perimeter = cv2.arcLength(largest_contour, True)
            
            # Circularity
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter * perimeter)
            else:
                circularity = 0
            
            # Bounding rectangle for aspect ratio
            x, y, w, h = cv2.boundingRect(largest_contour)
            aspect_ratio = w / h if h > 0 else 0
            
            # Shape complexity (convex hull ratio)
            hull = cv2.convexHull(largest_contour)
            hull_area = cv2.contourArea(hull)
            shape_complexity = area / hull_area if hull_area > 0 else 0
            
        else:
            area = perimeter = circularity = aspect_ratio = shape_complexity = 0
        
        # Corner detection
        corners = cv2.goodFeaturesToTrack(gray, maxCorners=100, qualityLevel=0.01, minDistance=10)
        corner_count = len(corners) if corners is not None else 0
        
        # Symmetry score (horizontal symmetry)
        height, width = gray.shape
        left_half = gray[:, :width//2]
        right_half = cv2.flip(gray[:, width//2:], 1)
        
        # Resize to match if needed
        min_width = min(left_half.shape[1], right_half.shape[1])
        left_half = left_half[:, :min_width]
        right_half = right_half[:, :min_width]
        
        symmetry_score = 1 - np.mean(np.abs(left_half.astype(float) - right_half.astype(float))) / 255
        
        return {
            'area': area,
            'perimeter': perimeter,
            'circularity': circularity,
            'aspect_ratio': aspect_ratio,
            'edge_density': edge_density,
            'corner_count': corner_count,
            'shape_complexity': shape_complexity,
            'symmetry_score': symmetry_score
        }
    
    def extract_texture_features(self, image):
        """Ekstraksi fitur tekstur menggunakan LBP dan GLCM"""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Local Binary Pattern
        radius = 3
        n_points = 8 * radius
        lbp = feature.local_binary_pattern(gray, n_points, radius, method='uniform')
        lbp_mean = np.mean(lbp)
        lbp_std = np.std(lbp)
        
        # Texture energy using Gabor filters
        texture_energy = 0
        for theta in [0, 45, 90, 135]:
            kernel = cv2.getGaborKernel((21, 21), 5, np.radians(theta), 2*np.pi*0.5, 0.5, 0, ktype=cv2.CV_32F)
            filtered = cv2.filter2D(gray, cv2.CV_8UC3, kernel)
            texture_energy += np.mean(filtered**2)
        texture_energy /= 4
        
        # GLCM features (simplified version)
        # Resize for computational efficiency
        small_gray = cv2.resize(gray, (64, 64))
        
        try:
            from skimage.feature import greycomatrix, greycoprops
            glcm = greycomatrix(small_gray, [1], [0, 45, 90, 135], 256, symmetric=True, normed=True)
            glcm_contrast = np.mean(greycoprops(glcm, 'contrast'))
            glcm_homogeneity = np.mean(greycoprops(glcm, 'homogeneity'))
        except:
            # Fallback if skimage not available
            glcm_contrast = np.std(gray)
            glcm_homogeneity = 1 / (1 + np.std(gray))
        
        return {
            'lbp_mean': lbp_mean,
            'lbp_std': lbp_std,
            'texture_energy': texture_energy,
            'glcm_contrast': glcm_contrast,
            'glcm_homogeneity': glcm_homogeneity
        }
    
    def extract_all_features(self, image_path):
        """Ekstraksi semua fitur dari gambar"""
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Cannot load image: {image_path}")
            
            # Resize for consistency
            image = cv2.resize(image, (300, 300))
            
            # Extract different types of features
            color_features = self.extract_color_features(image)
            shape_features = self.extract_shape_features(image)
            texture_features = self.extract_texture_features(image)
            
            # Combine all features
            all_features = {**color_features, **shape_features, **texture_features}
            
            return all_features
            
        except Exception as e:
            print(f"Error extracting features from {image_path}: {str(e)}")
            return None
    
    def extract_features_array(self, image_path):
        """Ekstraksi fitur dalam format array untuk machine learning"""
        features_dict = self.extract_all_features(image_path)
        if features_dict is None:
            return None
        
        # Convert to array in specific order
        return [features_dict[name] for name in self.feature_names]

class PSO_CommunicationClassifier:
    def __init__(self, n_particles=20, n_iterations=50):
        self.n_particles = n_particles
        self.n_iterations = n_iterations
        self.best_global_position = None
        self.best_global_fitness = float('inf')
        self.convergence_history = []
        
    def initialize_particles(self, n_features):
        """Inisialisasi partikel PSO"""
        particles = []
        for _ in range(self.n_particles):
            particle = {
                'position': np.random.uniform(-1, 1, n_features),
                'velocity': np.random.uniform(-0.1, 0.1, n_features),
                'best_position': None,
                'best_fitness': float('inf')
            }
            particles.append(particle)
        return particles
    
    def fitness_function(self, weights, X_train, y_train):
        """Fungsi fitness untuk evaluasi kualitas bobot"""
        try:
            # Apply weights to features
            weighted_features = X_train * weights
            
            # Simple neural network-like computation
            scores = np.tanh(np.sum(weighted_features, axis=1))
            predictions = (scores > 0).astype(int)
            
            # Calculate accuracy
            accuracy = np.mean(predictions == y_train)
            
            # Return negative accuracy (PSO minimizes)
            return 1 - accuracy
            
        except:
            return 1.0  # Worst fitness if error
    
    def update_particle(self, particle, global_best):
        """Update posisi dan kecepatan partikel"""
        w = 0.7  # Inertia weight
        c1 = 1.5  # Cognitive parameter
        c2 = 1.5  # Social parameter
        
        r1 = np.random.random(len(particle['position']))
        r2 = np.random.random(len(particle['position']))
        
        # Update velocity
        particle['velocity'] = (w * particle['velocity'] + 
                               c1 * r1 * (particle['best_position'] - particle['position']) +
                               c2 * r2 * (global_best - particle['position']))
        
        # Update position
        particle['position'] += particle['velocity']
        
        # Bound positions
        particle['position'] = np.clip(particle['position'], -2, 2)
    
    def optimize(self, X_train, y_train):
        """Optimasi menggunakan PSO"""
        n_features = X_train.shape[1]
        particles = self.initialize_particles(n_features)
        
        for iteration in range(self.n_iterations):
            for particle in particles:
                # Evaluate fitness
                fitness = self.fitness_function(particle['position'], X_train, y_train)
                
                # Update personal best
                if fitness < particle['best_fitness']:
                    particle['best_fitness'] = fitness
                    particle['best_position'] = particle['position'].copy()
                
                # Update global best
                if fitness < self.best_global_fitness:
                    self.best_global_fitness = fitness
                    self.best_global_position = particle['position'].copy()
            
            # Update particles
            for particle in particles:
                if particle['best_position'] is not None:
                    self.update_particle(particle, self.best_global_position)
            
            # Record convergence
            self.convergence_history.append(self.best_global_fitness)
            
            # Early stopping if converged
            if len(self.convergence_history) > 10:
                recent_improvement = (self.convergence_history[-10] - 
                                    self.convergence_history[-1])
                if recent_improvement < 0.001:
                    print(f"Converged at iteration {iteration}")
                    break
        
        return self.best_global_position, self.best_global_fitness
    
    def predict(self, X_test, weights):
        """Prediksi menggunakan bobot yang dioptimasi"""
        weighted_features = X_test * weights
        scores = np.tanh(np.sum(weighted_features, axis=1))
        predictions = (scores > 0).astype(int)
        confidence = np.abs(scores)
        return predictions, confidence

def create_sample_dataset():
    """Membuat dataset contoh untuk testing"""
    # Verbal communication objects
    verbal_data = [
        {'nama': 'Buku', 'kategori': 'Verbal', 'features': [240, 240, 240, 20, 20, 20, 1800, 180, 0.65, 1.2, 220, 85, 150, 12, 25, 100, 45, 15, 75, 0.8, 0.7, 0.6]},
        {'nama': 'Mikrofon', 'kategori': 'Verbal', 'features': [30, 30, 30, 15, 15, 15, 800, 120, 0.85, 0.9, 40, 145, 200, 18, 30, 120, 62, 20, 85, 0.7, 0.8, 0.5]},
        {'nama': 'Laptop', 'kategori': 'Verbal', 'features': [120, 120, 120, 25, 25, 25, 2200, 220, 0.68, 1.4, 110, 98, 180, 10, 40, 110, 42, 18, 70, 0.75, 0.9, 0.8]},
        {'nama': 'Smartphone', 'kategori': 'Verbal', 'features': [40, 40, 40, 10, 10, 10, 1000, 140, 0.78, 0.6, 50, 125, 190, 14, 35, 105, 48, 16, 80, 0.9, 0.85, 0.7]},
        {'nama': 'Papan Tulis', 'kategori': 'Verbal', 'features': [250, 250, 250, 30, 30, 30, 3000, 280, 0.72, 1.8, 230, 92, 170, 8, 20, 90, 38, 12, 65, 0.6, 0.6, 0.9]}
    ]
    
    # Non-verbal communication objects  
    nonverbal_data = [
        {'nama': 'Lukisan', 'kategori': 'Non-Verbal', 'features': [150, 120, 90, 40, 35, 30, 2500, 250, 0.58, 1.5, 130, 165, 220, 22, 50, 140, 75, 25, 95, 0.6, 0.4, 0.7]},
        {'nama': 'Patung', 'kategori': 'Non-Verbal', 'features': [140, 140, 140, 35, 35, 35, 3500, 300, 0.45, 2.1, 120, 185, 160, 28, 60, 150, 85, 30, 100, 0.5, 0.3, 0.8]},
        {'nama': 'Gestur Tangan', 'kategori': 'Non-Verbal', 'features': [190, 150, 120, 45, 40, 35, 900, 130, 0.38, 1.8, 140, 195, 210, 35, 70, 180, 92, 35, 110, 0.4, 0.2, 0.6]},
        {'nama': 'Rambu Lalu Lintas', 'kategori': 'Non-Verbal', 'features': [210, 40, 40, 30, 20, 20, 1800, 180, 0.75, 1.3, 95, 145, 250, 16, 45, 125, 55, 22, 90, 0.8, 0.7, 0.9]},
        {'nama': 'Bendera', 'kategori': 'Non-Verbal', 'features': [220, 60, 60, 35, 25, 25, 1400, 155, 0.55, 1.2, 110, 95, 200, 11, 40, 100, 42, 18, 75, 0.6, 0.5, 0.8]}
    ]
    
    return verbal_data + nonverbal_data

def main():
    """Fungsi utama untuk demonstrasi"""
    print("🎯 ADVANCED COMMUNICATION FEATURE EXTRACTOR")
    print("=" * 60)
    
    # Initialize extractor
    extractor = CommunicationFeatureExtractor()
    
    # Create sample dataset
    sample_data = create_sample_dataset()
    
    # Convert to training format
    X_train = np.array([item['features'] for item in sample_data])
    y_train = np.array([1 if item['kategori'] == 'Verbal' else 0 for item in sample_data])
    
    # Normalize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    print(f"📊 Dataset loaded: {len(sample_data)} samples")
    print(f"   - Verbal: {sum(y_train)} samples")
    print(f"   - Non-Verbal: {len(y_train) - sum(y_train)} samples")
    print(f"   - Features: {X_train.shape[1]} dimensions")
    
    # Test KNN Classifier
    print("\n🧠 Testing KNN Classifier...")
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train_scaled, y_train)
    knn_pred = knn.predict(X_train_scaled)
    knn_accuracy = accuracy_score(y_train, knn_pred)
    print(f"   KNN Accuracy: {knn_accuracy:.3f}")
    
    # Test PSO Classifier
    print("\n🔥 Testing PSO Optimizer...")
    pso = PSO_CommunicationClassifier(n_particles=20, n_iterations=30)
    best_weights, best_fitness = pso.optimize(X_train_scaled, y_train)
    pso_pred, pso_confidence = pso.predict(X_train_scaled, best_weights)
    pso_accuracy = accuracy_score(y_train, pso_pred)
    print(f"   PSO Accuracy: {pso_accuracy:.3f}")
    print(f"   Best Fitness: {best_fitness:.3f}")
    print(f"   Convergence: {len(pso.convergence_history)} iterations")
    
    # Save models and scaler
    print("\n💾 Saving models...")
    joblib.dump(knn, 'knn_communication_model.joblib')
    joblib.dump(scaler, 'feature_scaler.joblib')
    
    # Save PSO weights
    pso_model_data = {
        'weights': best_weights.tolist(),
        'fitness': best_fitness,
        'convergence_history': pso.convergence_history,
        'feature_names': extractor.feature_names
    }
    
    with open('pso_communication_model.json', 'w') as f:
        json.dump(pso_model_data, f, indent=2)
    
    # Create comparison report
    print("\n📈 Creating comparison report...")
    
    report_data = {
        'timestamp': datetime.now().isoformat(),
        'dataset_info': {
            'total_samples': len(sample_data),
            'verbal_samples': sum(y_train),
            'nonverbal_samples': len(y_train) - sum(y_train),
            'feature_count': X_train.shape[1]
        },
        'knn_results': {
            'accuracy': float(knn_accuracy),
            'algorithm': 'K-Nearest Neighbors',
            'parameters': {'k': 3}
        },
        'pso_results': {
            'accuracy': float(pso_accuracy),
            'algorithm': 'Particle Swarm Optimization',
            'parameters': {
                'particles': pso.n_particles,
                'iterations': len(pso.convergence_history),
                'final_fitness': float(best_fitness)
            }
        },
        'comparison': {
            'better_algorithm': 'PSO' if pso_accuracy > knn_accuracy else 'KNN',
            'accuracy_difference': abs(float(pso_accuracy - knn_accuracy)),
            'recommendation': 'PSO untuk optimasi fitur, KNN untuk interpretabilitas'
        }
    }
    
    with open('classification_report.json', 'w') as f:
        json.dump(report_data, f, indent=2)
    
    # Plot convergence
    if len(pso.convergence_history) > 1:
        plt.figure(figsize=(10, 6))
        plt.plot(pso.convergence_history, 'b-', linewidth=2, label='PSO Convergence')
        plt.axhline(y=1-knn_accuracy, color='r', linestyle='--', label=f'KNN Error Rate ({1-knn_accuracy:.3f})')
        plt.xlabel('Iteration')
        plt.ylabel('Fitness (Error Rate)')
        plt.title('PSO Convergence vs KNN Performance')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('convergence_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"✅ Analysis complete!")
    print(f"   📁 Files saved:")
    print(f"      - knn_communication_model.joblib")
    print(f"      - pso_communication_model.json")
    print(f"      - feature_scaler.joblib")
    print(f"      - classification_report.json")
    print(f"      - convergence_comparison.png")
    
    print(f"\n🎯 Summary:")
    print(f"   - KNN Accuracy: {knn_accuracy:.1%}")
    print(f"   - PSO Accuracy: {pso_accuracy:.1%}")
    print(f"   - Best Algorithm: {report_data['comparison']['better_algorithm']}")
    
    # Demo feature extraction on sample image
    print(f"\n🖼️ Feature extraction demo:")
    print(f"   Untuk menggunakan dengan gambar nyata:")
    print(f"   ```python")
    print(f"   extractor = CommunicationFeatureExtractor()")
    print(f"   features = extractor.extract_all_features('path/to/image.jpg')")
    print(f"   feature_array = extractor.extract_features_array('path/to/image.jpg')")
    print(f"   ```")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()