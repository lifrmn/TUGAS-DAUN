#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🗣️ EKSTRAKSI FITUR GAMBAR KOMUNIKASI
Untuk klasifikasi Verbal vs Non-Verbal menggunakan KNN dan PSO
"""

import cv2
import numpy as np
import pandas as pd
from PIL import Image
import os
import json
from datetime import datetime

def extractKomunikasiFeatures(image_path):
    """
    Ekstraksi fitur dari gambar objek komunikasi
    
    Args:
        image_path (str): Path ke file gambar
        
    Returns:
        dict: Dictionary berisi fitur yang diekstrak
    """
    try:
        # Baca gambar
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Tidak dapat membaca gambar: {image_path}")
        
        # Convert ke RGB untuk PIL
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        
        # Convert ke grayscale untuk analisis bentuk
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 1. FITUR WARNA (Color Features)
        color_features = extractColorFeatures(img_rgb)
        
        # 2. FITUR BENTUK (Shape Features)  
        shape_features = extractShapeFeatures(gray)
        
        # 3. FITUR TEKSTUR (Texture Features)
        texture_features = extractTextureFeatures(gray)
        
        # 4. FITUR STATISTIK (Statistical Features)
        stat_features = extractStatisticalFeatures(img_rgb)
        
        # Gabungkan semua fitur
        features = {
            **color_features,
            **shape_features, 
            **texture_features,
            **stat_features,
            'image_path': image_path,
            'extraction_time': datetime.now().isoformat()
        }
        
        return features
        
    except Exception as e:
        print(f"Error extracting features from {image_path}: {str(e)}")
        return None

def extractColorFeatures(img_rgb):
    """Ekstraksi fitur warna"""
    # Mean RGB values
    mean_red = np.mean(img_rgb[:, :, 0])
    mean_green = np.mean(img_rgb[:, :, 1]) 
    mean_blue = np.mean(img_rgb[:, :, 2])
    
    # Standard deviation RGB
    std_red = np.std(img_rgb[:, :, 0])
    std_green = np.std(img_rgb[:, :, 1])
    std_blue = np.std(img_rgb[:, :, 2])
    
    # Brightness (luminance)
    brightness = 0.299 * mean_red + 0.587 * mean_green + 0.114 * mean_blue
    
    # Dominant color (channel with highest mean)
    dominant_channel = np.argmax([mean_red, mean_green, mean_blue])
    dominant_color = ['Red', 'Green', 'Blue'][dominant_channel]
    
    # Color histogram features
    hist_r = cv2.calcHist([img_rgb], [0], None, [256], [0, 256])
    hist_g = cv2.calcHist([img_rgb], [1], None, [256], [0, 256])  
    hist_b = cv2.calcHist([img_rgb], [2], None, [256], [0, 256])
    
    # Histogram statistics
    hist_mean_r = np.mean(hist_r)
    hist_mean_g = np.mean(hist_g)
    hist_mean_b = np.mean(hist_b)
    
    return {
        'meanRed': float(mean_red),
        'meanGreen': float(mean_green), 
        'meanBlue': float(mean_blue),
        'stdRed': float(std_red),
        'stdGreen': float(std_green),
        'stdBlue': float(std_blue),
        'brightness': float(brightness),
        'dominantColor': dominant_color,
        'histMeanR': float(hist_mean_r),
        'histMeanG': float(hist_mean_g),
        'histMeanB': float(hist_mean_b)
    }

def extractShapeFeatures(gray):
    """Ekstraksi fitur bentuk"""
    # Threshold untuk segmentasi
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        return {
            'area': 0.0,
            'perimeter': 0.0,
            'circularity': 0.0,
            'aspectRatio': 1.0,
            'extent': 0.0,
            'solidity': 0.0,
            'compactness': 0.0
        }
    
    # Ambil contour terbesar
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Area dan perimeter
    area = cv2.contourArea(largest_contour)
    perimeter = cv2.arcLength(largest_contour, True)
    
    # Circularity (0-1, 1 = perfect circle)
    if perimeter > 0:
        circularity = 4 * np.pi * area / (perimeter * perimeter)
    else:
        circularity = 0
    
    # Bounding rectangle
    x, y, w, h = cv2.boundingRect(largest_contour)
    aspect_ratio = float(w) / h if h > 0 else 1.0
    rect_area = w * h
    extent = float(area) / rect_area if rect_area > 0 else 0
    
    # Convex hull
    hull = cv2.convexHull(largest_contour)
    hull_area = cv2.contourArea(hull)
    solidity = float(area) / hull_area if hull_area > 0 else 0
    
    # Compactness
    if area > 0:
        compactness = (perimeter * perimeter) / area
    else:
        compactness = 0
    
    return {
        'area': float(area),
        'perimeter': float(perimeter),
        'circularity': float(circularity),
        'aspectRatio': float(aspect_ratio),
        'extent': float(extent),
        'solidity': float(solidity),
        'compactness': float(compactness)
    }

def extractTextureFeatures(gray):
    """Ekstraksi fitur tekstur menggunakan Local Binary Pattern"""
    def calculate_lbp(image, radius=1, n_points=8):
        """Calculate Local Binary Pattern"""
        lbp = np.zeros_like(image)
        for i in range(radius, image.shape[0] - radius):
            for j in range(radius, image.shape[1] - radius):
                center = image[i, j]
                pattern = 0
                for k in range(n_points):
                    angle = 2 * np.pi * k / n_points
                    x = int(i + radius * np.cos(angle))
                    y = int(j + radius * np.sin(angle))
                    if 0 <= x < image.shape[0] and 0 <= y < image.shape[1]:
                        if image[x, y] >= center:
                            pattern |= (1 << k)
                lbp[i, j] = pattern
        return lbp
    
    # Calculate LBP
    lbp = calculate_lbp(gray)
    
    # LBP histogram
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=256, range=(0, 256))
    
    # Texture statistics
    lbp_mean = np.mean(lbp)
    lbp_std = np.std(lbp)
    lbp_energy = np.sum(lbp_hist ** 2)
    lbp_entropy = -np.sum(lbp_hist * np.log2(lbp_hist + 1e-7))
    
    # Gradient magnitude (edge information)
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    
    edge_density = np.mean(grad_magnitude)
    edge_std = np.std(grad_magnitude)
    
    return {
        'lbpMean': float(lbp_mean),
        'lbpStd': float(lbp_std),
        'lbpEnergy': float(lbp_energy),
        'lbpEntropy': float(lbp_entropy),
        'edgeDensity': float(edge_density),
        'edgeStd': float(edge_std)
    }

def extractStatisticalFeatures(img_rgb):
    """Ekstraksi fitur statistik"""
    # Convert to grayscale for some calculations
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    
    # Basic statistics
    mean_intensity = np.mean(gray)
    std_intensity = np.std(gray)
    min_intensity = np.min(gray)
    max_intensity = np.max(gray)
    
    # Skewness and Kurtosis approximation
    mean_val = np.mean(gray)
    std_val = np.std(gray)
    
    if std_val > 0:
        # Simplified skewness calculation
        skewness = np.mean(((gray - mean_val) / std_val) ** 3)
        # Simplified kurtosis calculation  
        kurtosis = np.mean(((gray - mean_val) / std_val) ** 4) - 3
    else:
        skewness = 0
        kurtosis = 0
    
    # Contrast (RMS contrast)
    contrast = np.sqrt(np.mean((gray - mean_intensity) ** 2))
    
    # Uniformity (energy)
    hist, _ = np.histogram(gray, bins=256, range=(0, 256))
    normalized_hist = hist / np.sum(hist)
    uniformity = np.sum(normalized_hist ** 2)
    
    # Entropy
    entropy = -np.sum(normalized_hist * np.log2(normalized_hist + 1e-7))
    
    return {
        'meanIntensity': float(mean_intensity),
        'stdIntensity': float(std_intensity),
        'minIntensity': float(min_intensity),
        'maxIntensity': float(max_intensity),
        'skewness': float(skewness),
        'kurtosis': float(kurtosis),
        'contrast': float(contrast),
        'uniformity': float(uniformity),
        'entropy': float(entropy)
    }

def processKomunikasiDataset(dataset_folder):
    """
    Proses seluruh dataset komunikasi dan ekstrak fitur
    
    Args:
        dataset_folder (str): Path ke folder dataset
        
    Returns:
        pandas.DataFrame: DataFrame berisi fitur dari semua gambar
    """
    features_list = []
    
    # Struktur folder: dataset_folder/Verbal/ dan dataset_folder/Non-Verbal/
    categories = ['Verbal', 'Non-Verbal']
    
    for category in categories:
        category_path = os.path.join(dataset_folder, category)
        if not os.path.exists(category_path):
            print(f"Warning: Folder {category_path} tidak ditemukan")
            continue
            
        print(f"Processing {category} images...")
        
        for filename in os.listdir(category_path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                image_path = os.path.join(category_path, filename)
                
                print(f"  Extracting features from: {filename}")
                features = extractKomunikasiFeatures(image_path)
                
                if features:
                    features['filename'] = filename
                    features['category'] = category
                    features_list.append(features)
    
    if features_list:
        df = pd.DataFrame(features_list)
        return df
    else:
        print("No features extracted!")
        return pd.DataFrame()

def saveFeaturesToExcel(features_df, output_file='komunikasi_features.xlsx'):
    """Save extracted features to Excel file"""
    try:
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            # Sheet 1: Raw features
            features_df.to_excel(writer, sheet_name='Raw_Features', index=False)
            
            # Sheet 2: Statistical summary
            numeric_cols = features_df.select_dtypes(include=[np.number]).columns
            summary_df = features_df[numeric_cols].describe()
            summary_df.to_excel(writer, sheet_name='Statistical_Summary')
            
            # Sheet 3: Category comparison
            if 'category' in features_df.columns:
                category_stats = features_df.groupby('category')[numeric_cols].mean()
                category_stats.to_excel(writer, sheet_name='Category_Comparison')
        
        print(f"✅ Features saved to: {output_file}")
        
    except Exception as e:
        print(f"❌ Error saving to Excel: {str(e)}")

def main():
    """Main execution function"""
    print("🗣️ KOMUNIKASI FEATURE EXTRACTOR")
    print("=" * 50)
    
    # Example usage for single image
    print("\n📸 Testing single image extraction...")
    test_image = "test_communication.jpg"  # Ganti dengan path gambar test
    
    if os.path.exists(test_image):
        features = extractKomunikasiFeatures(test_image)
        if features:
            print("✅ Features extracted successfully!")
            print(json.dumps(features, indent=2))
        else:
            print("❌ Failed to extract features")
    else:
        print(f"⚠️  Test image not found: {test_image}")
    
    # Example usage for dataset
    print("\n📁 Processing dataset...")
    dataset_folder = "komunikasi_dataset"  # Ganti dengan path dataset
    
    if os.path.exists(dataset_folder):
        features_df = processKomunikasiDataset(dataset_folder)
        
        if not features_df.empty:
            print(f"✅ Processed {len(features_df)} images")
            print(f"📊 Features shape: {features_df.shape}")
            
            # Save to Excel
            saveFeaturesToExcel(features_df)
            
            # Save to CSV for web application
            csv_file = 'web/komunikasi_features.csv'
            features_df.to_csv(csv_file, index=False)
            print(f"💾 Features saved to: {csv_file}")
            
        else:
            print("❌ No features extracted from dataset")
    else:
        print(f"⚠️  Dataset folder not found: {dataset_folder}")
        print("📝 Create folder structure:")
        print("   komunikasi_dataset/")
        print("   ├── Verbal/")
        print("   │   ├── book.jpg")
        print("   │   ├── microphone.jpg")
        print("   │   └── ...")
        print("   └── Non-Verbal/")
        print("       ├── painting.jpg")
        print("       ├── gesture.jpg")
        print("       └── ...")

if __name__ == "__main__":
    main()