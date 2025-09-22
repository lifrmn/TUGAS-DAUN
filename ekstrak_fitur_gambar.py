import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from skimage import feature, measure
import glob

def ekstrak_fitur_gambar(path_gambar):
    """
    Ekstrak fitur dari gambar untuk klasifikasi tanaman
    """
    try:
        # Baca gambar
        img = cv2.imread(path_gambar)
        if img is None:
            return None
            
        # Konversi ke RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Konversi ke grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 1. FITUR WARNA (RGB Statistics)
        mean_r = np.mean(img_rgb[:,:,0])
        mean_g = np.mean(img_rgb[:,:,1]) 
        mean_b = np.mean(img_rgb[:,:,2])
        
        std_r = np.std(img_rgb[:,:,0])
        std_g = np.std(img_rgb[:,:,1])
        std_b = np.std(img_rgb[:,:,2])
        
        # 2. FITUR TEKSTUR (LBP - Local Binary Pattern)
        lbp = feature.local_binary_pattern(gray, 8, 1, method='uniform')
        lbp_hist, _ = np.histogram(lbp.ravel(), bins=10, range=(0, 10))
        
        # Normalisasi histogram
        lbp_hist = lbp_hist / np.sum(lbp_hist)
        
        # 3. FITUR BENTUK
        # Threshold untuk mendapatkan bentuk daun
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Cari kontur
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Ambil kontur terbesar (daun utama)
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Area dan perimeter
            area = cv2.contourArea(largest_contour)
            perimeter = cv2.arcLength(largest_contour, True)
            
            # Aspect ratio
            x, y, w, h = cv2.boundingRect(largest_contour)
            aspect_ratio = float(w) / h if h != 0 else 0
            
            # Solidity (area/convex hull area)
            hull = cv2.convexHull(largest_contour)
            hull_area = cv2.contourArea(hull)
            solidity = float(area) / hull_area if hull_area != 0 else 0
            
            # Compactness
            compactness = (perimeter * perimeter) / area if area != 0 else 0
        else:
            area = perimeter = aspect_ratio = solidity = compactness = 0
        
        # 4. STATISTIK PIXEL
        brightness = np.mean(gray)
        contrast = np.std(gray)
        
        # Gabungkan semua fitur
        fitur = {
            'mean_red': mean_r,
            'mean_green': mean_g,
            'mean_blue': mean_b,
            'std_red': std_r,
            'std_green': std_g,
            'std_blue': std_b,
            'area': area,
            'perimeter': perimeter,
            'aspect_ratio': aspect_ratio,
            'solidity': solidity,
            'compactness': compactness,
            'brightness': brightness,
            'contrast': contrast
        }
        
        # Tambahkan fitur histogram LBP
        for i, val in enumerate(lbp_hist):
            fitur[f'lbp_hist_{i}'] = val
            
        return fitur
        
    except Exception as e:
        print(f"Error processing {path_gambar}: {e}")
        return None

def proses_dataset(folder_dataset):
    """
    Proses seluruh dataset gambar
    """
    data_fitur = []
    
    # Cari semua file gambar
    ekstensi_gambar = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    
    for root, dirs, files in os.walk(folder_dataset):
        for ekstensi in ekstensi_gambar:
            for file_path in glob.glob(os.path.join(root, ekstensi)):
                print(f"Memproses: {file_path}")
                
                # Ekstrak fitur
                fitur = ekstrak_fitur_gambar(file_path)
                
                if fitur is not None:
                    # Tentukan label berdasarkan nama folder atau file
                    folder_name = os.path.basename(os.path.dirname(file_path)).lower()
                    file_name = os.path.basename(file_path).lower()
                    
                    # Klasifikasi sederhana berdasarkan nama
                    if any(kata in folder_name + file_name for kata in ['medicinal', 'herbal', 'obat']):
                        label = 'Herbal'
                    else:
                        label = 'Non-Herbal'
                    
                    fitur['nama_file'] = os.path.basename(file_path)
                    fitur['path'] = file_path
                    fitur['label'] = label
                    
                    data_fitur.append(fitur)
    
    return data_fitur

def simpan_ke_excel(data_fitur, nama_file='fitur_gambar_tanaman.xlsx'):
    """
    Simpan data fitur ke Excel
    """
    df = pd.DataFrame(data_fitur)
    
    # Reorder kolom
    kolom_utama = ['nama_file', 'label', 'path']
    kolom_fitur = [col for col in df.columns if col not in kolom_utama]
    df = df[kolom_utama + kolom_fitur]
    
    # Simpan ke Excel
    with pd.ExcelWriter(nama_file, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Data_Fitur', index=False)
        
        # Buat sheet template KNN
        template_knn = pd.DataFrame({
            'Fitur': kolom_fitur,
            'Bobot': [1] * len(kolom_fitur),
            'Keterangan': ['Fitur untuk klasifikasi'] * len(kolom_fitur)
        })
        template_knn.to_excel(writer, sheet_name='Template_KNN', index=False)
    
    print(f"Data disimpan ke: {nama_file}")
    return df

# Fungsi utama
if __name__ == "__main__":
    # Path ke dataset
    folder_dataset = r"c:\Users\ASUS\OneDrive\Dokumen\FINAL PEMOGRAMAN WEB SESMETER 4"
    
    print("Memulai ekstraksi fitur gambar...")
    data_fitur = proses_dataset(folder_dataset)
    
    if data_fitur:
        print(f"Berhasil mengekstrak fitur dari {len(data_fitur)} gambar")
        df = simpan_ke_excel(data_fitur)
        print(f"Jumlah fitur per gambar: {len([col for col in df.columns if col not in ['nama_file', 'label', 'path']])}")
    else:
        print("Tidak ada gambar yang berhasil diproses")