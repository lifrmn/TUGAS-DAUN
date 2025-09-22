import os
import cv2
import numpy as np
import pandas as pd
from skimage import feature, measure
import glob

def extract_color_features(image):
    """Ekstrak fitur warna dari gambar"""
    # Konversi ke RGB jika perlu
    if len(image.shape) == 3:
        # Rata-rata RGB
        mean_r = np.mean(image[:,:,2])
        mean_g = np.mean(image[:,:,1])
        mean_b = np.mean(image[:,:,0])
        
        # Standar deviasi RGB
        std_r = np.std(image[:,:,2])
        std_g = np.std(image[:,:,1])
        std_b = np.std(image[:,:,0])
        
        return [mean_r, mean_g, mean_b, std_r, std_g, std_b]
    else:
        # Gambar grayscale
        mean_gray = np.mean(image)
        std_gray = np.std(image)
        return [mean_gray, mean_gray, mean_gray, std_gray, std_gray, std_gray]

def extract_texture_features(image):
    """Ekstrak fitur tekstur menggunakan LBP (Local Binary Pattern)"""
    # Konversi ke grayscale jika perlu
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Local Binary Pattern
    lbp = feature.local_binary_pattern(gray, 8, 1, method='uniform')
    hist, _ = np.histogram(lbp.ravel(), bins=10, range=(0, 10))
    
    # Normalisasi histogram
    hist = hist.astype(float)
    hist /= (hist.sum() + 1e-7)
    
    return hist.tolist()

def extract_shape_features(image):
    """Ekstrak fitur bentuk dari gambar"""
    # Konversi ke grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Threshold untuk mendapatkan binary image
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Temukan kontur
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) > 0:
        # Ambil kontur terbesar
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Hitung fitur bentuk
        area = cv2.contourArea(largest_contour)
        perimeter = cv2.arcLength(largest_contour, True)
        
        # Aspek rasio dari bounding rectangle
        x, y, w, h = cv2.boundingRect(largest_contour)
        aspect_ratio = float(w) / h if h != 0 else 0
        
        # Rectangularity
        rect_area = w * h
        rectangularity = area / rect_area if rect_area != 0 else 0
        
        # Circularity
        circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter != 0 else 0
        
        return [area, perimeter, aspect_ratio, rectangularity, circularity]
    else:
        return [0, 0, 0, 0, 0]

def process_images(dataset_path):
    """Proses semua gambar dalam dataset dan ekstrak fitur"""
    features_list = []
    labels = []
    
    # Iterasi melalui setiap folder (kelas)
    for class_folder in os.listdir(dataset_path):
        class_path = os.path.join(dataset_path, class_folder)
        
        if os.path.isdir(class_path):
            print(f"Memproses kelas: {class_folder}")
            
            # Tentukan apakah herbal atau non-herbal berdasarkan nama
            # Semua tanaman dalam dataset ini adalah tanaman obat (herbal)
            label = "Herbal"
            
            # Iterasi melalui gambar dalam folder
            image_files = glob.glob(os.path.join(class_path, "*.*"))
            
            for i, image_file in enumerate(image_files[:5]):  # Ambil maksimal 5 gambar per kelas
                try:
                    # Baca gambar
                    image = cv2.imread(image_file)
                    if image is None:
                        continue
                    
                    # Resize gambar untuk konsistensi
                    image = cv2.resize(image, (224, 224))
                    
                    # Ekstrak fitur
                    color_features = extract_color_features(image)
                    texture_features = extract_texture_features(image)
                    shape_features = extract_shape_features(image)
                    
                    # Gabungkan semua fitur
                    all_features = color_features + texture_features + shape_features
                    
                    features_list.append(all_features)
                    labels.append(label)
                    
                    print(f"  Diproses: {os.path.basename(image_file)}")
                    
                except Exception as e:
                    print(f"  Error memproses {image_file}: {e}")
                    continue
    
    return features_list, labels

def create_excel_file(features, labels, output_file):
    """Buat file Excel dengan fitur yang diekstrak"""
    # Buat nama kolom
    color_cols = ['Mean_R', 'Mean_G', 'Mean_B', 'Std_R', 'Std_G', 'Std_B']
    texture_cols = [f'LBP_{i}' for i in range(10)]
    shape_cols = ['Area', 'Perimeter', 'Aspect_Ratio', 'Rectangularity', 'Circularity']
    
    all_columns = color_cols + texture_cols + shape_cols + ['Label']
    
    # Buat DataFrame
    data = []
    for i, feature_row in enumerate(features):
        row = feature_row + [labels[i]]
        data.append(row)
    
    df = pd.DataFrame(data, columns=all_columns)
    
    # Simpan ke Excel
    df.to_excel(output_file, index=False)
    print(f"Data disimpan ke: {output_file}")
    
    # Tampilkan statistik
    print(f"\nStatistik Dataset:")
    print(f"Total sampel: {len(df)}")
    print(f"Total fitur: {len(all_columns) - 1}")
    print(f"Distribusi label:")
    print(df['Label'].value_counts())
    
    return df

if __name__ == "__main__":
    # Path dataset
    dataset_path = r"c:\Users\ASUS\OneDrive\Dokumen\FINAL PEMOGRAMAN WEB SESMETER 4\Medicinal Leaf Dataset\Segmented Medicinal Leaf Images"
    output_file = r"c:\Users\ASUS\OneDrive\Dokumen\FINAL PEMOGRAMAN WEB SESMETER 4\fitur_gambar_tanaman.xlsx"
    
    print("Memulai ekstraksi fitur dari gambar...")
    print("=" * 50)
    
    # Proses gambar
    features, labels = process_images(dataset_path)
    
    if len(features) > 0:
        # Buat file Excel
        df = create_excel_file(features, labels, output_file)
        
        print("\n" + "=" * 50)
        print("Ekstraksi fitur selesai!")
        print(f"File Excel tersimpan di: {output_file}")
        
        # Tampilkan contoh data
        print("\nContoh 5 baris pertama:")
        print(df.head())
        
    else:
        print("Tidak ada gambar yang berhasil diproses!")