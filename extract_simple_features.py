import os
import cv2
import numpy as np
import pandas as pd

def extract_simple_features(image):
    """Ekstrak fitur sederhana dari gambar"""
    features = []
    
    # Resize gambar ke ukuran standar
    image = cv2.resize(image, (100, 100))
    
    # 1. Fitur warna (rata-rata RGB)
    if len(image.shape) == 3:
        mean_b = np.mean(image[:,:,0])  # Blue
        mean_g = np.mean(image[:,:,1])  # Green
        mean_r = np.mean(image[:,:,2])  # Red
        
        std_b = np.std(image[:,:,0])
        std_g = np.std(image[:,:,1])
        std_r = np.std(image[:,:,2])
    else:
        mean_b = mean_g = mean_r = np.mean(image)
        std_b = std_g = std_r = np.std(image)
    
    features.extend([mean_r, mean_g, mean_b, std_r, std_g, std_b])
    
    # 2. Konversi ke grayscale untuk fitur lainnya
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # 3. Fitur tekstur sederhana (histogram intensitas)
    hist, _ = np.histogram(gray, bins=8, range=(0, 256))
    hist = hist / np.sum(hist)  # Normalisasi
    features.extend(hist.tolist())
    
    # 4. Fitur bentuk sederhana
    # Threshold untuk binary image
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Hitung area putih (objek)
    white_pixels = np.sum(binary == 255)
    total_pixels = binary.shape[0] * binary.shape[1]
    area_ratio = white_pixels / total_pixels
    
    # Hitung moments untuk mendapatkan centroid
    moments = cv2.moments(binary)
    if moments['m00'] != 0:
        cx = int(moments['m10'] / moments['m00'])
        cy = int(moments['m01'] / moments['m00'])
        centrality_x = abs(cx - binary.shape[1]/2) / (binary.shape[1]/2)
        centrality_y = abs(cy - binary.shape[0]/2) / (binary.shape[0]/2)
    else:
        centrality_x = centrality_y = 0
    
    features.extend([area_ratio, centrality_x, centrality_y])
    
    # 5. Fitur edge (tepi)
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.sum(edges > 0) / total_pixels
    features.append(edge_density)
    
    return features

def process_medicinal_dataset():
    """Proses dataset tanaman obat"""
    base_path = r"c:\Users\ASUS\OneDrive\Dokumen\FINAL PEMOGRAMAN WEB SESMETER 4\Medicinal Leaf Dataset\Segmented Medicinal Leaf Images"
    
    all_features = []
    all_labels = []
    all_names = []
    
    # Daftar folder tanaman
    plant_folders = [
        "Alpinia Galanga (Rasna)",
        "Amaranthus Viridis (Arive-Dantu)", 
        "Artocarpus Heterophyllus (Jackfruit)",
        "Azadirachta Indica (Neem)",
        "Basella Alba (Basale)",
        "Citrus Limon (Lemon)",
        "Hibiscus Rosa-sinensis",
        "Jasminum (Jasmine)",
        "Mentha (Mint)",
        "Moringa Oleifera (Drumstick)",
        "Ocimum Tenuiflorum (Tulsi)",
        "Piper Betle (Betel)"
    ]
    
    for plant_folder in plant_folders:
        plant_path = os.path.join(base_path, plant_folder)
        
        if os.path.exists(plant_path):
            print(f"Memproses: {plant_folder}")
            
            # Ambil nama tanaman yang lebih sederhana
            plant_name = plant_folder.split('(')[0].strip()
            
            # Ambil beberapa gambar dari folder
            image_files = []
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
                import glob
                image_files.extend(glob.glob(os.path.join(plant_path, ext)))
                image_files.extend(glob.glob(os.path.join(plant_path, ext.upper())))
            
            # Proses maksimal 3 gambar per tanaman
            for i, image_file in enumerate(image_files[:3]):
                try:
                    # Baca gambar
                    image = cv2.imread(image_file)
                    if image is None:
                        print(f"  Gagal membaca: {os.path.basename(image_file)}")
                        continue
                    
                    # Ekstrak fitur
                    features = extract_simple_features(image)
                    
                    all_features.append(features)
                    all_labels.append("Herbal")  # Semua tanaman dalam dataset ini herbal
                    all_names.append(plant_name)
                    
                    print(f"  Berhasil: {os.path.basename(image_file)}")
                    
                except Exception as e:
                    print(f"  Error: {e}")
                    continue
    
    return all_features, all_labels, all_names

def create_excel_dataset():
    """Buat dataset Excel dari fitur gambar"""
    print("Mulai ekstraksi fitur gambar...")
    
    # Proses gambar
    features, labels, names = process_medicinal_dataset()
    
    if len(features) == 0:
        print("Tidak ada gambar yang berhasil diproses!")
        return
    
    # Buat nama kolom
    column_names = [
        'Mean_Red', 'Mean_Green', 'Mean_Blue',
        'Std_Red', 'Std_Green', 'Std_Blue',
        'Hist_0', 'Hist_1', 'Hist_2', 'Hist_3', 
        'Hist_4', 'Hist_5', 'Hist_6', 'Hist_7',
        'Area_Ratio', 'Centrality_X', 'Centrality_Y', 'Edge_Density',
        'Plant_Name', 'Label'
    ]
    
    # Buat DataFrame
    data = []
    for i in range(len(features)):
        row = features[i] + [names[i], labels[i]]
        data.append(row)
    
    df = pd.DataFrame(data, columns=column_names)
    
    # Simpan ke Excel
    output_file = r"c:\Users\ASUS\OneDrive\Dokumen\FINAL PEMOGRAMAN WEB SESMETER 4\dataset_fitur_gambar.xlsx"
    df.to_excel(output_file, index=False)
    
    print(f"\nDataset berhasil dibuat!")
    print(f"File: {output_file}")
    print(f"Total sampel: {len(df)}")
    print(f"Total fitur: {len(column_names) - 2}")  # Minus nama dan label
    
    # Tampilkan statistik
    print(f"\nDistribusi tanaman:")
    print(df['Plant_Name'].value_counts())
    
    # Simpan juga versi CSV
    csv_file = r"c:\Users\ASUS\OneDrive\Dokumen\FINAL PEMOGRAMAN WEB SESMETER 4\dataset_fitur_gambar.csv"
    df.to_csv(csv_file, index=False)
    print(f"File CSV: {csv_file}")
    
    return df

if __name__ == "__main__":
    # Pastikan OpenCV tersedia
    try:
        import cv2
        print("OpenCV tersedia, memulai ekstraksi...")
        df = create_excel_dataset()
        
        if df is not None:
            print("\nContoh data (5 baris pertama):")
            print(df.head())
            
    except ImportError:
        print("OpenCV tidak tersedia. Silakan install dengan: pip install opencv-python")