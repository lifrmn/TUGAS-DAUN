// Data tanaman herbal dan non-herbal
const dataset = [
    {nama: 'Jahe', jenis: 'Herbal', warna: 'Hijau', bentuk: 'Lonjong', tinggi: 60, aroma: 'Kuat'},
    {nama: 'Kunyit', jenis: 'Herbal', warna: 'Hijau', bentuk: 'Lonjong', tinggi: 50, aroma: 'Kuat'},
    {nama: 'Sirih', jenis: 'Herbal', warna: 'Hijau', bentuk: 'Bulat', tinggi: 30, aroma: 'Kuat'},
    {nama: 'Lidah Buaya', jenis: 'Herbal', warna: 'Hijau', bentuk: 'Lancip', tinggi: 40, aroma: 'Lemah'},
    {nama: 'Pandan', jenis: 'Herbal', warna: 'Hijau', bentuk: 'Lonjong', tinggi: 70, aroma: 'Kuat'},
    {nama: 'Tomat', jenis: 'Non-Herbal', warna: 'Hijau', bentuk: 'Bulat', tinggi: 80, aroma: 'Tidak Ada'},
    {nama: 'Bayam', jenis: 'Non-Herbal', warna: 'Hijau', bentuk: 'Lonjong', tinggi: 30, aroma: 'Lemah'},
    {nama: 'Cabai', jenis: 'Non-Herbal', warna: 'Hijau', bentuk: 'Lancip', tinggi: 60, aroma: 'Kuat'},
    {nama: 'Terong', jenis: 'Non-Herbal', warna: 'Hijau', bentuk: 'Bulat', tinggi: 90, aroma: 'Tidak Ada'},
    {nama: 'Wortel', jenis: 'Non-Herbal', warna: 'Hijau', bentuk: 'Lonjong', tinggi: 40, aroma: 'Tidak Ada'}
];

// Extended dataset dengan fitur gambar (simulasi)
const extendedDataset = [
    {nama: 'Jahe', jenis: 'Herbal', meanRed: 120, meanGreen: 180, meanBlue: 90, area: 1500, brightness: 110},
    {nama: 'Kunyit', jenis: 'Herbal', meanRed: 110, meanGreen: 170, meanBlue: 85, area: 1400, brightness: 105},
    {nama: 'Sirih', jenis: 'Herbal', meanRed: 100, meanGreen: 160, meanBlue: 80, area: 1200, brightness: 100},
    {nama: 'Lidah Buaya', jenis: 'Herbal', meanRed: 95, meanGreen: 155, meanBlue: 75, area: 1100, brightness: 95},
    {nama: 'Pandan', jenis: 'Herbal', meanRed: 105, meanGreen: 165, meanBlue: 82, area: 1300, brightness: 102},
    {nama: 'Tomat', jenis: 'Non-Herbal', meanRed: 140, meanGreen: 200, meanBlue: 100, area: 2000, brightness: 130},
    {nama: 'Bayam', jenis: 'Non-Herbal', meanRed: 130, meanGreen: 190, meanBlue: 95, area: 1800, brightness: 125},
    {nama: 'Cabai', jenis: 'Non-Herbal', meanRed: 135, meanGreen: 195, meanBlue: 98, area: 1900, brightness: 128},
    {nama: 'Terong', jenis: 'Non-Herbal', meanRed: 145, meanGreen: 205, meanBlue: 105, area: 2100, brightness: 135},
    {nama: 'Wortel', jenis: 'Non-Herbal', meanRed: 125, meanGreen: 185, meanBlue: 92, area: 1700, brightness: 120}
];

// Konversi fitur kategorikal ke angka
function encodeTanaman(tanaman) {
    const warnaMap = {'Hijau': 0, 'Tua': 1, 'Muda': 2};
    const bentukMap = {'Lonjong': 0, 'Bulat': 1, 'Lancip': 2};
    const aromaMap = {'Kuat': 0, 'Lemah': 1, 'Tidak Ada': 2};
    return [
        warnaMap[tanaman.warna],
        bentukMap[tanaman.bentuk],
        tanaman.tinggi,
        aromaMap[tanaman.aroma]
    ];
}

// Normalisasi fitur untuk analisis gambar
function normalizeFeatures(features) {
    const normalized = [];
    const scales = [255, 255, 255, 5000, 255]; // RGB, area, brightness
    for (let i = 0; i < features.length; i++) {
        normalized.push(features[i] / scales[i]);
    }
    return normalized;
}

// Hitung jarak Euclidean
function euclidean(a, b) {
    let sum = 0;
    for (let i = 0; i < a.length; i++) {
        sum += Math.pow(a[i] - b[i], 2);
    }
    return Math.sqrt(sum);
}

// KNN prediksi
function knnPredict(input, k = 3, useExtended = false) {
    const dataToUse = useExtended ? extendedDataset : dataset;
    const inputEncoded = useExtended ? 
        normalizeFeatures([input.meanRed, input.meanGreen, input.meanBlue, input.area, input.brightness]) :
        encodeTanaman(input);
    
    const distances = dataToUse.map(data => ({
        nama: data.nama,
        jenis: data.jenis,
        jarak: euclidean(inputEncoded, useExtended ? 
            normalizeFeatures([data.meanRed, data.meanGreen, data.meanBlue, data.area, data.brightness]) :
            encodeTanaman(data))
    }));
    
    distances.sort((a, b) => a.jarak - b.jarak);
    const nearest = distances.slice(0, k);
    const count = { 'Herbal': 0, 'Non-Herbal': 0 };
    nearest.forEach(n => count[n.jenis]++);
    
    return {
        prediksi: count['Herbal'] > count['Non-Herbal'] ? 'Herbal' : 'Non-Herbal',
        confidence: Math.max(count['Herbal'], count['Non-Herbal']) / k,
        neighbors: nearest
    };
}

// VSO Algorithm (Variable Selection Optimization)
function vsoPredict(input) {
    // Simulasi algoritma VSO dengan pembobotan fitur
    const weights = [0.3, 0.25, 0.2, 0.15, 0.1]; // Bobot untuk setiap fitur
    const inputFeatures = normalizeFeatures([input.meanRed, input.meanGreen, input.meanBlue, input.area, input.brightness]);
    
    let herbalScore = 0;
    let nonHerbalScore = 0;
    
    extendedDataset.forEach(data => {
        const dataFeatures = normalizeFeatures([data.meanRed, data.meanGreen, data.meanBlue, data.area, data.brightness]);
        let weightedDistance = 0;
        
        for (let i = 0; i < inputFeatures.length; i++) {
            weightedDistance += weights[i] * Math.pow(inputFeatures[i] - dataFeatures[i], 2);
        }
        
        const similarity = 1 / (1 + Math.sqrt(weightedDistance));
        
        if (data.jenis === 'Herbal') {
            herbalScore += similarity;
        } else {
            nonHerbalScore += similarity;
        }
    });
    
    return {
        prediksi: herbalScore > nonHerbalScore ? 'Herbal' : 'Non-Herbal',
        confidence: Math.max(herbalScore, nonHerbalScore) / (herbalScore + nonHerbalScore),
        scores: { herbal: herbalScore, nonHerbal: nonHerbalScore }
    };
}

// Ekstraksi fitur dari gambar (simulasi)
function extractImageFeatures(imageData) {
    // Simulasi ekstraksi fitur dari canvas image data
    let r = 0, g = 0, b = 0, pixelCount = 0;
    
    for (let i = 0; i < imageData.data.length; i += 4) {
        r += imageData.data[i];
        g += imageData.data[i + 1];
        b += imageData.data[i + 2];
        pixelCount++;
    }
    
    return {
        meanRed: r / pixelCount,
        meanGreen: g / pixelCount,
        meanBlue: b / pixelCount,
        area: pixelCount / 100, // Simplified area calculation
        brightness: (r + g + b) / (3 * pixelCount)
    };
}

// Tab switching
function switchTab(tabName) {
    document.querySelectorAll('.tab-content').forEach(tab => tab.classList.remove('active'));
    document.querySelectorAll('.tab-button').forEach(btn => btn.classList.remove('active'));
    
    document.getElementById(tabName + 'Tab').classList.add('active');
    event.target.classList.add('active');
}

// Upload area functionality
const uploadArea = document.getElementById('uploadArea');
const imageInput = document.getElementById('imageInput');
const imagePreview = document.getElementById('imagePreview');
const predictButton = document.getElementById('predictButton');

uploadArea.addEventListener('click', () => imageInput.click());
uploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadArea.style.background = '#e8f5e8';
});
uploadArea.addEventListener('dragleave', () => {
    uploadArea.style.background = '';
});
uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadArea.style.background = '';
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        handleImageUpload(files[0]);
    }
});

imageInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
        handleImageUpload(e.target.files[0]);
    }
});

function handleImageUpload(file) {
    const reader = new FileReader();
    reader.onload = function(e) {
        const img = document.createElement('img');
        img.src = e.target.result;
        img.onload = function() {
            imagePreview.innerHTML = '';
            imagePreview.appendChild(img);
            predictButton.disabled = false;
        };
    };
    reader.readAsDataURL(file);
}

// Event handlers
const form = document.getElementById('formTanaman');
const uploadForm = document.getElementById('formUpload');
const hasil = document.getElementById('hasilPrediksi');
const detailHasil = document.getElementById('detailHasil');

// Manual form submission
form.addEventListener('submit', function(e) {
    e.preventDefault();
    const input = {
        warna: document.getElementById('warna').value,
        bentuk: document.getElementById('bentuk').value,
        tinggi: parseInt(document.getElementById('tinggi').value),
        aroma: document.getElementById('aroma').value
    };
    
    const result = knnPredict(input);
    hasil.innerHTML = `
        <div style="font-size: 1.5rem; margin-bottom: 8px;">
            Tanaman diprediksi: <strong>${result.prediksi}</strong>
        </div>
        <div style="font-size: 1rem; color: #666;">
            Confidence: ${(result.confidence * 100).toFixed(1)}%
        </div>
    `;
    
    detailHasil.innerHTML = `
        <h4>Detail Analisis KNN:</h4>
        <div class="feature-grid">
            ${result.neighbors.map((neighbor, index) => `
                <div class="feature-item">
                    <strong>Rank ${index + 1}:</strong> ${neighbor.nama}<br>
                    <small>Jarak: ${neighbor.jarak.toFixed(3)}</small>
                </div>
            `).join('')}
        </div>
    `;
});

// Upload form submission
uploadForm.addEventListener('submit', function(e) {
    e.preventDefault();
    
    const img = imagePreview.querySelector('img');
    if (!img) return;
    
    hasil.innerHTML = '<div class="loading"></div> Menganalisis gambar...';
    detailHasil.innerHTML = '';
    
    // Simulate image processing
    setTimeout(() => {
        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d');
        canvas.width = img.width;
        canvas.height = img.height;
        ctx.drawImage(img, 0, 0);
        
        const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
        const features = extractImageFeatures(imageData);
        
        const algorithm = document.getElementById('algorithm').value;
        let result;
        
        if (algorithm === 'knn') {
            result = knnPredict(features, 3, true);
            hasil.innerHTML = `
                <div style="font-size: 1.5rem; margin-bottom: 8px;">
                    Prediksi KNN: <strong>${result.prediksi}</strong>
                </div>
                <div style="font-size: 1rem; color: #666;">
                    Confidence: ${(result.confidence * 100).toFixed(1)}%
                </div>
            `;
            
            detailHasil.innerHTML = `
                <h4>Fitur yang Diekstrak:</h4>
                <div class="feature-grid">
                    <div class="feature-item">Mean Red: ${features.meanRed.toFixed(1)}</div>
                    <div class="feature-item">Mean Green: ${features.meanGreen.toFixed(1)}</div>
                    <div class="feature-item">Mean Blue: ${features.meanBlue.toFixed(1)}</div>
                    <div class="feature-item">Area: ${features.area.toFixed(1)}</div>
                    <div class="feature-item">Brightness: ${features.brightness.toFixed(1)}</div>
                </div>
                <h4>Tetangga Terdekat (KNN):</h4>
                <div class="feature-grid">
                    ${result.neighbors.map((neighbor, index) => `
                        <div class="feature-item">
                            <strong>Rank ${index + 1}:</strong> ${neighbor.nama}<br>
                            <small>Jarak: ${neighbor.jarak.toFixed(3)}</small>
                        </div>
                    `).join('')}
                </div>
            `;
        } else {
            result = vsoPredict(features);
            hasil.innerHTML = `
                <div style="font-size: 1.5rem; margin-bottom: 8px;">
                    Prediksi VSO: <strong>${result.prediksi}</strong>
                </div>
                <div style="font-size: 1rem; color: #666;">
                    Confidence: ${(result.confidence * 100).toFixed(1)}%
                </div>
            `;
            
            detailHasil.innerHTML = `
                <h4>Fitur yang Diekstrak:</h4>
                <div class="feature-grid">
                    <div class="feature-item">Mean Red: ${features.meanRed.toFixed(1)}</div>
                    <div class="feature-item">Mean Green: ${features.meanGreen.toFixed(1)}</div>
                    <div class="feature-item">Mean Blue: ${features.meanBlue.toFixed(1)}</div>
                    <div class="feature-item">Area: ${features.area.toFixed(1)}</div>
                    <div class="feature-item">Brightness: ${features.brightness.toFixed(1)}</div>
                </div>
                <h4>Skor VSO:</h4>
                <div class="feature-grid">
                    <div class="feature-item">Herbal Score: ${result.scores.herbal.toFixed(3)}</div>
                    <div class="feature-item">Non-Herbal Score: ${result.scores.nonHerbal.toFixed(3)}</div>
                </div>
            `;
        }
    }, 1500);
});
