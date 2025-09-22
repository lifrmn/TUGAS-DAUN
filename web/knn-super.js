// 🌿 AI PLANT CLASSIFIER PRO - SUPER ADVANCED JAVASCRIPT
// ================================================

// 📊 DATASETS
const dataset = [
    {nama: 'Jahe', jenis: 'Herbal', warna: 'Hijau', bentuk: 'Lonjong', tinggi: 60, aroma: 'Kuat', gambar: 'assets/images/jahe.jpg'},
    {nama: 'Kunyit', jenis: 'Herbal', warna: 'Hijau', bentuk: 'Lonjong', tinggi: 50, aroma: 'Kuat', gambar: 'assets/images/kunyit.jpg'},
    {nama: 'Sirih', jenis: 'Herbal', warna: 'Hijau', bentuk: 'Bulat', tinggi: 30, aroma: 'Kuat', gambar: 'assets/images/sirih.jpg'},
    {nama: 'Lidah Buaya', jenis: 'Herbal', warna: 'Hijau', bentuk: 'Lancip', tinggi: 40, aroma: 'Lemah', gambar: 'assets/images/lidah_buaya.jpg'},
    {nama: 'Pandan', jenis: 'Herbal', warna: 'Hijau', bentuk: 'Lonjong', tinggi: 70, aroma: 'Kuat', gambar: 'assets/images/pandan.jpg'},
    {nama: 'Tomat', jenis: 'Non-Herbal', warna: 'Hijau', bentuk: 'Bulat', tinggi: 80, aroma: 'Tidak Ada', gambar: 'assets/images/tomat.jpg'},
    {nama: 'Bayam', jenis: 'Non-Herbal', warna: 'Hijau', bentuk: 'Lonjong', tinggi: 30, aroma: 'Lemah', gambar: 'assets/images/bayam.jpg'},
    {nama: 'Cabai', jenis: 'Non-Herbal', warna: 'Hijau', bentuk: 'Lancip', tinggi: 60, aroma: 'Kuat', gambar: 'assets/images/cabai.jpg'},
    {nama: 'Terong', jenis: 'Non-Herbal', warna: 'Hijau', bentuk: 'Bulat', tinggi: 90, aroma: 'Tidak Ada', gambar: 'assets/images/terong.jpg'},
    {nama: 'Wortel', jenis: 'Non-Herbal', warna: 'Hijau', bentuk: 'Lonjong', tinggi: 40, aroma: 'Tidak Ada', gambar: 'assets/images/wortel.jpg'}
];

const extendedDataset = [
    {nama: 'Jahe', jenis: 'Herbal', meanRed: 120, meanGreen: 180, meanBlue: 90, area: 1500, brightness: 110, perimeter: 200, solidity: 0.75, gambar: 'assets/images/jahe.jpg'},
    {nama: 'Kunyit', jenis: 'Herbal', meanRed: 110, meanGreen: 170, meanBlue: 85, area: 1400, brightness: 105, perimeter: 190, solidity: 0.72, gambar: 'assets/images/kunyit.jpg'},
    {nama: 'Sirih', jenis: 'Herbal', meanRed: 100, meanGreen: 160, meanBlue: 80, area: 1200, brightness: 100, perimeter: 180, solidity: 0.70, gambar: 'assets/images/sirih.jpg'},
    {nama: 'Lidah Buaya', jenis: 'Herbal', meanRed: 95, meanGreen: 155, meanBlue: 75, area: 1100, brightness: 95, perimeter: 170, solidity: 0.68, gambar: 'assets/images/lidah_buaya.jpg'},
    {nama: 'Pandan', jenis: 'Herbal', meanRed: 105, meanGreen: 165, meanBlue: 82, area: 1300, brightness: 102, perimeter: 185, solidity: 0.73, gambar: 'assets/images/pandan.jpg'},
    {nama: 'Tomat', jenis: 'Non-Herbal', meanRed: 140, meanGreen: 200, meanBlue: 100, area: 2000, brightness: 130, perimeter: 250, solidity: 0.85, gambar: 'assets/images/tomat.jpg'},
    {nama: 'Bayam', jenis: 'Non-Herbal', meanRed: 130, meanGreen: 190, meanBlue: 95, area: 1800, brightness: 125, perimeter: 240, solidity: 0.82, gambar: 'assets/images/bayam.jpg'},
    {nama: 'Cabai', jenis: 'Non-Herbal', meanRed: 135, meanGreen: 195, meanBlue: 98, area: 1900, brightness: 128, perimeter: 245, solidity: 0.83, gambar: 'assets/images/cabai.jpg'},
    {nama: 'Terong', jenis: 'Non-Herbal', meanRed: 145, meanGreen: 205, meanBlue: 105, area: 2100, brightness: 135, perimeter: 260, solidity: 0.87, gambar: 'assets/images/terong.jpg'},
    {nama: 'Wortel', jenis: 'Non-Herbal', meanRed: 125, meanGreen: 185, meanBlue: 92, area: 1700, brightness: 120, perimeter: 235, solidity: 0.80, gambar: 'assets/images/wortel.jpg'}
];

// 🗣️ KOMUNIKASI DATASET - VERBAL vs NON-VERBAL
const komunikasiDataset = [
    // VERBAL - Objek komunikasi verbal
    {nama: 'Buku', kategori: 'Verbal', bentuk: 'Persegi', warna: 'Putih', ukuran: 'Sedang', tekstur: 'Halus', kompleksitas: 'Tinggi', meanRed: 240, meanGreen: 240, meanBlue: 240, area: 1800, brightness: 220, perimeter: 180},
    {nama: 'Papan Tulis', kategori: 'Verbal', bentuk: 'Persegi', warna: 'Putih', ukuran: 'Besar', tekstur: 'Halus', kompleksitas: 'Sedang', meanRed: 250, meanGreen: 250, meanBlue: 250, area: 3000, brightness: 230, perimeter: 280},
    {nama: 'Mikrofon', kategori: 'Verbal', bentuk: 'Bulat', warna: 'Hitam', ukuran: 'Kecil', tekstur: 'Logam', kompleksitas: 'Tinggi', meanRed: 30, meanGreen: 30, meanBlue: 30, area: 800, brightness: 40, perimeter: 120},
    {nama: 'Spidol', kategori: 'Verbal', bentuk: 'Silinder', warna: 'Biru', ukuran: 'Kecil', tekstur: 'Plastik', kompleksitas: 'Rendah', meanRed: 50, meanGreen: 100, meanBlue: 200, area: 600, brightness: 120, perimeter: 100},
    {nama: 'Laptop', kategori: 'Verbal', bentuk: 'Persegi', warna: 'Abu', ukuran: 'Sedang', tekstur: 'Logam', kompleksitas: 'Tinggi', meanRed: 120, meanGreen: 120, meanBlue: 120, area: 2200, brightness: 110, perimeter: 220},
    {nama: 'Smartphone', kategori: 'Verbal', bentuk: 'Persegi', warna: 'Hitam', ukuran: 'Kecil', tekstur: 'Kaca', kompleksitas: 'Tinggi', meanRed: 40, meanGreen: 40, meanBlue: 40, area: 1000, brightness: 50, perimeter: 140},
    {nama: 'Koran', kategori: 'Verbal', bentuk: 'Persegi', warna: 'Putih', ukuran: 'Sedang', tekstur: 'Kertas', kompleksitas: 'Sedang', meanRed: 230, meanGreen: 230, meanBlue: 230, area: 1600, brightness: 210, perimeter: 170},
    {nama: 'Pensil', kategori: 'Verbal', bentuk: 'Silinder', warna: 'Kuning', ukuran: 'Kecil', tekstur: 'Kayu', kompleksitas: 'Rendah', meanRed: 220, meanGreen: 180, meanBlue: 50, area: 500, brightness: 150, perimeter: 80},
    {nama: 'Radio', kategori: 'Verbal', bentuk: 'Persegi', warna: 'Hitam', ukuran: 'Sedang', tekstur: 'Plastik', kompleksitas: 'Sedang', meanRed: 60, meanGreen: 60, meanBlue: 60, area: 1400, brightness: 70, perimeter: 160},
    {nama: 'Telepon', kategori: 'Verbal', bentuk: 'Persegi', warna: 'Putih', ukuran: 'Sedang', tekstur: 'Plastik', kompleksitas: 'Sedang', meanRed: 200, meanGreen: 200, meanBlue: 200, area: 1200, brightness: 180, perimeter: 150},
    
    // NON-VERBAL - Objek komunikasi non-verbal
    {nama: 'Cermin', kategori: 'Non-Verbal', bentuk: 'Persegi', warna: 'Transparan', ukuran: 'Sedang', tekstur: 'Kaca', kompleksitas: 'Rendah', meanRed: 180, meanGreen: 180, meanBlue: 180, area: 1500, brightness: 160, perimeter: 160},
    {nama: 'Lukisan', kategori: 'Non-Verbal', bentuk: 'Persegi', warna: 'Beragam', ukuran: 'Besar', tekstur: 'Kanvas', kompleksitas: 'Tinggi', meanRed: 150, meanGreen: 120, meanBlue: 90, area: 2500, brightness: 130, perimeter: 250},
    {nama: 'Patung', kategori: 'Non-Verbal', bentuk: 'Tidak_Beraturan', warna: 'Abu', ukuran: 'Besar', tekstur: 'Batu', kompleksitas: 'Tinggi', meanRed: 140, meanGreen: 140, meanBlue: 140, area: 3500, brightness: 120, perimeter: 300},
    {nama: 'Emoji Plakat', kategori: 'Non-Verbal', bentuk: 'Bulat', warna: 'Kuning', ukuran: 'Kecil', tekstur: 'Plastik', kompleksitas: 'Sedang', meanRed: 250, meanGreen: 220, meanBlue: 50, area: 700, brightness: 170, perimeter: 110},
    {nama: 'Lampu Isyarat', kategori: 'Non-Verbal', bentuk: 'Bulat', warna: 'Merah', ukuran: 'Sedang', tekstur: 'Logam', kompleksitas: 'Sedang', meanRed: 200, meanGreen: 50, meanBlue: 50, area: 1300, brightness: 100, perimeter: 145},
    {nama: 'Bendera', kategori: 'Non-Verbal', bentuk: 'Persegi', warna: 'Merah', ukuran: 'Sedang', tekstur: 'Kain', kompleksitas: 'Rendah', meanRed: 220, meanGreen: 60, meanBlue: 60, area: 1400, brightness: 110, perimeter: 155},
    {nama: 'Poster', kategori: 'Non-Verbal', bentuk: 'Persegi', warna: 'Beragam', ukuran: 'Besar', tekstur: 'Kertas', kompleksitas: 'Sedang', meanRed: 160, meanGreen: 140, meanBlue: 100, area: 2000, brightness: 140, perimeter: 200},
    {nama: 'Rambu Lalu Lintas', kategori: 'Non-Verbal', bentuk: 'Segitiga', warna: 'Merah', ukuran: 'Besar', tekstur: 'Logam', kompleksitas: 'Sedang', meanRed: 210, meanGreen: 40, meanBlue: 40, area: 1800, brightness: 95, perimeter: 180},
    {nama: 'Gestur Tangan', kategori: 'Non-Verbal', bentuk: 'Tidak_Beraturan', warna: 'Kulit', ukuran: 'Kecil', tekstur: 'Organik', kompleksitas: 'Tinggi', meanRed: 190, meanGreen: 150, meanBlue: 120, area: 900, brightness: 140, perimeter: 130},
    {nama: 'Ekspresi Wajah', kategori: 'Non-Verbal', bentuk: 'Bulat', warna: 'Kulit', ukuran: 'Sedang', tekstur: 'Organik', kompleksitas: 'Tinggi', meanRed: 200, meanGreen: 160, meanBlue: 130, area: 1100, brightness: 150, perimeter: 140}
];

// 🧠 GLOBAL VARIABLES
let currentStream = null;
let predictionHistory = JSON.parse(localStorage.getItem('plantHistory') || '[]');
let charts = {};
let currentTheme = 'light';
let totalPredictions = parseInt(localStorage.getItem('totalPredictions') || '0');
let userScore = parseInt(localStorage.getItem('userScore') || '0');

// 🎵 SOUND EFFECTS
const sounds = {
    success: () => playSuccessSound(),
    error: () => playErrorSound(),
    click: () => playClickSound(),
    upload: () => playUploadSound()
};

// 🎨 THEME MANAGEMENT
function toggleTheme() {
    currentTheme = currentTheme === 'light' ? 'dark' : 'light';
    document.body.classList.toggle('dark-theme');
    
    const icon = document.querySelector('.theme-toggle i');
    icon.className = currentTheme === 'dark' ? 'fas fa-sun' : 'fas fa-moon';
    
    localStorage.setItem('theme', currentTheme);
    showNotification('Theme berhasil diubah!', 'success');
    sounds.click();
}

// 🔊 SOUND FUNCTIONS
function playSuccessSound() {
    const audioContext = new (window.AudioContext || window.webkitAudioContext)();
    const oscillator = audioContext.createOscillator();
    const gainNode = audioContext.createGain();
    
    oscillator.connect(gainNode);
    gainNode.connect(audioContext.destination);
    
    oscillator.frequency.setValueAtTime(523.25, audioContext.currentTime); // C5
    oscillator.frequency.setValueAtTime(659.25, audioContext.currentTime + 0.1); // E5
    oscillator.frequency.setValueAtTime(783.99, audioContext.currentTime + 0.2); // G5
    
    gainNode.gain.setValueAtTime(0.3, audioContext.currentTime);
    gainNode.gain.exponentialRampToValueAtTime(0.01, audioContext.currentTime + 0.3);
    
    oscillator.start(audioContext.currentTime);
    oscillator.stop(audioContext.currentTime + 0.3);
}

function playErrorSound() {
    const audioContext = new (window.AudioContext || window.webkitAudioContext)();
    const oscillator = audioContext.createOscillator();
    const gainNode = audioContext.createGain();
    
    oscillator.connect(gainNode);
    gainNode.connect(audioContext.destination);
    
    oscillator.frequency.setValueAtTime(220, audioContext.currentTime); // A3
    oscillator.frequency.setValueAtTime(185, audioContext.currentTime + 0.15); // F#3
    
    gainNode.gain.setValueAtTime(0.3, audioContext.currentTime);
    gainNode.gain.exponentialRampToValueAtTime(0.01, audioContext.currentTime + 0.3);
    
    oscillator.start(audioContext.currentTime);
    oscillator.stop(audioContext.currentTime + 0.3);
}

function playClickSound() {
    const audioContext = new (window.AudioContext || window.webkitAudioContext)();
    const oscillator = audioContext.createOscillator();
    const gainNode = audioContext.createGain();
    
    oscillator.connect(gainNode);
    gainNode.connect(audioContext.destination);
    
    oscillator.frequency.setValueAtTime(800, audioContext.currentTime);
    gainNode.gain.setValueAtTime(0.1, audioContext.currentTime);
    gainNode.gain.exponentialRampToValueAtTime(0.01, audioContext.currentTime + 0.1);
    
    oscillator.start(audioContext.currentTime);
    oscillator.stop(audioContext.currentTime + 0.1);
}

function playUploadSound() {
    const audioContext = new (window.AudioContext || window.webkitAudioContext)();
    const oscillator = audioContext.createOscillator();
    const gainNode = audioContext.createGain();
    
    oscillator.connect(gainNode);
    gainNode.connect(audioContext.destination);
    
    oscillator.frequency.setValueAtTime(440, audioContext.currentTime);
    oscillator.frequency.setValueAtTime(554.37, audioContext.currentTime + 0.1);
    oscillator.frequency.setValueAtTime(659.25, audioContext.currentTime + 0.2);
    
    gainNode.gain.setValueAtTime(0.2, audioContext.currentTime);
    gainNode.gain.exponentialRampToValueAtTime(0.01, audioContext.currentTime + 0.3);
    
    oscillator.start(audioContext.currentTime);
    oscillator.stop(audioContext.currentTime + 0.3);
}

// 🔔 NOTIFICATION SYSTEM
function showNotification(message, type = 'info') {
    const notification = document.getElementById('notification');
    const content = document.getElementById('notificationContent');
    
    content.innerHTML = `
        <div style="display: flex; align-items: center; gap: 10px;">
            <i class="fas fa-${type === 'success' ? 'check-circle' : type === 'error' ? 'exclamation-circle' : 'info-circle'}"></i>
            <span>${message}</span>
        </div>
    `;
    
    notification.className = `notification ${type} show`;
    
    setTimeout(() => {
        notification.classList.remove('show');
    }, 3000);
}

// 🧮 ENCODING & NORMALIZATION
function encodeTanaman(tanaman) {
    const warnaMap = {'Hijau': 0, 'Tua': 1, 'Muda': 2};
    const bentukMap = {'Lonjong': 0, 'Bulat': 1, 'Lancip': 2};
    const aromaMap = {'Kuat': 0, 'Lemah': 1, 'Tidak Ada': 2};
    return [
        warnaMap[tanaman.warna] || 0,
        bentukMap[tanaman.bentuk] || 0,
        tanaman.tinggi || 0,
        aromaMap[tanaman.aroma] || 0
    ];
}

function normalizeFeatures(features) {
    const scales = [255, 255, 255, 5000, 255, 500, 1.0];
    return features.map((feature, index) => feature / (scales[index] || 1));
}

function euclidean(a, b) {
    let sum = 0;
    for (let i = 0; i < Math.min(a.length, b.length); i++) {
        sum += Math.pow(a[i] - b[i], 2);
    }
    return Math.sqrt(sum);
}

// 🤖 MACHINE LEARNING ALGORITHMS

// K-Nearest Neighbors
function knnPredict(input, k = 3, useExtended = false) {
    const dataToUse = useExtended ? extendedDataset : dataset;
    const inputEncoded = useExtended ? 
        normalizeFeatures([input.meanRed, input.meanGreen, input.meanBlue, input.area, input.brightness, input.perimeter, input.solidity]) :
        encodeTanaman(input);
    
    const distances = dataToUse.map(data => ({
        nama: data.nama,
        jenis: data.jenis,
        gambar: data.gambar,
        jarak: euclidean(inputEncoded, useExtended ? 
            normalizeFeatures([data.meanRed, data.meanGreen, data.meanBlue, data.area, data.brightness, data.perimeter, data.solidity]) :
            encodeTanaman(data))
    }));
    
    distances.sort((a, b) => a.jarak - b.jarak);
    const nearest = distances.slice(0, k);
    const count = { 'Herbal': 0, 'Non-Herbal': 0 };
    nearest.forEach(n => count[n.jenis]++);
    
    return {
        prediksi: count['Herbal'] > count['Non-Herbal'] ? 'Herbal' : 'Non-Herbal',
        confidence: Math.max(count['Herbal'], count['Non-Herbal']) / k,
        neighbors: nearest,
        algorithm: 'KNN',
        details: {
            k: k,
            totalDistance: nearest.reduce((sum, n) => sum + n.jarak, 0),
            votingBreakdown: count
        }
    };
}

// Variable Selection Optimization
function vsoPredict(input) {
    const weights = [0.25, 0.20, 0.20, 0.15, 0.10, 0.05, 0.05];
    const inputFeatures = normalizeFeatures([input.meanRed, input.meanGreen, input.meanBlue, input.area, input.brightness, input.perimeter || 0, input.solidity || 0]);
    
    let herbalScore = 0;
    let nonHerbalScore = 0;
    let detailedScores = [];
    
    extendedDataset.forEach(data => {
        const dataFeatures = normalizeFeatures([data.meanRed, data.meanGreen, data.meanBlue, data.area, data.brightness, data.perimeter, data.solidity]);
        let weightedDistance = 0;
        
        for (let i = 0; i < Math.min(inputFeatures.length, dataFeatures.length, weights.length); i++) {
            weightedDistance += weights[i] * Math.pow(inputFeatures[i] - dataFeatures[i], 2);
        }
        
        const similarity = 1 / (1 + Math.sqrt(weightedDistance));
        
        detailedScores.push({
            nama: data.nama,
            jenis: data.jenis,
            similarity: similarity,
            gambar: data.gambar
        });
        
        if (data.jenis === 'Herbal') {
            herbalScore += similarity;
        } else {
            nonHerbalScore += similarity;
        }
    });
    
    detailedScores.sort((a, b) => b.similarity - a.similarity);
    
    return {
        prediksi: herbalScore > nonHerbalScore ? 'Herbal' : 'Non-Herbal',
        confidence: Math.max(herbalScore, nonHerbalScore) / (herbalScore + nonHerbalScore),
        scores: { herbal: herbalScore, nonHerbal: nonHerbalScore },
        algorithm: 'VSO',
        details: {
            weights: weights,
            topMatches: detailedScores.slice(0, 3),
            weightedFeatures: inputFeatures.map((f, i) => f * weights[i])
        }
    };
}

// 🐦 PSO (Particle Swarm Optimization) for Communication Classification
function psoClassifyKomunikasi(input) {
    console.log('🐦 Running PSO Classification for Communication...');
    
    // PSO Parameters
    const numParticles = 20;
    const numIterations = 50;
    const w = 0.7; // Inertia weight
    const c1 = 1.5; // Cognitive parameter
    const c2 = 1.5; // Social parameter
    const numFeatures = 6; // meanRed, meanGreen, meanBlue, area, brightness, perimeter
    
    // Initialize particles
    let particles = [];
    let globalBest = { position: null, fitness: -Infinity };
    
    // Create initial particles
    for (let i = 0; i < numParticles; i++) {
        let particle = {
            position: Array(numFeatures).fill(0).map(() => Math.random()),
            velocity: Array(numFeatures).fill(0).map(() => (Math.random() - 0.5) * 0.1),
            bestPosition: null,
            bestFitness: -Infinity
        };
        
        // Evaluate initial fitness
        particle.bestFitness = evaluatePSOFitness(particle.position, input);
        particle.bestPosition = [...particle.position];
        
        if (particle.bestFitness > globalBest.fitness) {
            globalBest.fitness = particle.bestFitness;
            globalBest.position = [...particle.position];
        }
        
        particles.push(particle);
    }
    
    // PSO Main Loop
    for (let iter = 0; iter < numIterations; iter++) {
        for (let i = 0; i < numParticles; i++) {
            let particle = particles[i];
            
            // Update velocity
            for (let j = 0; j < numFeatures; j++) {
                let r1 = Math.random();
                let r2 = Math.random();
                
                particle.velocity[j] = w * particle.velocity[j] +
                    c1 * r1 * (particle.bestPosition[j] - particle.position[j]) +
                    c2 * r2 * (globalBest.position[j] - particle.position[j]);
                
                // Apply velocity constraints
                particle.velocity[j] = Math.max(-0.1, Math.min(0.1, particle.velocity[j]));
            }
            
            // Update position
            for (let j = 0; j < numFeatures; j++) {
                particle.position[j] += particle.velocity[j];
                particle.position[j] = Math.max(0, Math.min(1, particle.position[j]));
            }
            
            // Evaluate fitness
            let fitness = evaluatePSOFitness(particle.position, input);
            
            // Update personal best
            if (fitness > particle.bestFitness) {
                particle.bestFitness = fitness;
                particle.bestPosition = [...particle.position];
            }
            
            // Update global best
            if (fitness > globalBest.fitness) {
                globalBest.fitness = fitness;
                globalBest.position = [...particle.position];
            }
        }
    }
    
    // Use optimal weights to classify
    const result = classifyWithPSOWeights(globalBest.position, input);
    
    return {
        prediksi: result.prediction,
        confidence: result.confidence,
        scores: result.scores,
        algorithm: 'PSO',
        details: {
            optimalWeights: globalBest.position,
            fitness: globalBest.fitness,
            iterations: numIterations,
            topMatches: result.topMatches,
            convergenceInfo: `Converged after ${numIterations} iterations with fitness: ${globalBest.fitness.toFixed(4)}`
        }
    };
}

// PSO Fitness Evaluation Function
function evaluatePSOFitness(weights, input) {
    const inputFeatures = normalizeFeatures([
        input.meanRed || 0, input.meanGreen || 0, input.meanBlue || 0,
        input.area || 0, input.brightness || 0, input.perimeter || 0
    ]);
    
    let correctClassifications = 0;
    let totalSamples = komunikasiDataset.length;
    
    // Test on communication dataset
    komunikasiDataset.forEach(data => {
        const dataFeatures = normalizeFeatures([
            data.meanRed, data.meanGreen, data.meanBlue,
            data.area, data.brightness, data.perimeter
        ]);
        
        let weightedDistance = 0;
        for (let i = 0; i < inputFeatures.length && i < weights.length; i++) {
            weightedDistance += weights[i] * Math.pow(inputFeatures[i] - dataFeatures[i], 2);
        }
        
        const similarity = 1 / (1 + Math.sqrt(weightedDistance));
        
        // Simple classification based on similarity threshold
        const predictedClass = similarity > 0.5 ? data.kategori : (data.kategori === 'Verbal' ? 'Non-Verbal' : 'Verbal');
        
        if (predictedClass === data.kategori) {
            correctClassifications++;
        }
    });
    
    return correctClassifications / totalSamples;
}

// Classify using PSO optimized weights
function classifyWithPSOWeights(weights, input) {
    const inputFeatures = normalizeFeatures([
        input.meanRed || 0, input.meanGreen || 0, input.meanBlue || 0,
        input.area || 0, input.brightness || 0, input.perimeter || 0
    ]);
    
    let verbalScore = 0;
    let nonVerbalScore = 0;
    let detailedScores = [];
    
    komunikasiDataset.forEach(data => {
        const dataFeatures = normalizeFeatures([
            data.meanRed, data.meanGreen, data.meanBlue,
            data.area, data.brightness, data.perimeter
        ]);
        
        let weightedDistance = 0;
        for (let i = 0; i < inputFeatures.length && i < weights.length; i++) {
            weightedDistance += weights[i] * Math.pow(inputFeatures[i] - dataFeatures[i], 2);
        }
        
        const similarity = 1 / (1 + Math.sqrt(weightedDistance));
        
        detailedScores.push({
            nama: data.nama,
            kategori: data.kategori,
            similarity: similarity
        });
        
        if (data.kategori === 'Verbal') {
            verbalScore += similarity;
        } else {
            nonVerbalScore += similarity;
        }
    });
    
    detailedScores.sort((a, b) => b.similarity - a.similarity);
    
    return {
        prediction: verbalScore > nonVerbalScore ? 'Verbal' : 'Non-Verbal',
        confidence: Math.max(verbalScore, nonVerbalScore) / (verbalScore + nonVerbalScore),
        scores: { verbal: verbalScore, nonVerbal: nonVerbalScore },
        topMatches: detailedScores.slice(0, 3)
    };
}

// 🤖 KNN for Communication Classification
function knnClassifyKomunikasi(input, k = 3) {
    console.log('🤖 Running KNN Classification for Communication...');
    
    const inputFeatures = normalizeFeatures([
        input.meanRed || 0, input.meanGreen || 0, input.meanBlue || 0,
        input.area || 0, input.brightness || 0, input.perimeter || 0
    ]);
    
    let distances = [];
    
    // Calculate distances to all training samples
    komunikasiDataset.forEach(data => {
        const dataFeatures = normalizeFeatures([
            data.meanRed, data.meanGreen, data.meanBlue,
            data.area, data.brightness, data.perimeter
        ]);
        
        let distance = 0;
        for (let i = 0; i < inputFeatures.length; i++) {
            distance += Math.pow(inputFeatures[i] - dataFeatures[i], 2);
        }
        distance = Math.sqrt(distance);
        
        distances.push({
            nama: data.nama,
            kategori: data.kategori,
            distance: distance,
            similarity: 1 / (1 + distance)
        });
    });
    
    // Sort by distance and get k nearest neighbors
    distances.sort((a, b) => a.distance - b.distance);
    const neighbors = distances.slice(0, k);
    
    // Voting mechanism
    let verbalCount = 0;
    let nonVerbalCount = 0;
    let verbalScore = 0;
    let nonVerbalScore = 0;
    
    neighbors.forEach(neighbor => {
        if (neighbor.kategori === 'Verbal') {
            verbalCount++;
            verbalScore += neighbor.similarity;
        } else {
            nonVerbalCount++;
            nonVerbalScore += neighbor.similarity;
        }
    });
    
    const prediction = verbalCount > nonVerbalCount ? 'Verbal' : 'Non-Verbal';
    const confidence = Math.max(verbalScore, nonVerbalScore) / (verbalScore + nonVerbalScore);
    
    return {
        prediksi: prediction,
        confidence: confidence,
        scores: { verbal: verbalScore, nonVerbal: nonVerbalScore },
        algorithm: 'KNN',
        details: {
            k: k,
            neighbors: neighbors,
            voting: { verbal: verbalCount, nonVerbal: nonVerbalCount }
        }
    };
}

// Support Vector Machine (Simplified)
function svmPredict(input) {
    const inputFeatures = normalizeFeatures([input.meanRed, input.meanGreen, input.meanBlue, input.area, input.brightness]);
    
    // Simplified SVM with linear kernel
    const weights = [0.3, -0.2, 0.4, 0.1, 0.2];
    const bias = -0.1;
    
    let score = bias;
    for (let i = 0; i < inputFeatures.length; i++) {
        score += weights[i] * inputFeatures[i];
    }
    
    const confidence = 1 / (1 + Math.exp(-Math.abs(score))); // Sigmoid activation
    
    return {
        prediksi: score > 0 ? 'Herbal' : 'Non-Herbal',
        confidence: confidence,
        algorithm: 'SVM',
        details: {
            score: score,
            weights: weights,
            bias: bias,
            features: inputFeatures
        }
    };
}

// Neural Network Simulator
function neuralNetworkPredict(input) {
    const inputFeatures = normalizeFeatures([input.meanRed, input.meanGreen, input.meanBlue, input.area, input.brightness]);
    
    // Hidden layer (3 neurons)
    const hiddenWeights = [
        [0.5, -0.3, 0.8, 0.2, -0.1],
        [-0.2, 0.7, -0.4, 0.6, 0.3],
        [0.1, 0.4, -0.6, -0.2, 0.9]
    ];
    const hiddenBias = [0.1, -0.2, 0.3];
    
    // Calculate hidden layer output
    const hiddenOutput = hiddenWeights.map((weights, i) => {
        let sum = hiddenBias[i];
        for (let j = 0; j < inputFeatures.length; j++) {
            sum += weights[j] * inputFeatures[j];
        }
        return 1 / (1 + Math.exp(-sum)); // Sigmoid activation
    });
    
    // Output layer
    const outputWeights = [0.8, -0.5, 0.3];
    const outputBias = -0.1;
    
    let outputSum = outputBias;
    for (let i = 0; i < hiddenOutput.length; i++) {
        outputSum += outputWeights[i] * hiddenOutput[i];
    }
    
    const output = 1 / (1 + Math.exp(-outputSum));
    
    return {
        prediksi: output > 0.5 ? 'Herbal' : 'Non-Herbal',
        confidence: Math.abs(output - 0.5) * 2,
        algorithm: 'Neural Network',
        details: {
            hiddenLayer: hiddenOutput,
            outputRaw: output,
            layers: {
                input: inputFeatures,
                hidden: hiddenOutput,
                output: output
            }
        }
    };
}

// 📷 IMAGE PROCESSING
function extractImageFeatures(imageData) {
    let r = 0, g = 0, b = 0, pixelCount = 0;
    let minR = 255, maxR = 0, minG = 255, maxG = 0, minB = 255, maxB = 0;
    
    for (let i = 0; i < imageData.data.length; i += 4) {
        const red = imageData.data[i];
        const green = imageData.data[i + 1];
        const blue = imageData.data[i + 2];
        
        r += red;
        g += green;
        b += blue;
        pixelCount++;
        
        minR = Math.min(minR, red);
        maxR = Math.max(maxR, red);
        minG = Math.min(minG, green);
        maxG = Math.max(maxG, green);
        minB = Math.min(minB, blue);
        maxB = Math.max(maxB, blue);
    }
    
    const meanRed = r / pixelCount;
    const meanGreen = g / pixelCount;
    const meanBlue = b / pixelCount;
    const brightness = (meanRed + meanGreen + meanBlue) / 3;
    
    // Calculate variance for texture analysis
    let varianceR = 0, varianceG = 0, varianceB = 0;
    for (let i = 0; i < imageData.data.length; i += 4) {
        varianceR += Math.pow(imageData.data[i] - meanRed, 2);
        varianceG += Math.pow(imageData.data[i + 1] - meanGreen, 2);
        varianceB += Math.pow(imageData.data[i + 2] - meanBlue, 2);
    }
    
    return {
        meanRed: meanRed,
        meanGreen: meanGreen,
        meanBlue: meanBlue,
        area: pixelCount / 100,
        brightness: brightness,
        perimeter: Math.sqrt(pixelCount) * 4, // Simplified perimeter estimation
        solidity: 0.75 + (brightness / 255) * 0.25, // Estimated solidity
        contrast: {
            red: maxR - minR,
            green: maxG - minG,
            blue: maxB - minB
        },
        variance: {
            red: varianceR / pixelCount,
            green: varianceG / pixelCount,
            blue: varianceB / pixelCount
        }
    };
}

// 📊 CHART FUNCTIONS
function createConfidenceChart(result) {
    const ctx = document.getElementById('confidenceChart').getContext('2d');
    
    if (charts.confidence) {
        charts.confidence.destroy();
    }
    
    charts.confidence = new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: ['Confidence', 'Uncertainty'],
            datasets: [{
                data: [result.confidence * 100, (1 - result.confidence) * 100],
                backgroundColor: [
                    result.prediksi === 'Herbal' ? '#2ed573' : '#ff6b6b',
                    '#e0e0e0'
                ],
                borderWidth: 0
            }]
        },
        options: {
            responsive: true,
            plugins: {
                title: {
                    display: true,
                    text: `Confidence Score - ${result.algorithm}`
                },
                legend: {
                    position: 'bottom'
                }
            }
        }
    });
}

function createFeatureChart(features) {
    const ctx = document.getElementById('featureChart').getContext('2d');
    
    if (charts.feature) {
        charts.feature.destroy();
    }
    
    charts.feature = new Chart(ctx, {
        type: 'radar',
        data: {
            labels: ['Red', 'Green', 'Blue', 'Area', 'Brightness'],
            datasets: [{
                label: 'Input Features',
                data: [
                    features.meanRed / 255,
                    features.meanGreen / 255,
                    features.meanBlue / 255,
                    features.area / 1000,
                    features.brightness / 255
                ],
                borderColor: '#667eea',
                backgroundColor: 'rgba(102, 126, 234, 0.2)',
                pointBackgroundColor: '#667eea'
            }]
        },
        options: {
            responsive: true,
            plugins: {
                title: {
                    display: true,
                    text: 'Feature Analysis'
                }
            },
            scales: {
                r: {
                    beginAtZero: true,
                    max: 1
                }
            }
        }
    });
}

function createComparisonChart(results) {
    const ctx = document.getElementById('comparisonChart').getContext('2d');
    
    if (charts.comparison) {
        charts.comparison.destroy();
    }
    
    charts.comparison = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: results.map(r => r.algorithm),
            datasets: [{
                label: 'Confidence',
                data: results.map(r => r.confidence * 100),
                backgroundColor: [
                    '#667eea',
                    '#764ba2',
                    '#4ecdc4',
                    '#45b7d1'
                ]
            }]
        },
        options: {
            responsive: true,
            plugins: {
                title: {
                    display: true,
                    text: 'Algorithm Comparison'
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    max: 100
                }
            }
        }
    });
}

// 💾 HISTORY MANAGEMENT
function saveToHistory(result, input, features = null) {
    const historyItem = {
        id: Date.now(),
        timestamp: new Date().toLocaleString(),
        result: result,
        input: input,
        features: features,
        score: Math.round(result.confidence * 100)
    };
    
    predictionHistory.unshift(historyItem);
    if (predictionHistory.length > 50) {
        predictionHistory = predictionHistory.slice(0, 50);
    }
    
    localStorage.setItem('plantHistory', JSON.stringify(predictionHistory));
    updateHistoryDisplay();
    updateUserStats();
}

function updateHistoryDisplay() {
    const historyList = document.getElementById('historyList');
    historyList.innerHTML = '';
    
    predictionHistory.forEach(item => {
        const div = document.createElement('div');
        div.className = 'history-item';
        div.innerHTML = `
            <div style="display: flex; justify-content: between; align-items: center;">
                <div>
                    <strong>${item.result.prediksi}</strong> (${item.result.algorithm})
                    <br>
                    <small>${item.timestamp}</small>
                    <br>
                    <span style="color: #667eea;">Confidence: ${(item.result.confidence * 100).toFixed(1)}%</span>
                </div>
                <div style="text-align: right;">
                    <div style="font-size: 1.5rem;">${item.result.prediksi === 'Herbal' ? '🌿' : '🥬'}</div>
                    <button onclick="deleteHistoryItem(${item.id})" style="background: #ff6b6b; padding: 5px 10px; font-size: 0.8rem;">
                        <i class="fas fa-trash"></i>
                    </button>
                </div>
            </div>
        `;
        historyList.appendChild(div);
    });
}

function deleteHistoryItem(id) {
    predictionHistory = predictionHistory.filter(item => item.id !== id);
    localStorage.setItem('plantHistory', JSON.stringify(predictionHistory));
    updateHistoryDisplay();
    showNotification('Item dihapus dari riwayat', 'success');
    sounds.click();
}

function clearHistory() {
    if (confirm('Yakin ingin menghapus semua riwayat?')) {
        predictionHistory = [];
        localStorage.setItem('plantHistory', JSON.stringify(predictionHistory));
        updateHistoryDisplay();
        showNotification('Riwayat berhasil dihapus!', 'success');
        sounds.click();
    }
}

function exportHistory() {
    if (predictionHistory.length === 0) {
        showNotification('Tidak ada data untuk diekspor', 'error');
        return;
    }
    
    let csv = 'Timestamp,Prediksi,Algorithm,Confidence,Input\n';
    predictionHistory.forEach(item => {
        csv += `"${item.timestamp}","${item.result.prediksi}","${item.result.algorithm}","${(item.result.confidence * 100).toFixed(2)}%","${JSON.stringify(item.input).replace(/"/g, '""')}"\n`;
    });
    
    const blob = new Blob([csv], { type: 'text/csv' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `plant_predictions_${new Date().toISOString().split('T')[0]}.csv`;
    a.click();
    window.URL.revokeObjectURL(url);
    
    showNotification('Riwayat berhasil diekspor!', 'success');
    sounds.success();
}

function updateUserStats() {
    totalPredictions++;
    if (predictionHistory[0] && predictionHistory[0].result.confidence > 0.8) {
        userScore += 10;
    } else if (predictionHistory[0] && predictionHistory[0].result.confidence > 0.6) {
        userScore += 5;
    } else {
        userScore += 1;
    }
    
    localStorage.setItem('totalPredictions', totalPredictions.toString());
    localStorage.setItem('userScore', userScore.toString());
    
    // Update display somewhere if needed
    document.title = `🌿 AI Plant Classifier Pro - Score: ${userScore}`;
}

// 🎥 CAMERA FUNCTIONS
async function startCamera() {
    try {
        currentStream = await navigator.mediaDevices.getUserMedia({ 
            video: { 
                width: { ideal: 640 },
                height: { ideal: 480 },
                facingMode: 'environment' 
            } 
        });
        
        const video = document.getElementById('videoElement');
        video.srcObject = currentStream;
        video.style.display = 'block';
        video.play();
        
        document.getElementById('startCamera').style.display = 'none';
        document.getElementById('capturePhoto').style.display = 'inline-block';
        document.getElementById('stopCamera').style.display = 'inline-block';
        
        showNotification('Kamera berhasil dimulai!', 'success');
        sounds.success();
    } catch (error) {
        showNotification('Error mengakses kamera: ' + error.message, 'error');
        sounds.error();
    }
}

function capturePhoto() {
    const video = document.getElementById('videoElement');
    const canvas = document.getElementById('captureCanvas');
    const ctx = canvas.getContext('2d');
    
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    ctx.drawImage(video, 0, 0);
    
    const imageDataURL = canvas.toDataURL('image/jpeg', 0.8);
    
    // Create image element for preview
    const img = document.createElement('img');
    img.src = imageDataURL;
    img.onload = function() {
        const imagePreview = document.getElementById('imagePreview');
        imagePreview.innerHTML = '';
        imagePreview.appendChild(img);
        
        // Enable predict button
        document.getElementById('predictButton').disabled = false;
        
        showNotification('Foto berhasil diambil!', 'success');
        sounds.upload();
        
        // Switch to upload tab to see the result
        switchTab('upload');
    };
}

function stopCamera() {
    if (currentStream) {
        currentStream.getTracks().forEach(track => track.stop());
        currentStream = null;
    }
    
    const video = document.getElementById('videoElement');
    video.style.display = 'none';
    video.srcObject = null;
    
    document.getElementById('startCamera').style.display = 'inline-block';
    document.getElementById('capturePhoto').style.display = 'none';
    document.getElementById('stopCamera').style.display = 'none';
    
    showNotification('Kamera dihentikan', 'info');
    sounds.click();
}

// 🎯 TAB MANAGEMENT
function switchTab(tabName) {
    sounds.click();
    
    document.querySelectorAll('.tab-content').forEach(tab => tab.classList.remove('active'));
    document.querySelectorAll('.tab-button').forEach(btn => btn.classList.remove('active'));
    
    document.getElementById(tabName + 'Tab').classList.add('active');
    event.target.classList.add('active');
    
    // Reset form states when switching tabs
    if (tabName !== 'camera' && currentStream) {
        stopCamera();
    }
    
    // Clear results when switching tabs
    document.getElementById('hasilPrediksi').innerHTML = '';
    document.getElementById('detailHasil').innerHTML = '';
}

// 🎊 PARTICLES INITIALIZATION
function initParticles() {
    particlesJS('particles-js', {
        particles: {
            number: { value: 80, density: { enable: true, value_area: 800 } },
            color: { value: "#ffffff" },
            shape: { type: "circle" },
            opacity: { value: 0.5, random: false },
            size: { value: 3, random: true },
            line_linked: { enable: true, distance: 150, color: "#ffffff", opacity: 0.4, width: 1 },
            move: { enable: true, speed: 6, direction: "none", random: false, straight: false, out_mode: "out", bounce: false }
        },
        interactivity: {
            detect_on: "canvas",
            events: { onhover: { enable: true, mode: "repulse" }, onclick: { enable: true, mode: "push" }, resize: true },
            modes: { grab: { distance: 400, line_linked: { opacity: 1 } }, bubble: { distance: 400, size: 40, duration: 2, opacity: 8, speed: 3 }, repulse: { distance: 200, duration: 0.4 }, push: { particles_nb: 4 }, remove: { particles_nb: 2 } }
        },
        retina_detect: true
    });
}

// 📱 UPLOAD FUNCTIONALITY
const uploadArea = document.getElementById('uploadArea');
const imageInput = document.getElementById('imageInput');
const imagePreview = document.getElementById('imagePreview');
const predictButton = document.getElementById('predictButton');

uploadArea.addEventListener('click', () => {
    imageInput.click();
    sounds.click();
});

uploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadArea.classList.add('dragover');
});

uploadArea.addEventListener('dragleave', () => {
    uploadArea.classList.remove('dragover');
});

uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadArea.classList.remove('dragover');
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
    if (file.size > 5 * 1024 * 1024) {
        showNotification('File terlalu besar! Maksimal 5MB', 'error');
        sounds.error();
        return;
    }
    
    if (!file.type.startsWith('image/')) {
        showNotification('File harus berupa gambar!', 'error');
        sounds.error();
        return;
    }
    
    const reader = new FileReader();
    reader.onload = function(e) {
        const img = document.createElement('img');
        img.src = e.target.result;
        img.onload = function() {
            imagePreview.innerHTML = '';
            imagePreview.appendChild(img);
            predictButton.disabled = false;
            
            showNotification('Gambar berhasil diupload!', 'success');
            sounds.upload();
        };
    };
    reader.readAsDataURL(file);
}

// 📋 FORM EVENT HANDLERS
const form = document.getElementById('formTanaman');
const uploadForm = document.getElementById('formUpload');
const hasil = document.getElementById('hasilPrediksi');
const detailHasil = document.getElementById('detailHasil');
const confidenceBar = document.getElementById('confidenceBar');
const confidenceFill = document.getElementById('confidenceFill');
const chartsContainer = document.getElementById('chartsContainer');

// Manual form submission
form.addEventListener('submit', function(e) {
    e.preventDefault();
    sounds.click();
    
    const input = {
        warna: document.getElementById('warna').value,
        bentuk: document.getElementById('bentuk').value,
        tinggi: parseInt(document.getElementById('tinggi').value),
        aroma: document.getElementById('aroma').value
    };
    
    if (!input.warna || !input.bentuk || !input.tinggi || !input.aroma) {
        showNotification('Mohon lengkapi semua field!', 'error');
        sounds.error();
        return;
    }
    
    // Show loading
    hasil.innerHTML = '<div class="loading"></div> Menganalisis data...';
    detailHasil.innerHTML = '';
    chartsContainer.style.display = 'none';
    
    setTimeout(() => {
        const result = knnPredict(input);
        displayResult(result, input);
        saveToHistory(result, input);
        sounds.success();
    }, 1500);
});

// Upload form submission
uploadForm.addEventListener('submit', function(e) {
    e.preventDefault();
    sounds.click();
    
    const img = imagePreview.querySelector('img');
    if (!img) {
        showNotification('Mohon upload gambar terlebih dahulu!', 'error');
        sounds.error();
        return;
    }
    
    hasil.innerHTML = '<div class="loading"></div> Menganalisis gambar...';
    detailHasil.innerHTML = '';
    chartsContainer.style.display = 'none';
    
    // Simulate image processing
    setTimeout(() => {
        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d');
        canvas.width = img.naturalWidth || img.width;
        canvas.height = img.naturalHeight || img.height;
        ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
        
        const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
        const features = extractImageFeatures(imageData);
        
        const algorithm = document.getElementById('algorithm').value;
        let result;
        
        switch (algorithm) {
            case 'knn':
                result = knnPredict(features, 3, true);
                break;
            case 'vso':
                result = vsoPredict(features);
                break;
            case 'svm':
                result = svmPredict(features);
                break;
            case 'neural':
                result = neuralNetworkPredict(features);
                break;
            default:
                result = knnPredict(features, 3, true);
        }
        
        displayResult(result, features, features);
        saveToHistory(result, features, features);
        sounds.success();
    }, 2000);
});

function displayResult(result, input, features = null) {
    // Display main result
    hasil.className = `result ${result.prediksi.toLowerCase().replace('-', '-')}`;
    hasil.innerHTML = `
        <div style="font-size: 2rem; margin-bottom: 15px;">
            ${result.prediksi === 'Herbal' ? '🌿' : '🥬'} <strong>${result.prediksi}</strong>
        </div>
        <div style="font-size: 1.2rem; color: #666; margin-bottom: 10px;">
            Algorithm: ${result.algorithm}
        </div>
        <div style="font-size: 1rem; color: #888;">
            Confidence: ${(result.confidence * 100).toFixed(1)}%
        </div>
    `;
    
    // Show confidence bar
    confidenceBar.style.display = 'block';
    confidenceFill.style.width = (result.confidence * 100) + '%';
    
    // Display detailed results
    let detailHTML = `
        <h4><i class="fas fa-analytics"></i> Detail Analisis ${result.algorithm}:</h4>
    `;
    
    if (features) {
        detailHTML += `
            <h5>🔬 Fitur yang Diekstrak:</h5>
            <div class="feature-grid">
                <div class="feature-item">
                    <strong>Mean Red:</strong> ${features.meanRed?.toFixed(1) || 'N/A'}
                </div>
                <div class="feature-item">
                    <strong>Mean Green:</strong> ${features.meanGreen?.toFixed(1) || 'N/A'}
                </div>
                <div class="feature-item">
                    <strong>Mean Blue:</strong> ${features.meanBlue?.toFixed(1) || 'N/A'}
                </div>
                <div class="feature-item">
                    <strong>Area:</strong> ${features.area?.toFixed(1) || 'N/A'}
                </div>
                <div class="feature-item">
                    <strong>Brightness:</strong> ${features.brightness?.toFixed(1) || 'N/A'}
                </div>
                <div class="feature-item">
                    <strong>Perimeter:</strong> ${features.perimeter?.toFixed(1) || 'N/A'}
                </div>
            </div>
        `;
    }
    
    if (result.algorithm === 'KNN' && result.neighbors) {
        detailHTML += `
            <h5>🎯 Tetangga Terdekat (K=${result.details.k}):</h5>
            <div class="feature-grid">
                ${result.neighbors.map((neighbor, index) => `
                    <div class="feature-item">
                        <strong>Rank ${index + 1}:</strong> ${neighbor.nama}<br>
                        <small>Jarak: ${neighbor.jarak.toFixed(3)}</small><br>
                        <small>Jenis: ${neighbor.jenis}</small>
                        ${neighbor.gambar ? `<br><img src="${neighbor.gambar}" style="width: 50px; height: 50px; object-fit: cover; border-radius: 5px; margin-top: 5px;" onerror="this.style.display='none'">` : ''}
                    </div>
                `).join('')}
            </div>
        `;
    }
    
    if (result.algorithm === 'VSO' && result.details) {
        detailHTML += `
            <h5>⚡ Skor VSO:</h5>
            <div class="feature-grid">
                <div class="feature-item">
                    <strong>Herbal Score:</strong> ${result.scores.herbal.toFixed(3)}
                </div>
                <div class="feature-item">
                    <strong>Non-Herbal Score:</strong> ${result.scores.nonHerbal.toFixed(3)}
                </div>
            </div>
            <h5>🔝 Top Matches:</h5>
            <div class="feature-grid">
                ${result.details.topMatches.map((match, index) => `
                    <div class="feature-item">
                        <strong>${match.nama}</strong><br>
                        <small>Similarity: ${match.similarity.toFixed(3)}</small><br>
                        <small>Jenis: ${match.jenis}</small>
                        ${match.gambar ? `<br><img src="${match.gambar}" style="width: 50px; height: 50px; object-fit: cover; border-radius: 5px; margin-top: 5px;" onerror="this.style.display='none'">` : ''}
                    </div>
                `).join('')}
            </div>
        `;
    }
    
    if (result.algorithm === 'SVM' && result.details) {
        detailHTML += `
            <h5>🎯 SVM Analysis:</h5>
            <div class="feature-grid">
                <div class="feature-item">
                    <strong>Decision Score:</strong> ${result.details.score.toFixed(3)}
                </div>
                <div class="feature-item">
                    <strong>Bias:</strong> ${result.details.bias}
                </div>
            </div>
        `;
    }
    
    if (result.algorithm === 'Neural Network' && result.details) {
        detailHTML += `
            <h5>🧠 Neural Network:</h5>
            <div class="feature-grid">
                <div class="feature-item">
                    <strong>Hidden Layer 1:</strong> ${result.details.hiddenLayer[0].toFixed(3)}
                </div>
                <div class="feature-item">
                    <strong>Hidden Layer 2:</strong> ${result.details.hiddenLayer[1].toFixed(3)}
                </div>
                <div class="feature-item">
                    <strong>Hidden Layer 3:</strong> ${result.details.hiddenLayer[2].toFixed(3)}
                </div>
                <div class="feature-item">
                    <strong>Output Raw:</strong> ${result.details.outputRaw.toFixed(3)}
                </div>
            </div>
        `;
    }
    
    detailHasil.innerHTML = detailHTML;
    
    // Show charts
    chartsContainer.style.display = 'block';
    
    setTimeout(() => {
        createConfidenceChart(result);
        if (features) {
            createFeatureChart(features);
        }
        
        // Run multiple algorithms for comparison if features available
        if (features) {
            const algorithms = ['knn', 'vso', 'svm', 'neural'];
            const results = algorithms.map(alg => {
                switch (alg) {
                    case 'knn': return knnPredict(features, 3, true);
                    case 'vso': return vsoPredict(features);
                    case 'svm': return svmPredict(features);
                    case 'neural': return neuralNetworkPredict(features);
                    default: return result;
                }
            });
            createComparisonChart(results);
        }
    }, 100);
}

// 🚀 INITIALIZATION
document.addEventListener('DOMContentLoaded', function() {
    // Initialize particles
    initParticles();
    
    // Load saved theme
    const savedTheme = localStorage.getItem('theme');
    if (savedTheme === 'dark') {
        toggleTheme();
    }
    
    // Initialize history
    updateHistoryDisplay();
    
    // Show welcome notification
    setTimeout(() => {
        showNotification('🌿 Selamat datang di AI Plant Classifier Pro!', 'success');
    }, 1000);
    
    // Add click sounds to all buttons
    document.querySelectorAll('button').forEach(btn => {
        btn.addEventListener('click', () => sounds.click());
    });
    
    // 🗣️ KOMUNIKASI EVENT HANDLERS
    // Image upload for communication
    const imageInputKomunikasi = document.getElementById('imageInputKomunikasi');
    const uploadAreaKomunikasi = document.getElementById('uploadAreaKomunikasi');
    const previewKomunikasi = document.getElementById('previewKomunikasi');
    const previewImageKomunikasi = document.getElementById('previewImageKomunikasi');
    const predictKomunikasiButton = document.getElementById('predictKomunikasiButton');
    
    if (imageInputKomunikasi) {
        imageInputKomunikasi.addEventListener('change', handleKomunikasiImageUpload);
    }
    
    if (uploadAreaKomunikasi) {
        // Drag and drop for communication
        uploadAreaKomunikasi.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadAreaKomunikasi.classList.add('dragover');
        });
        
        uploadAreaKomunikasi.addEventListener('dragleave', () => {
            uploadAreaKomunikasi.classList.remove('dragover');
        });
        
        uploadAreaKomunikasi.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadAreaKomunikasi.classList.remove('dragover');
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                handleKomunikasiImageFile(files[0]);
            }
        });
    }
    
    // Communication form submission
    const formKomunikasi = document.getElementById('formKomunikasi');
    if (formKomunikasi) {
        formKomunikasi.addEventListener('submit', handleKomunikasiPrediction);
    }
    
    console.log('🌿 AI Plant Classifier Pro - Super Advanced loaded successfully!');
    console.log(`📊 Total predictions: ${totalPredictions}`);
    console.log(`🏆 User score: ${userScore}`);
});

// 🗣️ COMMUNICATION IMAGE HANDLERS
function handleKomunikasiImageUpload(event) {
    const file = event.target.files[0];
    if (file) {
        handleKomunikasiImageFile(file);
    }
}

function handleKomunikasiImageFile(file) {
    if (!file.type.startsWith('image/')) {
        showNotification('❌ Mohon pilih file gambar yang valid!', 'error');
        return;
    }
    
    const reader = new FileReader();
    reader.onload = function(e) {
        const previewKomunikasi = document.getElementById('previewKomunikasi');
        const previewImageKomunikasi = document.getElementById('previewImageKomunikasi');
        const uploadAreaKomunikasi = document.getElementById('uploadAreaKomunikasi');
        const predictKomunikasiButton = document.getElementById('predictKomunikasiButton');
        
        previewImageKomunikasi.src = e.target.result;
        previewKomunikasi.style.display = 'block';
        uploadAreaKomunikasi.querySelector('.upload-content').style.display = 'none';
        predictKomunikasiButton.disabled = false;
        
        sounds.upload();
        showNotification('✅ Gambar komunikasi berhasil diupload!', 'success');
    };
    reader.readAsDataURL(file);
}

function removeImageKomunikasi() {
    const previewKomunikasi = document.getElementById('previewKomunikasi');
    const uploadAreaKomunikasi = document.getElementById('uploadAreaKomunikasi');
    const predictKomunikasiButton = document.getElementById('predictKomunikasiButton');
    const imageInputKomunikasi = document.getElementById('imageInputKomunikasi');
    
    previewKomunikasi.style.display = 'none';
    uploadAreaKomunikasi.querySelector('.upload-content').style.display = 'block';
    predictKomunikasiButton.disabled = true;
    imageInputKomunikasi.value = '';
}

// 🗣️ COMMUNICATION PREDICTION HANDLER
function handleKomunikasiPrediction(event) {
    event.preventDefault();
    
    const previewImageKomunikasi = document.getElementById('previewImageKomunikasi');
    const algorithmKomunikasi = document.getElementById('algorithmKomunikasi').value;
    
    if (!previewImageKomunikasi.src && !hasManualKomunikasiInput()) {
        showNotification('❌ Mohon upload gambar atau isi fitur manual!', 'error');
        return;
    }
    
    showNotification('🔄 Menganalisis komunikasi...', 'info');
    
    // Extract features from image or use manual input
    let features = getKomunikasiFeatures();
    
    if (previewImageKomunikasi.src && previewImageKomunikasi.src !== '') {
        // Extract features from uploaded image
        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d');
        canvas.width = previewImageKomunikasi.naturalWidth;
        canvas.height = previewImageKomunikasi.naturalHeight;
        ctx.drawImage(previewImageKomunikasi, 0, 0);
        
        const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
        features = extractImageFeatures(imageData);
    }
    
    // Run classification
    let result;
    switch (algorithmKomunikasi) {
        case 'knn-komunikasi':
            result = knnClassifyKomunikasi(features);
            break;
        case 'pso-komunikasi':
            result = psoClassifyKomunikasi(features);
            break;
        default:
            result = knnClassifyKomunikasi(features);
    }
    
    // Display results
    displayKomunikasiResults(result, features);
    
    // Save to history
    saveKomunikasiToHistory(result, features);
    
    sounds.success();
}

function hasManualKomunikasiInput() {
    const fields = ['meanRedKom', 'meanGreenKom', 'meanBlueKom', 'areaKom', 'brightnessKom', 'perimeterKom'];
    return fields.some(field => document.getElementById(field).value.trim() !== '');
}

function getKomunikasiFeatures() {
    return {
        meanRed: parseInt(document.getElementById('meanRedKom').value) || 0,
        meanGreen: parseInt(document.getElementById('meanGreenKom').value) || 0,
        meanBlue: parseInt(document.getElementById('meanBlueKom').value) || 0,
        area: parseInt(document.getElementById('areaKom').value) || 0,
        brightness: parseInt(document.getElementById('brightnessKom').value) || 0,
        perimeter: parseInt(document.getElementById('perimeterKom').value) || 0
    };
}

function displayKomunikasiResults(result, features) {
    const hasilElement = document.getElementById('hasilPrediksi');
    const detailElement = document.getElementById('detailHasil');
    
    // Main result
    const resultClass = result.prediksi === 'Verbal' ? 'verbal' : 'non-verbal';
    const icon = result.prediksi === 'Verbal' ? '🗣️' : '🤝';
    const color = result.prediksi === 'Verbal' ? '#3498db' : '#e67e22';
    
    hasilElement.innerHTML = `
        <div style="background: linear-gradient(135deg, ${color}, rgba(${color.replace('#', '')}, 0.7)); color: white; padding: 25px; border-radius: 20px; text-align: center; box-shadow: 0 10px 30px rgba(0,0,0,0.2);">
            <h2>${icon} Klasifikasi: ${result.prediksi}</h2>
            <p>Confidence: ${(result.confidence * 100).toFixed(1)}%</p>
            <small>Algorithm: ${result.algorithm}</small>
        </div>
    `;
    
    // Detailed results
    let detailsHtml = `
        <h3><i class="fas fa-chart-bar"></i> Detail Analisis Komunikasi</h3>
        <div class="analysis-grid">
            <div class="analysis-card">
                <h4><i class="fas fa-comments"></i> Skor Klasifikasi</h4>
                <div class="score-bar">
                    <div class="score-item">
                        <span>Verbal:</span>
                        <div class="progress-bar">
                            <div class="progress-fill" style="width: ${(result.scores.verbal / (result.scores.verbal + result.scores.nonVerbal) * 100)}%; background: #3498db;"></div>
                        </div>
                        <span>${result.scores.verbal.toFixed(3)}</span>
                    </div>
                    <div class="score-item">
                        <span>Non-Verbal:</span>
                        <div class="progress-bar">
                            <div class="progress-fill" style="width: ${(result.scores.nonVerbal / (result.scores.verbal + result.scores.nonVerbal) * 100)}%; background: #e67e22;"></div>
                        </div>
                        <span>${result.scores.nonVerbal.toFixed(3)}</span>
                    </div>
                </div>
            </div>
        </div>
    `;
    
    // Add algorithm specific details
    if (result.details) {
        if (result.algorithm === 'PSO') {
            detailsHtml += `
                <div class="analysis-card">
                    <h4><i class="fas fa-cogs"></i> PSO Algorithm Details</h4>
                    <p><strong>Optimal Weights:</strong> [${result.details.optimalWeights.map(w => w.toFixed(3)).join(', ')}]</p>
                    <p><strong>Fitness Score:</strong> ${result.details.fitness.toFixed(4)}</p>
                    <p><strong>Convergence:</strong> ${result.details.convergenceInfo}</p>
                </div>
            `;
        } else if (result.algorithm === 'KNN') {
            detailsHtml += `
                <div class="analysis-card">
                    <h4><i class="fas fa-search"></i> KNN Neighbors (k=${result.details.k})</h4>
                    <div class="neighbors-list">
                        ${result.details.neighbors.slice(0, 3).map(neighbor => `
                            <div class="neighbor-item">
                                <span><strong>${neighbor.nama}</strong> (${neighbor.kategori})</span>
                                <span>Distance: ${neighbor.distance.toFixed(3)}</span>
                            </div>
                        `).join('')}
                    </div>
                </div>
            `;
        }
        
        // Top matches
        if (result.details.topMatches) {
            detailsHtml += `
                <div class="analysis-card">
                    <h4><i class="fas fa-trophy"></i> Top Matches</h4>
                    <div class="matches-list">
                        ${result.details.topMatches.map(match => `
                            <div class="match-item">
                                <span><strong>${match.nama}</strong> (${match.kategori})</span>
                                <span>Similarity: ${(match.similarity * 100).toFixed(1)}%</span>
                            </div>
                        `).join('')}
                    </div>
                </div>
            `;
        }
    }
    
    detailElement.innerHTML = detailsHtml;
    
    // Show confidence bar
    const confidenceBar = document.getElementById('confidenceBar');
    const confidenceFill = document.getElementById('confidenceFill');
    if (confidenceBar && confidenceFill) {
        confidenceBar.style.display = 'block';
        confidenceFill.style.width = (result.confidence * 100) + '%';
        confidenceFill.style.background = color;
    }
}

function saveKomunikasiToHistory(result, features) {
    const historyItem = {
        timestamp: new Date().toLocaleString(),
        type: 'Komunikasi',
        prediction: result.prediksi,
        confidence: result.confidence,
        algorithm: result.algorithm,
        features: features
    };
    
    predictionHistory.unshift(historyItem);
    if (predictionHistory.length > 50) {
        predictionHistory = predictionHistory.slice(0, 50);
    }
    
    localStorage.setItem('plantHistory', JSON.stringify(predictionHistory));
    updateHistoryDisplay();
    
    // Update stats
    totalPredictions++;
    userScore += Math.round(result.confidence * 100);
    localStorage.setItem('totalPredictions', totalPredictions.toString());
    localStorage.setItem('userScore', userScore.toString());
}