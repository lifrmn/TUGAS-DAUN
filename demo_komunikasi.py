#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🎮 DEMO LENGKAP KLASIFIKASI KOMUNIKASI
Demo comprehensive untuk testing KNN & PSO Classification System
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image, ImageDraw, ImageFont
import base64
import io
import json
import requests
import time
import os
from datetime import datetime

class CommunicationDemo:
    def __init__(self):
        self.demo_images = {}
        self.results_history = []
        
    def create_sample_images(self):
        """Membuat gambar sample untuk testing"""
        print("🎨 Creating sample images...")
        
        # Verbal communication samples
        verbal_samples = {
            'book': self.create_book_image(),
            'microphone': self.create_microphone_image(),
            'laptop': self.create_laptop_image(),
            'smartphone': self.create_smartphone_image(),
            'newspaper': self.create_newspaper_image()
        }
        
        # Non-verbal communication samples
        nonverbal_samples = {
            'painting': self.create_painting_image(),
            'statue': self.create_statue_image(),
            'hand_gesture': self.create_hand_gesture_image(),
            'traffic_sign': self.create_traffic_sign_image(),
            'flag': self.create_flag_image()
        }
        
        self.demo_images = {**verbal_samples, **nonverbal_samples}
        
        # Save images to disk
        for name, image in self.demo_images.items():
            cv2.imwrite(f'demo_{name}.jpg', image)
        
        print(f"✅ Created {len(self.demo_images)} sample images")
        return self.demo_images
    
    def create_book_image(self):
        """Create book image (Verbal)"""
        img = np.ones((300, 300, 3), dtype=np.uint8) * 240  # Light background
        
        # Draw book cover
        cv2.rectangle(img, (50, 80), (250, 220), (200, 180, 160), -1)
        cv2.rectangle(img, (50, 80), (250, 220), (100, 80, 60), 3)
        
        # Add book spine
        cv2.rectangle(img, (45, 80), (55, 220), (150, 120, 100), -1)
        
        # Add text lines (simulated)
        for i in range(5):
            y = 120 + i * 15
            cv2.rectangle(img, (70, y), (230, y+8), (50, 50, 50), -1)
        
        return img
    
    def create_microphone_image(self):
        """Create microphone image (Verbal)"""
        img = np.ones((300, 300, 3), dtype=np.uint8) * 50  # Dark background
        
        # Microphone head (circle)
        cv2.circle(img, (150, 100), 40, (80, 80, 80), -1)
        cv2.circle(img, (150, 100), 35, (200, 200, 200), -1)
        
        # Microphone body (rectangle)
        cv2.rectangle(img, (140, 140), (160, 250), (60, 60, 60), -1)
        
        # Base
        cv2.rectangle(img, (120, 250), (180, 270), (40, 40, 40), -1)
        
        # Grid pattern on head
        for i in range(-20, 21, 5):
            cv2.line(img, (150+i, 85), (150+i, 115), (150, 150, 150), 1)
        
        return img
    
    def create_laptop_image(self):
        """Create laptop image (Verbal)"""
        img = np.ones((300, 300, 3), dtype=np.uint8) * 120  # Gray background
        
        # Laptop base
        cv2.rectangle(img, (50, 180), (250, 250), (80, 80, 80), -1)
        cv2.rectangle(img, (55, 185), (245, 245), (40, 40, 40), -1)
        
        # Screen
        cv2.rectangle(img, (70, 80), (230, 180), (20, 20, 20), -1)
        cv2.rectangle(img, (80, 90), (220, 170), (0, 100, 200), -1)  # Blue screen
        
        # Keyboard (simplified)
        for row in range(4):
            for col in range(10):
                x = 70 + col * 17
                y = 195 + row * 12
                cv2.rectangle(img, (x, y), (x+12, y+8), (200, 200, 200), -1)
        
        return img
    
    def create_smartphone_image(self):
        """Create smartphone image (Verbal)"""
        img = np.ones((300, 300, 3), dtype=np.uint8) * 60  # Dark background
        
        # Phone body
        cv2.rectangle(img, (110, 50), (190, 250), (40, 40, 40), -1)
        
        # Screen
        cv2.rectangle(img, (115, 70), (185, 230), (0, 0, 0), -1)
        cv2.rectangle(img, (120, 75), (180, 225), (0, 150, 255), -1)  # Blue screen
        
        # Home button
        cv2.circle(img, (150, 240), 8, (180, 180, 180), -1)
        
        # App icons (simplified)
        for row in range(4):
            for col in range(3):
                x = 130 + col * 20
                y = 90 + row * 25
                cv2.rectangle(img, (x, y), (x+15, y+15), (255, 100, 100), -1)
        
        return img
    
    def create_newspaper_image(self):
        """Create newspaper image (Verbal)"""
        img = np.ones((300, 300, 3), dtype=np.uint8) * 230  # Light background
        
        # Newspaper outline
        cv2.rectangle(img, (40, 40), (260, 260), (200, 200, 200), -1)
        cv2.rectangle(img, (40, 40), (260, 260), (100, 100, 100), 2)
        
        # Header
        cv2.rectangle(img, (50, 50), (250, 80), (50, 50, 50), -1)
        
        # Columns of text
        for col in range(3):
            x_start = 60 + col * 60
            for line in range(15):
                y = 100 + line * 10
                width = np.random.randint(30, 50)
                cv2.rectangle(img, (x_start, y), (x_start + width, y + 6), (80, 80, 80), -1)
        
        return img
    
    def create_painting_image(self):
        """Create painting image (Non-Verbal)"""
        img = np.ones((300, 300, 3), dtype=np.uint8) * 160  # Neutral background
        
        # Frame
        cv2.rectangle(img, (30, 30), (270, 270), (139, 69, 19), 15)  # Brown frame
        
        # Canvas
        cv2.rectangle(img, (50, 50), (250, 250), (240, 240, 220), -1)  # Off-white canvas
        
        # Abstract painting elements
        colors = [(255, 100, 100), (100, 255, 100), (100, 100, 255), (255, 255, 100)]
        
        # Random brush strokes
        for _ in range(10):
            x1, y1 = np.random.randint(60, 240, 2)
            x2, y2 = np.random.randint(60, 240, 2)
            color = colors[np.random.randint(0, len(colors))]
            cv2.line(img, (x1, y1), (x2, y2), color, np.random.randint(3, 8))
        
        # Some shapes
        cv2.circle(img, (120, 120), 30, (255, 100, 100), -1)
        cv2.rectangle(img, (160, 160), (220, 220), (100, 100, 255), -1)
        
        return img
    
    def create_statue_image(self):
        """Create statue image (Non-Verbal)"""
        img = np.ones((300, 300, 3), dtype=np.uint8) * 140  # Gray background
        
        # Pedestal
        cv2.rectangle(img, (80, 220), (220, 280), (100, 100, 100), -1)
        cv2.rectangle(img, (75, 215), (225, 225), (120, 120, 120), -1)
        
        # Statue body
        cv2.rectangle(img, (120, 120), (180, 220), (160, 160, 160), -1)
        
        # Head
        cv2.circle(img, (150, 100), 25, (160, 160, 160), -1)
        
        # Arms
        cv2.rectangle(img, (90, 130), (120, 180), (160, 160, 160), -1)  # Left arm
        cv2.rectangle(img, (180, 130), (210, 180), (160, 160, 160), -1)  # Right arm
        
        # Highlights and shadows
        cv2.rectangle(img, (125, 125), (135, 215), (200, 200, 200), -1)  # Highlight
        cv2.rectangle(img, (165, 125), (175, 215), (120, 120, 120), -1)  # Shadow
        
        return img
    
    def create_hand_gesture_image(self):
        """Create hand gesture image (Non-Verbal)"""
        img = np.ones((300, 300, 3), dtype=np.uint8) * 180  # Light background
        
        # Simplified hand shape
        # Palm
        cv2.ellipse(img, (150, 180), (40, 60), 0, 0, 360, (200, 160, 130), -1)
        
        # Fingers
        finger_positions = [(120, 120), (135, 110), (150, 105), (165, 110), (180, 120)]
        for i, (x, y) in enumerate(finger_positions):
            # Finger
            cv2.rectangle(img, (x-5, y), (x+5, y+50), (200, 160, 130), -1)
            cv2.circle(img, (x, y), 8, (200, 160, 130), -1)  # Fingertip
        
        # Thumb
        cv2.ellipse(img, (110, 160), (15, 25), 45, 0, 360, (200, 160, 130), -1)
        
        # Wrist
        cv2.rectangle(img, (130, 230), (170, 280), (200, 160, 130), -1)
        
        return img
    
    def create_traffic_sign_image(self):
        """Create traffic sign image (Non-Verbal)"""
        img = np.ones((300, 300, 3), dtype=np.uint8) * 100  # Dark background
        
        # Sign post
        cv2.rectangle(img, (145, 200), (155, 280), (80, 80, 80), -1)
        
        # Sign background (red circle)
        cv2.circle(img, (150, 120), 60, (50, 50, 200), -1)  # Red background
        cv2.circle(img, (150, 120), 55, (255, 255, 255), -1)  # White inner
        
        # Stop sign symbol
        cv2.rectangle(img, (120, 110), (180, 130), (50, 50, 200), -1)  # Red bar
        
        # Border
        cv2.circle(img, (150, 120), 60, (200, 200, 200), 3)
        
        return img
    
    def create_flag_image(self):
        """Create flag image (Non-Verbal)"""
        img = np.ones((300, 300, 3), dtype=np.uint8) * 120  # Gray background
        
        # Flagpole
        cv2.rectangle(img, (48, 50), (55, 280), (80, 60, 40), -1)
        
        # Flag
        flag_colors = [(50, 50, 200), (255, 255, 255), (200, 50, 50)]  # Red, white, blue
        
        for i, color in enumerate(flag_colors):
            y_start = 60 + i * 30
            y_end = y_start + 30
            cv2.rectangle(img, (60, y_start), (220, y_end), color, -1)
        
        # Flag border
        cv2.rectangle(img, (60, 60), (220, 150), (100, 100, 100), 2)
        
        # Wind effect (wavy right edge)
        for y in range(60, 150, 5):
            x_offset = int(10 * np.sin((y - 60) * 0.2))
            cv2.line(img, (220, y), (225 + x_offset, y), (100, 100, 100), 2)
        
        return img
    
    def image_to_base64(self, image):
        """Convert OpenCV image to base64 string"""
        _, buffer = cv2.imencode('.jpg', image)
        image_base64 = base64.b64encode(buffer).decode('utf-8')
        return f"data:image/jpeg;base64,{image_base64}"
    
    def test_api_classification(self, api_url="http://localhost:5000"):
        """Test API classification dengan sample images"""
        print(f"\n🔬 Testing API classification at {api_url}")
        
        results = {}
        
        for name, image in self.demo_images.items():
            print(f"   Testing {name}...")
            
            try:
                # Convert image to base64
                image_base64 = self.image_to_base64(image)
                
                # Call API
                response = requests.post(f"{api_url}/api/classify", 
                    json={
                        'image': image_base64,
                        'algorithm': 'both'
                    },
                    timeout=30
                )
                
                if response.status_code == 200:
                    result = response.json()
                    results[name] = result
                    
                    # Display results
                    if result['success']:
                        knn_pred = result['knn_result']['prediction']
                        pso_pred = result['pso_result']['prediction'] 
                        agreement = "✅" if knn_pred == pso_pred else "❌"
                        
                        print(f"      KNN: {knn_pred} ({result['knn_result']['confidence']:.2f})")
                        print(f"      PSO: {pso_pred} ({result['pso_result']['confidence']:.2f})")
                        print(f"      Agreement: {agreement}")
                    else:
                        print(f"      ❌ Failed: {result.get('error', 'Unknown error')}")
                else:
                    print(f"      ❌ HTTP Error: {response.status_code}")
                    
            except requests.exceptions.ConnectionError:
                print(f"      ❌ Connection Error: API not running?")
                break
            except Exception as e:
                print(f"      ❌ Error: {str(e)}")
            
            time.sleep(0.5)  # Small delay between requests
        
        return results
    
    def test_local_classification(self):
        """Test local classification tanpa API"""
        print("\n🧪 Testing local classification...")
        
        try:
            from advanced_komunikasi_extractor import CommunicationFeatureExtractor, PSO_CommunicationClassifier
            from sklearn.neighbors import KNeighborsClassifier
            
            extractor = CommunicationFeatureExtractor()
            
            # Test data
            verbal_expected = ['book', 'microphone', 'laptop', 'smartphone', 'newspaper']
            nonverbal_expected = ['painting', 'statue', 'hand_gesture', 'traffic_sign', 'flag']
            
            correct_knn = 0
            correct_pso = 0
            total_tests = 0
            
            for name, image in self.demo_images.items():
                print(f"   Testing {name}...")
                
                # Save temp image
                temp_path = f'temp_{name}.jpg'
                cv2.imwrite(temp_path, image)
                
                # Extract features
                features = extractor.extract_features_array(temp_path)
                os.remove(temp_path)
                
                if features is None:
                    print(f"      ❌ Feature extraction failed")
                    continue
                
                # Expected category
                expected = 'Verbal' if name in verbal_expected else 'Non-Verbal'
                
                # Test with simplified KNN (using dataset similarity)
                from communication_api import knn_classify, pso_classify
                
                knn_result = knn_classify(features[:11])  # Use first 11 features
                pso_result = pso_classify(features[:11])
                
                knn_correct = knn_result['prediction'] == expected
                pso_correct = pso_result['prediction'] == expected
                
                if knn_correct: correct_knn += 1
                if pso_correct: correct_pso += 1
                total_tests += 1
                
                print(f"      Expected: {expected}")
                print(f"      KNN: {knn_result['prediction']} {'✅' if knn_correct else '❌'}")
                print(f"      PSO: {pso_result['prediction']} {'✅' if pso_correct else '❌'}")
            
            print(f"\n📊 Local Test Results:")
            print(f"   KNN Accuracy: {correct_knn}/{total_tests} ({correct_knn/total_tests*100:.1f}%)")
            print(f"   PSO Accuracy: {correct_pso}/{total_tests} ({correct_pso/total_tests*100:.1f}%)")
            
        except ImportError as e:
            print(f"   ❌ Import Error: {str(e)}")
            print("   Required modules not available for local testing")
        except Exception as e:
            print(f"   ❌ Error: {str(e)}")
    
    def create_visual_report(self, results=None):
        """Create visual report dari testing results"""
        print("\n📊 Creating visual report...")
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 5, figsize=(20, 8))
        fig.suptitle('Communication Classification Demo Results', fontsize=16, fontweight='bold')
        
        # Display sample images
        for i, (name, image) in enumerate(self.demo_images.items()):
            row = i // 5
            col = i % 5
            
            # Convert BGR to RGB for display
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            axes[row, col].imshow(image_rgb)
            axes[row, col].set_title(name.replace('_', ' ').title(), fontsize=10, fontweight='bold')
            axes[row, col].axis('off')
            
            # Add category label
            category = 'Verbal' if name in ['book', 'microphone', 'laptop', 'smartphone', 'newspaper'] else 'Non-Verbal'
            color = 'green' if category == 'Verbal' else 'orange'
            axes[row, col].text(0.5, -0.1, category, transform=axes[row, col].transAxes, 
                              ha='center', fontweight='bold', color=color)
        
        plt.tight_layout()
        plt.savefig('communication_demo_samples.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create accuracy comparison chart if results available
        if results:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Accuracy by algorithm
            knn_accuracies = []
            pso_accuracies = []
            labels = []
            
            for name, result in results.items():
                if result.get('success'):
                    labels.append(name.replace('_', ' ').title())
                    knn_accuracies.append(result['knn_result']['confidence'])
                    pso_accuracies.append(result['pso_result']['confidence'])
            
            x = np.arange(len(labels))
            width = 0.35
            
            ax1.bar(x - width/2, knn_accuracies, width, label='KNN', alpha=0.8, color='skyblue')
            ax1.bar(x + width/2, pso_accuracies, width, label='PSO', alpha=0.8, color='lightcoral')
            
            ax1.set_xlabel('Sample Images')
            ax1.set_ylabel('Confidence Score')
            ax1.set_title('Classification Confidence by Algorithm')
            ax1.set_xticks(x)
            ax1.set_xticklabels(labels, rotation=45, ha='right')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Agreement analysis
            agreements = []
            predictions = []
            
            for name, result in results.items():
                if result.get('success'):
                    knn_pred = result['knn_result']['prediction']
                    pso_pred = result['pso_result']['prediction']
                    agreements.append(1 if knn_pred == pso_pred else 0)
                    predictions.append(knn_pred)
            
            # Pie chart for algorithm agreement
            agreement_counts = [sum(agreements), len(agreements) - sum(agreements)]
            ax2.pie(agreement_counts, labels=['Agreement', 'Disagreement'], 
                   colors=['lightgreen', 'lightpink'], autopct='%1.1f%%', startangle=90)
            ax2.set_title('Algorithm Agreement Rate')
            
            plt.tight_layout()
            plt.savefig('communication_demo_results.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        print("✅ Visual reports saved:")
        print("   - communication_demo_samples.png")
        if results:
            print("   - communication_demo_results.png")
    
    def run_complete_demo(self):
        """Run complete demonstration"""
        print("🎮 COMMUNICATION CLASSIFICATION COMPLETE DEMO")
        print("=" * 60)
        
        # Create sample images
        self.create_sample_images()
        
        # Test local classification
        self.test_local_classification()
        
        # Test API classification
        api_results = self.test_api_classification()
        
        # Create visual report
        self.create_visual_report(api_results)
        
        # Summary
        print("\n🎯 Demo Summary:")
        print(f"   ✅ Created {len(self.demo_images)} sample images")
        print(f"   ✅ Tested local classification")
        print(f"   ✅ Tested API classification")
        print(f"   ✅ Generated visual reports")
        
        print("\n📁 Files created:")
        print("   - demo_*.jpg (sample images)")
        print("   - communication_demo_samples.png")
        if api_results:
            print("   - communication_demo_results.png")
        
        print("\n🚀 Next steps:")
        print("   1. Start API server: python communication_api.py")
        print("   2. Open browser: http://localhost:5000")
        print("   3. Test with demo images or your own images")
        print("   4. Compare KNN vs PSO results")

def main():
    """Main demo function"""
    demo = CommunicationDemo()
    demo.run_complete_demo()

if __name__ == "__main__":
    main()