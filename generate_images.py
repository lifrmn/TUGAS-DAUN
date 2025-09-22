#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🖼️ PLANT IMAGE GENERATOR - PLACEHOLDER IMAGES
=============================================
Generates placeholder images for plant classification web interface
"""

import os
from PIL import Image, ImageDraw, ImageFont
import random

def create_plant_placeholder(name, plant_type, size=(300, 300)):
    """
    Create a placeholder image for a plant
    """
    # Create base image
    img = Image.new('RGB', size, color='white')
    draw = ImageDraw.Draw(img)
    
    # Color scheme based on plant type
    if plant_type.lower() == 'herbal':
        base_color = '#27ae60'  # Green for herbal
        accent_color = '#2ecc71'
    else:
        base_color = '#e67e22'  # Orange for non-herbal
        accent_color = '#f39c12'
    
    # Draw background gradient effect
    for y in range(size[1]):
        ratio = y / size[1]
        r1, g1, b1 = tuple(int(base_color[i:i+2], 16) for i in (1, 3, 5))
        r2, g2, b2 = tuple(int(accent_color[i:i+2], 16) for i in (1, 3, 5))
        
        r = int(r1 + (r2 - r1) * ratio)
        g = int(g1 + (g2 - g1) * ratio)
        b = int(b1 + (b2 - b1) * ratio)
        
        draw.line([(0, y), (size[0], y)], fill=(r, g, b))
    
    # Draw plant silhouette
    center_x, center_y = size[0] // 2, size[1] // 2
    
    # Stem
    stem_width = 8
    stem_height = size[1] // 3
    draw.rectangle([
        center_x - stem_width // 2, 
        center_y + stem_height // 2,
        center_x + stem_width // 2, 
        size[1] - 20
    ], fill='#2d5a27')
    
    # Leaves
    leaf_size = 40
    for i in range(3):
        angle = i * 60 + random.randint(-15, 15)
        leaf_x = center_x + random.randint(-30, 30)
        leaf_y = center_y - i * 20 + random.randint(-10, 10)
        
        # Draw leaf as ellipse
        draw.ellipse([
            leaf_x - leaf_size, leaf_y - leaf_size // 2,
            leaf_x + leaf_size, leaf_y + leaf_size // 2
        ], fill='#27ae60', outline='#1e8449', width=2)
    
    # Add decorative elements based on plant type
    if plant_type.lower() == 'herbal':
        # Add small flowers for herbal plants
        for _ in range(3):
            flower_x = center_x + random.randint(-50, 50)
            flower_y = center_y + random.randint(-50, 30)
            draw.ellipse([
                flower_x - 5, flower_y - 5,
                flower_x + 5, flower_y + 5
            ], fill='#f1c40f')
    else:
        # Add fruits/vegetables for non-herbal
        fruit_x = center_x + random.randint(-20, 20)
        fruit_y = center_y + random.randint(-20, 20)
        draw.ellipse([
            fruit_x - 15, fruit_y - 15,
            fruit_x + 15, fruit_y + 15
        ], fill='#e74c3c', outline='#c0392b', width=2)
    
    # Add plant name
    try:
        # Try to use a nice font
        font = ImageFont.truetype("arial.ttf", 24)
    except:
        # Fallback to default font
        font = ImageFont.load_default()
    
    # Text background
    text_bbox = draw.textbbox((0, 0), name, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    
    text_x = (size[0] - text_width) // 2
    text_y = size[1] - 50
    
    # Semi-transparent background for text
    draw.rectangle([
        text_x - 10, text_y - 5,
        text_x + text_width + 10, text_y + text_height + 5
    ], fill=(255, 255, 255, 200))
    
    # Draw text
    draw.text((text_x, text_y), name, fill='#2c3e50', font=font)
    
    # Add type badge
    badge_text = f"🌿 {plant_type}"
    badge_x = 10
    badge_y = 10
    
    draw.rectangle([
        badge_x, badge_y,
        badge_x + 120, badge_y + 30
    ], fill=(255, 255, 255, 180), outline=base_color, width=2)
    
    draw.text((badge_x + 10, badge_y + 8), badge_text, fill=base_color, font=font)
    
    return img

def generate_plant_images():
    """
    Generate placeholder images for all plants in dataset
    """
    # Plant data from dataset
    plants = [
        ("Jahe", "Herbal"),
        ("Kunyit", "Herbal"),
        ("Sirih", "Herbal"),
        ("Lidah Buaya", "Herbal"),
        ("Pandan", "Herbal"),
        ("Tomat", "Non-Herbal"),
        ("Bayam", "Non-Herbal"),
        ("Cabai", "Non-Herbal"),
        ("Terong", "Non-Herbal"),
        ("Wortel", "Non-Herbal")
    ]
    
    # Create images directory
    images_dir = os.path.join('web', 'images')
    os.makedirs(images_dir, exist_ok=True)
    
    print("🖼️  Generating plant placeholder images...")
    
    for plant_name, plant_type in plants:
        print(f"   Creating image for {plant_name} ({plant_type})...")
        
        # Generate image
        img = create_plant_placeholder(plant_name, plant_type)
        
        # Save image
        filename = f"{plant_name.lower().replace(' ', '_')}.png"
        filepath = os.path.join(images_dir, filename)
        img.save(filepath, 'PNG', quality=95)
        
        print(f"   ✅ Saved: {filepath}")
    
    # Create sample prediction images
    print("\n🔍 Creating sample prediction images...")
    
    sample_plants = [
        ("Unknown Plant 1", "Unknown"),
        ("Unknown Plant 2", "Unknown"),
        ("Test Sample", "Unknown")
    ]
    
    for plant_name, plant_type in sample_plants:
        img = create_plant_placeholder(plant_name, plant_type)
        filename = f"sample_{plant_name.lower().replace(' ', '_')}.png"
        filepath = os.path.join(images_dir, filename)
        img.save(filepath, 'PNG', quality=95)
        print(f"   ✅ Saved: {filepath}")
    
    print(f"\n🎉 Successfully generated {len(plants) + len(sample_plants)} placeholder images!")
    print(f"📁 Images saved in: {images_dir}")

def create_logo_icon():
    """
    Create a logo/icon for the application
    """
    size = (512, 512)
    img = Image.new('RGB', size, color='white')
    draw = ImageDraw.Draw(img)
    
    # Background circle
    center = size[0] // 2
    radius = center - 50
    
    # Gradient background
    for r in range(radius, 0, -1):
        ratio = (radius - r) / radius
        color_val = int(102 + (118 - 102) * ratio)  # From #667eea to #764ba2
        draw.ellipse([
            center - r, center - r,
            center + r, center + r
        ], fill=(color_val, 126, 234))
    
    # Plant silhouette in center
    # Stem
    draw.rectangle([
        center - 8, center + 50,
        center + 8, center + 150
    ], fill='white')
    
    # Leaves
    leaf_points = [
        (center - 60, center - 20),
        (center - 20, center - 60),
        (center, center - 20),
        (center - 20, center + 20)
    ]
    draw.polygon(leaf_points, fill='white')
    
    leaf_points = [
        (center + 60, center - 20),
        (center + 20, center - 60),
        (center, center - 20),
        (center + 20, center + 20)
    ]
    draw.polygon(leaf_points, fill='white')
    
    # Center flower
    draw.ellipse([
        center - 20, center - 20,
        center + 20, center + 20
    ], fill='white')
    
    # Save logo
    logo_path = os.path.join('web', 'logo.png')
    img.save(logo_path, 'PNG', quality=95)
    print(f"🎨 Logo created: {logo_path}")

if __name__ == "__main__":
    print("🌿 PLANT CLASSIFIER - IMAGE GENERATOR")
    print("=" * 40)
    
    # Generate plant images
    generate_plant_images()
    
    # Create logo
    create_logo_icon()
    
    print("\n✅ All images generated successfully!")
    print("🚀 Your Plant Classifier is now ready with visual assets!")