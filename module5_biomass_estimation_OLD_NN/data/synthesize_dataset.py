"""
synthesize_dataset.py
---------------------
Generates a mock dataset for training Module 5 (Biomass Estimation)
in experimental / capstone environments where physical destructive 
biomass sampling data is unavailable.

1. Scans a directory for cropped tree images (from Module 3)
2. Generates plausible synthetic masks (tree, canopy, trunk)
3. Extracts the 22 Handcrafted Geometric Features using our module.
4. Uses allometric scaling (DBH estimation from width) to assign a plausible 'biomass_kg' label.
5. Splits the data into train.csv and val.csv
"""

import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from features.handcrafted import GeometricFeatureExtractor

def generate_synthetic_masks(img_shape):
    """
    Creates plausible masks for a cropped tree image.
    In a real pipeline, these come from Module 4 predictions.
    """
    h, w = img_shape[:2]
    
    tree_mask = np.zeros((h, w), dtype=np.uint8)
    canopy_mask = np.zeros((h, w), dtype=np.uint8)
    trunk_mask = np.zeros((h, w), dtype=np.uint8)
    
    # Random canopy (ellipse near the top/center)
    cx, cy = w // 2, h // 3
    rx, ry = int(w * 0.4), int(h * 0.3)
    cv2.ellipse(canopy_mask, (cx, cy), (rx, ry), 0, 0, 360, 1, -1)
    
    # Random trunk (rectangle near the bottom)
    tx, ty = w // 2, int(h * 0.7)
    tw, th = max(10, int(w * 0.1)), int(h * 0.3)
    cv2.rectangle(trunk_mask, (tx - tw//2, ty), (tx + tw//2, ty + th), 2, -1)
    
    # Tree mask is the union
    tree_mask = cv2.bitwise_or(canopy_mask, trunk_mask)
    
    # Dummy polygon (bounding box of the tree mask)
    polygon = [(tx - tw//2, ty), (tx + tw//2, ty), (cx + rx, cy), (cx - rx, cy)]
    
    return tree_mask, canopy_mask, trunk_mask, polygon


def allometric_biomass_equation(trunk_width_px, image_width_px, physical_scale_meters=5.0):
    """
    Estimates a plausible biomass in kg using a mock allometric equation.
    Assume the image width represents 'physical_scale_meters' in real life.
    """
    # Estimate DBH (Diameter at Breast Height) in cm
    trunk_ratio = trunk_width_px / max(1, image_width_px)
    dbh_cm = (trunk_ratio * physical_scale_meters) * 100 
    
    # Jenkins et al. general hardwood equation: Biomass = exp(-2.01 + 2.43 * ln(DBH))
    if dbh_cm < 1.0: dbh_cm = 1.0
    biomass_kg = np.exp(-2.01 + 2.43 * np.log(dbh_cm))
    
    # Add random noise (±20%) to simulate real-world variance
    noise = np.random.uniform(0.8, 1.2)
    return max(1.0, biomass_kg * noise)


def synthesize():
    # Setup paths (Adjust these to point to your actual cropped images on Kaggle / Local)
    # Using a placeholder folder path that you should replace with your dataset path
    raw_images_dir = "data/raw" 
    splits_dir = "data/splits"
    os.makedirs(raw_images_dir, exist_ok=True)
    os.makedirs(splits_dir, exist_ok=True)
    
    # If no images exist, generate some dummy colored images for testing
    if len(os.listdir(raw_images_dir)) == 0:
        print(f"No images found in {raw_images_dir}. Generating 100 dummy images for testing...")
        for i in range(100):
            dummy_img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            cv2.imwrite(os.path.join(raw_images_dir, f"tree_{i}.jpg"), dummy_img)
            
    images = [f for f in os.listdir(raw_images_dir) if f.endswith(('.jpg', '.png'))]
    print(f"Found {len(images)} images. Synthesizing features and passing through allometric equations...")
    
    geo_extractor = GeometricFeatureExtractor()
    data = []
    
    for img_name in tqdm(images):
        img_path = os.path.join(raw_images_dir, img_name)
        img = cv2.imread(img_path)
        h, w = img.shape[:2]
        
        # 1. Gen masks
        t_mask, c_mask, tr_mask, poly = generate_synthetic_masks(img.shape)
        
        # 2. Extract 22 features
        geo_features = geo_extractor.extract(t_mask, c_mask, tr_mask, poly)
        
        # 3. Gen attributes
        species_id = np.random.randint(0, 5) # 5 dummy species
        health_id = np.random.randint(0, 3)  # 0: Unhealthy, 1: Moderate, 2: Healthy
        
        # 4. Synthesize ground truth label
        # Trunk width is feature index 1 (bounding box width in our extractor logic from mask)
        trunk_w = max(10, w * 0.1) # Just a rough approximation based on mask gen
        biomass = allometric_biomass_equation(trunk_w, w)
        
        row = {
            'image_name': img_name,
            'species_id': species_id,
            'health_id': health_id,
            'biomass_kg': round(biomass, 2)
        }
        
        for i in range(22):
            row[f'feat_{i}'] = geo_features[i]
            
        data.append(row)
        
    df = pd.DataFrame(data)
    
    # Split
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    
    train_csv = os.path.join(splits_dir, "train.csv")
    val_csv = os.path.join(splits_dir, "val.csv")
    
    train_df.to_csv(train_csv, index=False)
    val_df.to_csv(val_csv, index=False)
    
    print(f"\nSuccessfully generated synthesized dataset!")
    print(f"Train samples: {len(train_df)} -> {train_csv}")
    print(f"Val samples:   {len(val_df)} -> {val_csv}")

if __name__ == "__main__":
    synthesize()
