import os
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

class BiomassDataset(Dataset):
    """
    Loads samples mappings from NEON/ForestGEO synthesized datasets:
    - Cropped Tree Image
    - Precomputed Handcrafted Features (22 dims)
    - Species Class ID
    - Health Class ID
    - Label: AGB (kg)
    """
    def __init__(self, csv_file: str, img_dir: str, mode: str = "train", image_size: int = 224):
        self.df = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.mode = mode
        
        # Albumentations pipelines
        if mode == "train":
            self.transform = A.Compose([
                A.Resize(image_size, image_size),
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.2),
                A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.5),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2()
            ])
        else:
            self.transform = A.Compose([
                A.Resize(image_size, image_size),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2()
            ])
            
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Image
        img_path = os.path.join(self.img_dir, row['image_name'])
        image = cv2.imread(img_path)
        if image is None:
            # Fallback for missing images in synthesized mapping
            image = np.zeros((224, 224, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
        augmented = self.transform(image=image)
        image_tensor = augmented['image']
        
        # Handcrafted geometric/mask features (precomputed during data prep)
        # Expected column format: feat_0, feat_1, ..., feat_21
        feat_cols = [f'feat_{i}' for i in range(22)]
        if all(col in self.df.columns for col in feat_cols):
            handcrafted = row[feat_cols].values.astype(np.float32)
        else:
            handcrafted = np.zeros(22, dtype=np.float32)
            
        # Categorical
        species = int(row['species_id'])
        health = int(row['health_id'])
        
        # Label
        agb = np.float32(row['biomass_kg'])
        
        return {
            'image': image_tensor,
            'handcrafted': torch.tensor(handcrafted),
            'species': torch.tensor(species, dtype=torch.long),
            'health': torch.tensor(health, dtype=torch.long),
            'label': torch.tensor([agb], dtype=torch.float32)
        }
