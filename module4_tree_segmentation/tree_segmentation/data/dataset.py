"""
dataset.py
-----------
PyTorch Dataset for the merged tree-segmentation dataset, plus Albumentations
train/val/test transform pipelines applied jointly to image + mask.

Returns (image_tensor, mask_tensor, has_part_supervision) — the third value
lets training/engine.py collapse the 3-class loss to binary tree-vs-background
for samples that only have whole-tree (not trunk) ground truth, so the model
is never taught a false canopy/trunk distinction from unlabeled sources (see
data/prepare_dataset.py docstring for the full rationale).
"""

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def get_train_transforms(image_size: int) -> A.Compose:
    return A.Compose([
        A.RandomResizedCrop(size=(image_size, image_size), scale=(0.75, 1.0), ratio=(0.9, 1.1)),
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=15, border_mode=cv2.BORDER_REFLECT_101, p=0.5),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.4),
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 5), p=1.0),
            A.MotionBlur(blur_limit=5, p=1.0),
        ], p=0.15),
        A.GaussNoise(std_range=(0.02, 0.05), p=0.2),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def get_eval_transforms(image_size: int) -> A.Compose:
    return A.Compose([
        A.Resize(height=image_size, width=image_size),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


class TreeSegmentationDataset(Dataset):
    """Reads a CSV manifest with columns [image, mask, source, has_part_supervision]."""

    def __init__(self, csv_path: str, transforms: A.Compose = None):
        self.df = pd.read_csv(csv_path)
        self.df["has_part_supervision"] = self.df["has_part_supervision"].astype(bool)
        self.transforms = transforms

        required_cols = {"image", "mask", "has_part_supervision"}
        if not required_cols.issubset(self.df.columns):
            raise ValueError(f"{csv_path} must contain columns: {required_cols}")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]

        image = cv2.imread(row["image"])
        if image is None:
            image = np.zeros((640, 640, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(row["mask"], cv2.IMREAD_GRAYSCALE)
        if mask is None:
            mask = np.zeros(image.shape[:2], dtype=np.uint8)

        if self.transforms is not None:
            augmented = self.transforms(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]
        else:
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
            mask = torch.from_numpy(mask)

        mask = mask.long()
        has_part_supervision = bool(row["has_part_supervision"])

        return image, mask, has_part_supervision
