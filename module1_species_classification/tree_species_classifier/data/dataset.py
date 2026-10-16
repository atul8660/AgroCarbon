"""
dataset.py
-----------
PyTorch Dataset for the merged tree-species dataset, plus Albumentations
train/val/test transform pipelines.

Augmentation rationale (field close-up photos of leaves/bark):
- Rotation/flip: leaves are photographed at arbitrary angles, no canonical
  "up" direction.
- Color jitter / HSV shift / brightness: strong outdoor lighting variance.
- CoarseDropout: simulates partial occlusion by other leaves/branches/hand.
- GaussianBlur / MotionBlur: simulates camera shake / phone autofocus misses.
- RandomResizedCrop: simulates variable camera distance & framing.
- Normalize with ImageNet stats: required for ImageNet-pretrained EfficientNet.
"""

from pathlib import Path

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
        A.RandomResizedCrop(size=(image_size, image_size), scale=(0.7, 1.0), ratio=(0.85, 1.15)),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.Rotate(limit=180, border_mode=cv2.BORDER_REFLECT_101, p=0.7),
        A.OneOf([
            A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05, p=1.0),
            A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=25, val_shift_limit=15, p=1.0),
        ], p=0.7),
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 5), p=1.0),
            A.MotionBlur(blur_limit=5, p=1.0),
        ], p=0.2),
        A.CoarseDropout(num_holes_range=(1, 4), hole_height_range=(0.05, 0.15),
                         hole_width_range=(0.05, 0.15), p=0.3),
        A.GaussNoise(std_range=(0.02, 0.08), p=0.2),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def get_eval_transforms(image_size: int) -> A.Compose:
    return A.Compose([
        A.Resize(height=image_size, width=image_size),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


class TreeSpeciesDataset(Dataset):
    """Reads a CSV manifest with columns [filepath, species, label]."""

    def __init__(self, csv_path: str, transforms: A.Compose = None):
        self.df = pd.read_csv(csv_path)
        self.transforms = transforms

        if not {"filepath", "label"}.issubset(self.df.columns):
            raise ValueError(f"{csv_path} must contain 'filepath' and 'label' columns")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img_path = row["filepath"]

        image = cv2.imread(img_path)
        if image is None:
            # Fallback for unreadable files encountered at train time
            image = np.zeros((300, 300, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transforms is not None:
            augmented = self.transforms(image=image)
            image = augmented["image"]
        else:
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0

        label = int(row["label"])
        return image, label

    def get_class_counts(self) -> np.ndarray:
        """Returns array of sample counts indexed by label, for class weighting."""
        n_classes = self.df["label"].nunique()
        counts = np.zeros(n_classes, dtype=np.int64)
        for label, cnt in self.df["label"].value_counts().items():
            counts[label] = cnt
        return counts


def load_label_map(splits_dir: str) -> dict:
    """Returns {label_int: species_name}."""
    df = pd.read_csv(Path(splits_dir) / "label_map.csv")
    return dict(zip(df["label"], df["species"]))
