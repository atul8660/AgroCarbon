"""
dataset.py
-----------
PyTorch Dataset for the merged tree-health dataset, plus Albumentations
train/val/test transform pipelines.

Augmentation set (as specified for Module 2), with rationale:
- RandomRotation: leaf/lesion orientation is arbitrary in field photos.
- HorizontalFlip / VerticalFlip: no canonical leaf orientation.
- RandomResizedCrop (random crop equivalent): varies framing/distance,
  forces the model to rely on lesion texture rather than full-leaf shape.
- ColorJitter: brightness/contrast/saturation/hue variance from outdoor light.
- RandomBrightnessContrast: explicit separate brightness/contrast control,
  important because disease symptoms (chlorosis, necrosis) are color/contrast
  dependent — over-aggressive joint jitter can wash out subtle discoloration,
  so this is tuned more conservatively than in Module 1.
- GaussNoise: sensor noise robustness (phone cameras in low light).
- Blur (Gaussian/Motion): camera shake / autofocus miss robustness.
- Normalize with ImageNet stats: required for ImageNet-pretrained EfficientNet.
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
        A.VerticalFlip(p=0.3),
        A.Rotate(limit=45, border_mode=cv2.BORDER_REFLECT_101, p=0.6),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.03, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.5),
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 5), p=1.0),
            A.MotionBlur(blur_limit=5, p=1.0),
        ], p=0.15),
        A.GaussNoise(std_range=(0.02, 0.06), p=0.25),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def get_eval_transforms(image_size: int) -> A.Compose:
    return A.Compose([
        A.Resize(height=image_size, width=image_size),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


class TreeHealthDataset(Dataset):
    """Reads a CSV manifest with columns [filepath, health, label]."""

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
        n_classes = self.df["label"].nunique()
        counts = np.zeros(n_classes, dtype=np.int64)
        for label, cnt in self.df["label"].value_counts().items():
            counts[label] = cnt
        return counts


def load_label_map(splits_dir: str) -> dict:
    """Returns {label_int: health_class_name}."""
    df = pd.read_csv(f"{splits_dir}/label_map.csv")
    return dict(zip(df["label"], df["health"]))
