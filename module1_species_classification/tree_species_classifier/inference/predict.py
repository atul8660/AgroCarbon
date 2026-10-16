#!/usr/bin/env python3
"""
predict.py
-----------
Single-image inference for the 13-species EfficientNet-B3 model.

IMPORTANT: The checkpoint was trained using torchvision.models.efficientnet_b3
(NOT timm). Using the wrong library causes silent key mismatches and wrong
predictions. This file uses torchvision exclusively.

Usage:
    python inference/predict.py --checkpoint outputs_pilot/checkpoints/best_13_species_model.pt \
        --image /path/to/leaf.jpg
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
)
logger = logging.getLogger("predict")

# The 13 species from the trained model (alphabetical order, matches training label_map)
CLASS_NAMES = [
    'AlstoniaScholaris', 'Arjun', 'Basil', 'Chinar', 'CurryLeaf',
    'Eucalyptus', 'Guava', 'Jamun', 'Jatropha', 'Lemon',
    'Mango', 'Pomegranate', 'Pongamia'
]
NUM_CLASSES = 13
IMAGE_SIZE = 224


def _build_torchvision_model(num_classes: int) -> nn.Module:
    """Build EfficientNet-B3 using torchvision (matches training notebook exactly)."""
    model = models.efficientnet_b3(weights=None)
    num_ftrs = model.classifier[1].in_features  # 1536
    model.classifier[1] = nn.Linear(num_ftrs, num_classes)
    return model


def _get_transforms():
    """ImageNet-normalised eval transforms at 224x224 (matches training)."""
    return T.Compose([
        T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


class TreeSpeciesPredictor:
    """Reusable predictor — instantiate once, call .predict() many times."""

    def __init__(self, checkpoint_path: str, device: str = None):
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.num_classes = NUM_CLASSES
        self.class_names = CLASS_NAMES
        self.label_map = {i: name for i, name in enumerate(CLASS_NAMES)}
        self.transforms = _get_transforms()

        # Build torchvision EfficientNet-B3 (NOT timm)
        self.model = _build_torchvision_model(self.num_classes)

        # Load checkpoint — MUST be strict=True to catch mismatches
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint  # raw state_dict

        self.model.load_state_dict(state_dict, strict=True)
        self.model.to(self.device)
        self.model.eval()

        logger.info(
            f"Loaded torchvision EfficientNet-B3 with {self.num_classes} classes "
            f"on {self.device} (strict=True, all keys matched)"
        )

    def _preprocess_path(self, image_path: str) -> torch.Tensor:
        """Load and preprocess an image from file path."""
        img = Image.open(image_path).convert("RGB")
        return self.transforms(img).unsqueeze(0)

    def _preprocess_array(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess a numpy array (BGR from cv2 or RGB)."""
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        elif image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(image)
        return self.transforms(img).unsqueeze(0)

    @torch.no_grad()
    def predict(self, image_path: str, top_k: int = 1) -> dict:
        """Predict species from an image file path."""
        tensor = self._preprocess_path(image_path).to(self.device)
        return self._run_inference(tensor, top_k, image_path=image_path)

    @torch.no_grad()
    def predict_from_array(self, image: np.ndarray, top_k: int = 1) -> dict:
        """Predict species from a numpy image array (used by backend pipeline)."""
        tensor = self._preprocess_array(image).to(self.device)
        return self._run_inference(tensor, top_k)

    def _run_inference(self, tensor: torch.Tensor, top_k: int = 1,
                       image_path: str = None) -> dict:
        logits = self.model(tensor)
        probs = torch.softmax(logits, dim=1).squeeze(0)

        top_k = min(top_k, self.num_classes)
        top_probs, top_indices = torch.topk(probs, k=top_k)

        predictions = [
            {"species": self.label_map[idx.item()], "confidence": round(prob.item(), 4)}
            for prob, idx in zip(top_probs, top_indices)
        ]

        result = {
            "predicted_species": predictions[0]["species"],
            "confidence": predictions[0]["confidence"],
            "top_k": predictions,
        }
        if image_path:
            result["image"] = image_path
        return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str,
                        default="outputs_pilot/checkpoints/best_13_species_model.pt")
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--top_k", type=int, default=1)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    predictor = TreeSpeciesPredictor(args.checkpoint, device=args.device)
    result = predictor.predict(args.image, top_k=args.top_k)

    print(json.dumps(result, indent=2))
    logger.info(
        f"Predicted species: {result['predicted_species']} "
        f"(confidence: {result['confidence']:.2%})"
    )


if __name__ == "__main__":
    main()
