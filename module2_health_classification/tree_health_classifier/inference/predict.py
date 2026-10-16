#!/usr/bin/env python3
"""
predict.py
-----------
Single-image inference: loads a trained checkpoint and returns the predicted
tree health class plus a confidence score for a given close-up leaf/bark photo.

Predictions below `inference.confidence_threshold` (configs/config.yaml) are
returned as "Unknown" rather than a low-confidence guess, since Unknown was
never trained as a class (see README section 5) — this is a serving-time
safety gate, not a model output.

This is the interface the larger carbon-estimation pipeline calls into
(a "Diseased"/"Pest_Affected" result can trigger a biomass-estimate discount
downstream).

Usage:
    python inference/predict.py --checkpoint outputs/checkpoints/best.pt \
        --image /path/to/leaf.jpg

    python inference/predict.py --checkpoint outputs/checkpoints/best.pt \
        --image /path/to/leaf.jpg --top_k 5
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import cv2
import torch
import yaml

sys.path.append(str(Path(__file__).resolve().parents[1]))

from data.dataset import get_eval_transforms
from models.model import build_model
from training.utils import get_device

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
)
logger = logging.getLogger("predict")


class TreeHealthPredictor:
    """Reusable predictor object — instantiate once, call `.predict()` many times.
    This is the class the downstream carbon-pipeline module should import.
    """

    def __init__(self, checkpoint_path: str, device: str = None):
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        self.cfg = checkpoint["config"]
        self.label_map = checkpoint["label_map"]  # {label_int: health_class_name}
        self.num_classes = checkpoint["num_classes"]
        self.image_size = self.cfg["data"]["image_size"]
        self.confidence_threshold = self.cfg["inference"]["confidence_threshold"]

        self.device = get_device(device or self.cfg["project"]["device"])
        self.model = build_model(self.cfg, self.num_classes).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        self.transforms = get_eval_transforms(self.image_size)
        logger.info(
            f"Loaded model ({self.cfg['model']['backbone']}) with "
            f"{self.num_classes} health classes on {self.device} "
            f"(confidence_threshold={self.confidence_threshold})"
        )

    def _preprocess(self, image_path: str) -> torch.Tensor:
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        augmented = self.transforms(image=image)
        tensor = augmented["image"].unsqueeze(0)  # add batch dim
        return tensor

    @torch.no_grad()
    def predict(self, image_path: str, top_k: int = 1) -> dict:
        """Returns predicted health class + confidence, gated by confidence_threshold,
        plus optional top-k alternatives for debugging/UX purposes.
        """
        tensor = self._preprocess(image_path).to(self.device)
        logits = self.model(tensor)
        probs = torch.softmax(logits, dim=1).squeeze(0)  # [num_classes]

        top_k = min(top_k, self.num_classes)
        top_probs, top_indices = torch.topk(probs, k=top_k)

        alternatives = [
            {"health": self.label_map[idx.item()], "confidence": round(prob.item(), 4)}
            for prob, idx in zip(top_probs, top_indices)
        ]

        best_health = alternatives[0]["health"]
        best_confidence = alternatives[0]["confidence"]

        if best_confidence < self.confidence_threshold:
            final_health = "Unknown"
        else:
            final_health = best_health

        result = {
            "image": image_path,
            "health": final_health,
            "confidence": best_confidence,
            "top_k": alternatives,
        }
        return result


def predict_api(checkpoint_path: str, image_path: str, device: str = None) -> dict:
    """Lightweight functional entrypoint matching the Module 2 contract exactly:

        {"health": "Healthy", "confidence": 0.97}

    Intended for direct import by the downstream carbon-estimation pipeline.
    For repeated calls, prefer instantiating TreeHealthPredictor directly to
    avoid reloading the checkpoint on every call.
    """
    predictor = TreeHealthPredictor(checkpoint_path, device=device)
    result = predictor.predict(image_path, top_k=1)
    return {"health": result["health"], "confidence": result["confidence"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--top_k", type=int, default=1)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    predictor = TreeHealthPredictor(args.checkpoint, device=args.device)
    result = predictor.predict(args.image, top_k=args.top_k)

    api_response = {"health": result["health"], "confidence": result["confidence"]}
    print(json.dumps(api_response, indent=2))
    logger.info(
        f"Predicted health: {result['health']} (confidence: {result['confidence']:.2%})"
    )


if __name__ == "__main__":
    main()
