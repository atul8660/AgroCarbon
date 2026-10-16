"""
model.py
---------
Thin wrapper around ultralytics' YOLO class, kept as its own module (matching
Module 1/2's models/model.py convention) so training/evaluation/inference
code depends on a single, consistent construction path rather than calling
`ultralytics.YOLO(...)` directly in five different places.

YOLOv11 is used as specified in the requirements — Ultralytics' latest
detection architecture, offered in n/s/m/l/x sizes. We default to `yolo11m`
(medium) as the accuracy/speed trade-off point: trees have highly variable
scale (a nearby trunk vs. a distant forest tree) and irregular, non-rigid
silhouettes, so the extra capacity over yolo11n/s pays off, while yolo11l/x
would be unnecessarily slow for what's ultimately a single-class detector
feeding a real-time-ish pipeline.
"""

import logging

from ultralytics import YOLO

logger = logging.getLogger("model")


def build_model(cfg: dict) -> YOLO:
    """Builds a YOLOv11 model for transfer learning, initialized from the
    COCO-pretrained checkpoint named in configs/config.yaml -> model.pretrained_weights.
    """
    model_cfg = cfg["model"]
    weights = model_cfg["pretrained_weights"]
    logger.info(f"Building YOLOv11 model from pretrained weights: {weights}")
    model = YOLO(weights)
    return model


def load_model(checkpoint_path: str) -> YOLO:
    """Loads a fine-tuned checkpoint (.pt) for evaluation/inference/export."""
    logger.info(f"Loading model checkpoint: {checkpoint_path}")
    model = YOLO(checkpoint_path)
    return model
