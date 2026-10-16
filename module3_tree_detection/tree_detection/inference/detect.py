#!/usr/bin/env python3
"""
detect.py
----------
Reusable TreeDetector class: the single entrypoint the rest of the pipeline
(and Module 4) should import. Detects all tree instances in a whole-tree
scene image, applies the documented primary-tree selection rule (README
section 6) when multiple trees are found, crops the chosen tree with padding
at high resolution, and writes a bbox-derived mask placeholder for Module 4.

Methods:
    load_model()           -- loads the YOLOv11 checkpoint
    detect_tree()           -- runs YOLO inference, returns raw candidate boxes
    select_primary_tree()   -- applies the documented selection rule
    crop_tree()             -- pads + crops + upsamples-if-needed + mask placeholder
    predict_api()           -- single call: detect -> select -> crop -> JSON contract

Usage:
    python inference/detect.py --checkpoint outputs/checkpoints/best.pt \
        --image /path/to/scene.jpg
"""

import argparse
import json
import logging
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

sys.path.append(str(Path(__file__).resolve().parents[1]))

from models.model import load_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
)
logger = logging.getLogger("detect")


class TreeDetector:
    """Whole-tree detector + primary-tree selector + Module-4 crop producer."""

    def __init__(self, checkpoint_path: str, config_path: str = "configs/config.yaml"):
        self.cfg = self._load_config(config_path)
        self.checkpoint_path = checkpoint_path
        self.model = None
        self.load_model()

    @staticmethod
    def _load_config(config_path: str) -> dict:
        with open(config_path, "r") as f:
            return yaml.safe_load(f)

    # -----------------------------------------------------------------
    def load_model(self):
        """Loads the YOLOv11 checkpoint into self.model."""
        self.model = load_model(self.checkpoint_path)
        logger.info(f"TreeDetector ready (checkpoint={self.checkpoint_path})")
        return self.model

    # -----------------------------------------------------------------
    def detect_tree(self, image_path: str) -> tuple:
        """Runs YOLO inference and returns all raw candidate detections above
        inference.min_confidence, as a list of dicts with bbox/confidence,
        plus the (width, height) of the source image. Does NOT yet decide
        which one is "the" tree — see select_primary_tree.
        """
        infer_cfg = self.cfg["inference"]
        results = self.model.predict(
            source=image_path,
            imgsz=self.cfg["model"]["input_size"],
            conf=infer_cfg["min_confidence"],
            verbose=False,
        )
        result = results[0]

        candidates = []
        for box in result.boxes:
            xyxy = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            candidates.append({
                "bbox": [round(v, 2) for v in xyxy],
                "confidence": round(conf, 4),
            })

        img_h, img_w = result.orig_shape
        logger.info(f"Detected {len(candidates)} tree candidate(s) in {image_path} "
                    f"({img_w}x{img_h})")
        return candidates, (img_w, img_h)

    # -----------------------------------------------------------------
    def select_primary_tree(self, candidates: list, image_size: tuple) -> dict:
        """Applies the documented primary-tree selection rule (README section 6):

        - 0 candidates  -> None (caller returns a structured NO_TREE_DETECTED error)
        - 1 candidate   -> used directly
        - >1 candidates -> composite score:
              score = area_weight * normalized_bbox_area
                    + center_weight * (1 - normalized_distance_from_image_center)

        Returns the chosen candidate dict, augmented with 'selection_rule' and,
        when multiple candidates existed, 'all_candidate_scores' for auditability.
        """
        if not candidates:
            return None

        if len(candidates) == 1:
            chosen = dict(candidates[0])
            chosen["selection_rule"] = "single_detection"
            return chosen

        img_w, img_h = image_size
        img_area = img_w * img_h
        img_cx, img_cy = img_w / 2.0, img_h / 2.0
        max_dist = math.hypot(img_cx, img_cy)  # distance from center to a corner (normalizer)

        sel_cfg = self.cfg["selection"]
        area_weight = sel_cfg["area_weight"]
        center_weight = sel_cfg["center_weight"]

        scored = []
        for cand in candidates:
            x1, y1, x2, y2 = cand["bbox"]
            box_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
            normalized_area = box_area / img_area if img_area > 0 else 0.0

            box_cx, box_cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            dist_from_center = math.hypot(box_cx - img_cx, box_cy - img_cy)
            normalized_dist = dist_from_center / max_dist if max_dist > 0 else 0.0

            score = area_weight * normalized_area + center_weight * (1 - normalized_dist)
            scored.append({
                **cand,
                "normalized_area": round(normalized_area, 4),
                "normalized_distance_from_center": round(normalized_dist, 4),
                "primary_score": round(score, 4),
            })

        scored.sort(key=lambda c: c["primary_score"], reverse=True)
        chosen = dict(scored[0])
        chosen["selection_rule"] = "composite_score_area_and_centrality"
        chosen["all_candidate_scores"] = scored
        return chosen

    # -----------------------------------------------------------------
    def crop_tree(self, image_path: str, bbox: list, out_dir: str, out_stem: str) -> dict:
        """Pads the bbox, crops, upsamples if below min_crop_resolution, and
        writes an elliptical bbox-derived mask placeholder for Module 4.

        VERY IMPORTANT (per spec): pads generously so branches/canopy edges
        are never clipped, and never crops the tree tighter than its full
        visible extent.
        """
        infer_cfg = self.cfg["inference"]
        padding_ratio = infer_cfg["crop_padding_ratio"]
        min_resolution = infer_cfg["min_crop_resolution"]

        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")
        img_h, img_w = image.shape[:2]

        x1, y1, x2, y2 = bbox
        box_w, box_h = x2 - x1, y2 - y1
        pad_x, pad_y = box_w * padding_ratio, box_h * padding_ratio

        crop_x1 = max(0, int(x1 - pad_x))
        crop_y1 = max(0, int(y1 - pad_y))
        crop_x2 = min(img_w, int(x2 + pad_x))
        crop_y2 = min(img_h, int(y2 + pad_y))

        crop = image[crop_y1:crop_y2, crop_x1:crop_x2]
        crop_h, crop_w = crop.shape[:2]

        # Guarantee sufficient resolution for Module 4's segmentation input.
        if max(crop_w, crop_h) < min_resolution and crop_w > 0 and crop_h > 0:
            scale = min_resolution / max(crop_w, crop_h)
            crop = cv2.resize(
                crop, (int(crop_w * scale), int(crop_h * scale)),
                interpolation=cv2.INTER_LANCZOS4,
            )
            crop_h, crop_w = crop.shape[:2]

        out_dir_path = Path(out_dir)
        out_dir_path.mkdir(parents=True, exist_ok=True)
        crop_path = out_dir_path / f"{out_stem}.jpg"
        cv2.imwrite(str(crop_path), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])

        # Bbox-derived elliptical mask placeholder — explicitly NOT a real
        # segmentation. Module 4 owns producing the authoritative mask; this
        # exists only as an overwritable starting point / sanity-check input.
        mask = np.zeros((crop_h, crop_w), dtype=np.uint8)
        center = (crop_w // 2, crop_h // 2)
        axes = (int(crop_w * 0.42), int(crop_h * 0.42))
        cv2.ellipse(mask, center, axes, 0, 0, 360, color=255, thickness=-1)
        mask_path = out_dir_path / f"{out_stem}_mask_placeholder.png"
        cv2.imwrite(str(mask_path), mask)

        return {
            "crop_path": str(crop_path),
            "mask_placeholder_path": str(mask_path),
            "crop_bbox_in_original": [crop_x1, crop_y1, crop_x2, crop_y2],
            "crop_resolution": [crop_w, crop_h],
            "padding_ratio": padding_ratio,
        }

    # -----------------------------------------------------------------
    def predict_api(self, image_path: str, out_dir: str = "outputs/crops",
                     out_stem: str = None) -> dict:
        """Single-call entrypoint: detect -> select primary -> crop -> JSON contract.

        Returns exactly:
            {"bbox": [...], "confidence": ..., "crop_path": "...", ...}
        on success, or a structured error dict when no tree is found.
        """
        out_stem = out_stem or Path(image_path).stem

        candidates, image_size = self.detect_tree(image_path)
        primary = self.select_primary_tree(candidates, image_size)

        if primary is None:
            logger.warning(f"NO_TREE_DETECTED in {image_path}")
            return {
                "error": "NO_TREE_DETECTED",
                "bbox": None,
                "confidence": 0.0,
                "cropped_tree": None,
                "module4_input": None,
                "mask_placeholder": None,
                "metadata": {
                    "image": image_path,
                    "image_size": list(image_size),
                    "num_candidates": 0,
                },
            }

        crop_info = self.crop_tree(image_path, primary["bbox"], out_dir, out_stem)

        result = {
            "bbox": primary["bbox"],
            "confidence": primary["confidence"],
            "cropped_tree": crop_info["crop_path"],
            "module4_input": crop_info["crop_path"],
            "mask_placeholder": crop_info["mask_placeholder_path"],
            "selection_rule": primary["selection_rule"],
            "metadata": {
                "image": image_path,
                "image_size": list(image_size),
                "num_candidates": len(candidates),
                "crop_resolution": crop_info["crop_resolution"],
                "crop_bbox_in_original": crop_info["crop_bbox_in_original"],
                "padding_ratio": crop_info["padding_ratio"],
                "candidates": primary.get("all_candidate_scores", candidates),
            },
        }
        return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="outputs/crops")
    args = parser.parse_args()

    detector = TreeDetector(args.checkpoint, config_path=args.config)
    result = detector.predict_api(args.image, out_dir=args.out_dir)

    print(json.dumps(result, indent=2))
    if "error" in result:
        logger.warning(f"Result: {result['error']}")
    else:
        logger.info(
            f"Primary tree selected: bbox={result['bbox']} "
            f"confidence={result['confidence']:.2%} -> {result['cropped_tree']}"
        )


if __name__ == "__main__":
    main()
