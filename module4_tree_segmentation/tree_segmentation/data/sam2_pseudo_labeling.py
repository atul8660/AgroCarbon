#!/usr/bin/env python3
"""
sam2_pseudo_labeling.py
-------------------------
Uses SAM2, prompted with NEON Tree Evaluation bounding boxes, to generate
pseudo binary tree masks. This is SAM2's supporting role in this module (see
README section 2 / design notes) — bootstrapping training data, not serving
production inference.

NEON ships boxes, not pixel masks (aerial canopy-crown benchmark). A
box-prompted SAM2 call turns each box into a plausible tree silhouette mask,
which prepare_dataset.py then ingests via the "pseudo_mask_png" source
format, same on-disk layout as the Open Images converter.

Output:
    data/raw/neon_pseudo_labeled/images/<idx>.jpg
    data/raw/neon_pseudo_labeled/masks/<idx>.png

Usage:
    python data/sam2_pseudo_labeling.py --config configs/config.yaml
"""

import argparse
import logging
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np
from omegaconf import OmegaConf

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
)
logger = logging.getLogger("sam2_pseudo_labeling")


def load_config(config_path: str):
    return OmegaConf.load(config_path)


def load_sam2_predictor(checkpoint: str, model_config: str, device: str):
    """Lazily imports and builds the SAM2 image predictor. Raises a clear,
    actionable error if the sam2 package / checkpoint isn't available, since
    this whole script is an optional data-bootstrapping step, not a hard
    dependency of the rest of the module."""
    try:
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
    except ImportError as e:
        raise ImportError(
            "The 'sam2' package is required for pseudo-labeling. "
            "Install it per https://github.com/facebookresearch/sam2 and "
            "download the checkpoint referenced in configs/config.yaml -> model.sam2.checkpoint."
        ) from e

    if not Path(checkpoint).exists():
        raise FileNotFoundError(
            f"SAM2 checkpoint not found at {checkpoint}. Download it per "
            f"https://github.com/facebookresearch/sam2#model-checkpoints "
            f"and update configs/config.yaml -> model.sam2.checkpoint."
        )

    sam2_model = build_sam2(model_config, checkpoint, device=device)
    predictor = SAM2ImagePredictor(sam2_model)
    return predictor


def parse_neon_boxes(xml_path: Path) -> list:
    """Parses Pascal-VOC-style NEON annotation XML (same format used by
    Module 3's converter) into a list of absolute-pixel [xmin,ymin,xmax,ymax] boxes."""
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except ET.ParseError:
        return []

    boxes = []
    for obj in root.findall("object"):
        bndbox = obj.find("bndbox")
        if bndbox is None:
            continue
        try:
            boxes.append([
                float(bndbox.findtext("xmin")),
                float(bndbox.findtext("ymin")),
                float(bndbox.findtext("xmax")),
                float(bndbox.findtext("ymax")),
            ])
        except (TypeError, ValueError):
            continue
    return boxes


def generate_pseudo_mask(predictor, image_rgb: np.ndarray, boxes: list) -> np.ndarray:
    """Runs SAM2 once per box (best single-object accuracy) and unions the
    resulting masks into one binary tree mask covering all boxes in the image."""
    predictor.set_image(image_rgb)
    h, w = image_rgb.shape[:2]
    union_mask = np.zeros((h, w), dtype=np.uint8)

    for box in boxes:
        box_arr = np.array(box)
        masks, scores, _ = predictor.predict(box=box_arr, multimask_output=False)
        best_mask = masks[0].astype(np.uint8)
        union_mask = np.maximum(union_mask, best_mask)

    return union_mask


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if not cfg.model.sam2.enabled_for_pseudo_labeling:
        logger.info("model.sam2.enabled_for_pseudo_labeling is False — skipping.")
        return

    device = "cuda" if cfg.project.device == "cuda" else "cpu"
    predictor = load_sam2_predictor(
        cfg.model.sam2.checkpoint, cfg.model.sam2.model_config, device
    )

    neon_dir = Path(cfg.paths.raw_data_dir) / "neon_raw" / "repo"
    images_dir = neon_dir / "evaluation"
    annotations_dir = neon_dir / "annotations"

    out_dir = Path(cfg.paths.raw_data_dir) / "neon_pseudo_labeled"
    out_images_dir = out_dir / "images"
    out_masks_dir = out_dir / "masks"
    out_images_dir.mkdir(parents=True, exist_ok=True)
    out_masks_dir.mkdir(parents=True, exist_ok=True)

    if not annotations_dir.exists():
        logger.warning(f"NEON annotations dir not found: {annotations_dir}. Run download_datasets.sh first.")
        return

    count = 0
    for idx, xml_path in enumerate(sorted(annotations_dir.glob("*.xml"))):
        img_path = images_dir / f"{xml_path.stem}.tif"
        if not img_path.exists():
            img_path = images_dir / f"{xml_path.stem}.jpg"
        if not img_path.exists():
            continue

        boxes = parse_neon_boxes(xml_path)
        if not boxes:
            continue

        image_bgr = cv2.imread(str(img_path))
        if image_bgr is None:
            continue
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        mask = generate_pseudo_mask(predictor, image_rgb, boxes)
        if mask.sum() == 0:
            continue

        out_stem = f"neon_pseudo_{idx:06d}"
        cv2.imwrite(str(out_images_dir / f"{out_stem}.jpg"), image_bgr)
        cv2.imwrite(str(out_masks_dir / f"{out_stem}.png"), mask * 255)
        count += 1

        if count % 50 == 0:
            logger.info(f"Pseudo-labeled {count} images so far...")

    logger.info(f"Done. Generated {count} SAM2 pseudo-masks -> {out_dir}")


if __name__ == "__main__":
    main()
