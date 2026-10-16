#!/usr/bin/env python3
"""
prepare_dataset.py
-------------------
Cleans, validates, and converts all raw segmentation sources into a unified
processed dataset:

    data/processed/images/<source>_<idx>.jpg
    data/processed/masks/<source>_<idx>.png   # single-channel, pixel values:
                                                #   0 = background, 1 = canopy, 2 = trunk
    data/processed/manifest.csv                # image, mask, source, has_part_supervision

Design note on part supervision: only `urban_street_tree` ships explicit
trunk polygons. Sources without trunk annotations (TreeSeg, Open Images,
NEON-pseudo) have their entire foreground assigned to class 1 (canopy) as a
safe default, and are flagged `has_part_supervision=False` in the manifest.
`training/engine.py` uses that flag to collapse the canopy/trunk loss to a
binary tree-vs-background loss for those samples, so the model is never
taught a false "this is definitely canopy, not trunk" signal from sources
that simply didn't label the trunk.

Steps:
1. Image cleaning: drop corrupt/unreadable images, drop near-blank images.
2. Mask validation: drop masks below min_mask_area_px / min_mask_coverage_ratio,
   drop masks with no foreground pixels at all.
3. Polygon -> mask conversion for polygon-based sources (COCO-style and the
   Urban Street Tree per-organ polygon format).
4. Resizing: image + mask resized together (nearest-neighbor for masks) to
   a consistent long-edge cap.

Usage:
    python data/prepare_dataset.py --config configs/config.yaml
"""

import argparse
import csv
import json
import logging
import shutil
from pathlib import Path

import cv2
import numpy as np
from omegaconf import OmegaConf
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
)
logger = logging.getLogger("prepare_dataset")

MAX_LONG_EDGE = 1024
MIN_STD_DEV = 5.0
VALID_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def load_config(config_path: str):
    return OmegaConf.load(config_path)


def is_low_information(img: Image.Image) -> bool:
    arr = np.asarray(img.convert("L"), dtype=np.float32)
    return float(arr.std()) < MIN_STD_DEV


# -----------------------------------------------------------------
# Polygon -> mask rasterization
# -----------------------------------------------------------------

def polygon_to_mask(polygons: list, height: int, width: int, class_id: int,
                     mask: np.ndarray = None) -> np.ndarray:
    """Rasterizes a list of [x1,y1,x2,y2,...] flat polygons onto (or into a
    new) single-channel mask, stamping `class_id` inside each polygon."""
    if mask is None:
        mask = np.zeros((height, width), dtype=np.uint8)
    for poly in polygons:
        pts = np.array(poly, dtype=np.float64).reshape(-1, 2)
        pts = np.round(pts).astype(np.int32)
        cv2.fillPoly(mask, [pts], color=class_id)
    return mask


# -----------------------------------------------------------------
# Format-specific converters: each yields (image_path, mask_array, has_part_supervision)
# -----------------------------------------------------------------

def convert_coco_polygon(src_dir: Path):
    """TreeSeg-style Roboflow COCO export: single 'tree' category, no organ split."""
    ann_files = list(src_dir.rglob("*.json"))
    images_dir = src_dir
    for ann_file in ann_files:
        try:
            with open(ann_file, "r") as f:
                coco = json.load(f)
        except Exception as e:
            logger.warning(f"  Skipping unreadable COCO file {ann_file}: {e}")
            continue

        images_by_id = {img["id"]: img for img in coco.get("images", [])}
        anns_by_image = {}
        for ann in coco.get("annotations", []):
            anns_by_image.setdefault(ann["image_id"], []).append(ann)

        for image_id, image_info in images_by_id.items():
            img_path = images_dir / image_info["file_name"]
            if not img_path.exists():
                continue
            h, w = image_info["height"], image_info["width"]
            mask = np.zeros((h, w), dtype=np.uint8)
            polygons = []
            for ann in anns_by_image.get(image_id, []):
                for seg in ann.get("segmentation", []):
                    if isinstance(seg, list) and len(seg) >= 6:
                        polygons.append(seg)
            if not polygons:
                continue
            mask = polygon_to_mask(polygons, h, w, class_id=1, mask=mask)  # whole tree -> canopy
            yield img_path, mask, False


def convert_polygon_json_urban_street_tree(src_dir: Path):
    """Urban Street Tree per-organ polygon JSON: separate 'tree' (canopy proxy)
    and 'trunk' organ subfolders, each with its own polygon annotation files
    keyed by the same base image filename, merged into one 3-class mask."""
    tree_dir = src_dir / "tree"
    trunk_dir = src_dir / "trunk"
    if not tree_dir.exists():
        return

    tree_anns = list((tree_dir / "annotations").rglob("*.json")) if (tree_dir / "annotations").exists() else []

    for ann_file in tree_anns:
        try:
            with open(ann_file, "r") as f:
                record = json.load(f)
        except Exception as e:
            logger.warning(f"  Skipping unreadable JSON {ann_file}: {e}")
            continue

        img_name = record.get("image_filename", ann_file.stem + ".jpg")
        img_path = tree_dir / "images" / img_name
        if not img_path.exists():
            continue

        w, h = record.get("width"), record.get("height")
        if not w or not h:
            with Image.open(img_path) as im:
                w, h = im.size

        mask = np.zeros((h, w), dtype=np.uint8)
        tree_polys = [obj["polygon"] for obj in record.get("objects", []) if "polygon" in obj]
        mask = polygon_to_mask(tree_polys, h, w, class_id=1, mask=mask)  # canopy layer first

        # Overlay trunk polygons (if a matching annotation exists) on top,
        # since trunk pixels are a subset of the tree silhouette.
        trunk_ann_path = trunk_dir / "annotations" / ann_file.name
        has_part_supervision = False
        if trunk_ann_path.exists():
            try:
                with open(trunk_ann_path, "r") as f:
                    trunk_record = json.load(f)
                trunk_polys = [obj["polygon"] for obj in trunk_record.get("objects", []) if "polygon" in obj]
                if trunk_polys:
                    mask = polygon_to_mask(trunk_polys, h, w, class_id=2, mask=mask)
                    has_part_supervision = True
            except Exception as e:
                logger.warning(f"  Could not read trunk annotation {trunk_ann_path}: {e}")

        yield img_path, mask, has_part_supervision


def convert_oid_mask_png(src_dir: Path):
    """OIDv7-style per-instance binary PNG masks in a masks/ subfolder,
    matched to images/ by shared filename stem."""
    images_dir = src_dir / "images"
    masks_dir = src_dir / "masks"
    if not masks_dir.exists():
        return

    for mask_path in masks_dir.glob("*.png"):
        img_path = None
        for ext in VALID_IMAGE_EXTS:
            candidate = images_dir / f"{mask_path.stem}{ext}"
            if candidate.exists():
                img_path = candidate
                break
        if img_path is None:
            continue

        raw_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if raw_mask is None:
            continue
        binary = (raw_mask > 127).astype(np.uint8)
        mask = binary * 1  # whole tree -> canopy
        yield img_path, mask, False


def convert_pseudo_mask_png(src_dir: Path):
    """SAM2-generated pseudo-masks (see sam2_pseudo_labeling.py) — same
    on-disk layout as convert_oid_mask_png."""
    yield from convert_oid_mask_png(src_dir)


CONVERTERS = {
    "coco_polygon": convert_coco_polygon,
    "polygon_json": convert_polygon_json_urban_street_tree,
    "oid_mask_png": convert_oid_mask_png,
    "pseudo_mask_png": convert_pseudo_mask_png,
}


# -----------------------------------------------------------------
# Main processing
# -----------------------------------------------------------------

def process_sample(img_path: Path, mask: np.ndarray, has_part_supervision: bool,
                    dst_images_dir: Path, dst_masks_dir: Path, out_stem: str,
                    min_mask_area_px: int, min_mask_coverage_ratio: float) -> bool:
    try:
        with Image.open(img_path) as img:
            img = img.convert("RGB")
            if is_low_information(img):
                return False
            orig_w, orig_h = img.size

            mask_area = int((mask > 0).sum())
            coverage = mask_area / (orig_w * orig_h) if orig_w * orig_h > 0 else 0.0
            if mask_area < min_mask_area_px or coverage < min_mask_coverage_ratio:
                return False

            if max(orig_w, orig_h) > MAX_LONG_EDGE:
                scale = MAX_LONG_EDGE / max(orig_w, orig_h)
                new_w, new_h = int(orig_w * scale), int(orig_h * scale)
                img = img.resize((new_w, new_h), Image.LANCZOS)
                mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

            dst_images_dir.mkdir(parents=True, exist_ok=True)
            dst_masks_dir.mkdir(parents=True, exist_ok=True)

            img.save(dst_images_dir / f"{out_stem}.jpg", format="JPEG", quality=92)
            cv2.imwrite(str(dst_masks_dir / f"{out_stem}.png"), mask)
        return True
    except Exception as e:
        logger.warning(f"  Skipping corrupt/unreadable sample {img_path}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    raw_dir = Path(cfg.paths.raw_data_dir)
    processed_dir = Path(cfg.paths.processed_data_dir)
    min_mask_area_px = cfg.data.min_mask_area_px
    min_mask_coverage_ratio = cfg.data.min_mask_coverage_ratio

    if processed_dir.exists():
        logger.info(f"Clearing existing processed dir: {processed_dir}")
        shutil.rmtree(processed_dir)

    dst_images_dir = processed_dir / "images"
    dst_masks_dir = processed_dir / "masks"
    manifest_rows = []

    for source_name, source_cfg in cfg.sources.items():
        src_dir = raw_dir / source_cfg.raw_subdir
        fmt = source_cfg.format
        converter = CONVERTERS.get(fmt)

        if not src_dir.exists():
            logger.warning(f"[{source_name}] Missing raw dir (skipped): {src_dir}")
            continue
        if converter is None:
            logger.warning(f"[{source_name}] Unknown format '{fmt}' (skipped)")
            continue

        logger.info(f"[{source_name}] Converting with format '{fmt}'...")
        count = 0
        for idx, (img_path, mask, has_parts) in enumerate(converter(src_dir)):
            out_stem = f"{source_name}_{idx:06d}"
            if process_sample(
                img_path, mask, has_parts, dst_images_dir, dst_masks_dir, out_stem,
                min_mask_area_px, min_mask_coverage_ratio,
            ):
                manifest_rows.append({
                    "image": str(dst_images_dir / f"{out_stem}.jpg"),
                    "mask": str(dst_masks_dir / f"{out_stem}.png"),
                    "source": source_name,
                    "has_part_supervision": has_parts,
                })
                count += 1
        logger.info(f"[{source_name}] Successfully processed: {count} samples")

    processed_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = processed_dir / "manifest.csv"
    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "mask", "source", "has_part_supervision"])
        writer.writeheader()
        writer.writerows(manifest_rows)

    logger.info("=" * 60)
    logger.info(f"Merge complete: {len(manifest_rows)} samples -> {manifest_path}")
    part_supervised = sum(1 for r in manifest_rows if r["has_part_supervision"])
    logger.info(f"Samples with real trunk supervision: {part_supervised} / {len(manifest_rows)}")
    if len(manifest_rows) == 0:
        logger.warning(
            "No samples were processed. Verify raw dataset directories match "
            "the structure documented in configs/config.yaml -> sources."
        )
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
