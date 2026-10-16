#!/usr/bin/env python3
"""
prepare_dataset.py
-------------------
Cleans, validates, and converts all raw detection sources (Auto Arborist,
Open Images "Tree" class, NEON Tree Evaluation) into a single unified
YOLO-format dataset:

    data/processed/images/<source>_<idx>.jpg
    data/processed/labels/<source>_<idx>.txt   # "0 x_center y_center w h" (normalized)

Steps:
1. Image cleaning: drop corrupt/unreadable images, drop near-blank images.
2. Label validation: drop malformed boxes (out-of-bounds, zero/negative
   area, below min_box_size_px), cap max boxes/image against corrupt files.
3. YOLO annotation conversion: format-specific converters for each source
   registered in configs/config.yaml -> sources.
4. Image resizing: long edge capped for consistent disk usage while
   preserving enough resolution for small/distant trees.

Usage:
    python data/prepare_dataset.py --config configs/config.yaml
"""

import argparse
import csv
import json
import logging
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import yaml
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
)
logger = logging.getLogger("prepare_dataset")

MAX_LONG_EDGE = 1280           # px; preserves detail for small/distant trees
MIN_STD_DEV = 5.0              # reject near-blank/corrupt images
VALID_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# -----------------------------------------------------------------
# Box utilities
# -----------------------------------------------------------------

def clip01(v: float) -> float:
    return max(0.0, min(1.0, v))


def xyxy_norm_to_yolo(xmin, ymin, xmax, ymax) -> tuple:
    """Convert normalized [0,1] xyxy box to YOLO (x_center, y_center, w, h)."""
    xmin, xmax = sorted((clip01(xmin), clip01(xmax)))
    ymin, ymax = sorted((clip01(ymin), clip01(ymax)))
    w = xmax - xmin
    h = ymax - ymin
    x_center = xmin + w / 2
    y_center = ymin + h / 2
    return x_center, y_center, w, h


def is_valid_yolo_box(x_center, y_center, w, h, img_w, img_h, min_box_size_px) -> bool:
    if w <= 0 or h <= 0:
        return False
    if not (0.0 <= x_center <= 1.0 and 0.0 <= y_center <= 1.0):
        return False
    box_w_px = w * img_w
    box_h_px = h * img_h
    if box_w_px < min_box_size_px or box_h_px < min_box_size_px:
        return False
    return True


def is_low_information(img: Image.Image) -> bool:
    arr = np.asarray(img.convert("L"), dtype=np.float32)
    return float(arr.std()) < MIN_STD_DEV


# -----------------------------------------------------------------
# Format-specific converters: each returns {image_path: [(xmin,ymin,xmax,ymax_norm), ...]}
# -----------------------------------------------------------------

def convert_auto_arborist_json(src_dir: Path) -> dict:
    """Auto Arborist per-tree JSON records with normalized xmin/xmax/ymin/ymax."""
    results = {}
    json_files = list(src_dir.rglob("*.json"))
    for jf in json_files:
        try:
            with open(jf, "r") as f:
                record = json.load(f)
        except Exception as e:
            logger.warning(f"  Skipping unreadable JSON {jf}: {e}")
            continue

        img_path = jf.with_suffix(".jpg")
        if not img_path.exists():
            img_path = jf.with_suffix(".png")
        if not img_path.exists():
            continue

        trees = record.get("trees", [record]) if isinstance(record, dict) else record
        boxes = []
        for t in trees:
            try:
                boxes.append((t["xmin"], t["ymin"], t["xmax"], t["ymax"]))
            except (KeyError, TypeError):
                continue
        if boxes:
            results[img_path] = boxes
    return results


def convert_open_images_csv(src_dir: Path) -> dict:
    """OIDv7-style CSV: ImageID,XMin,XMax,YMin,YMax (normalized), images/ subfolder."""
    results = {}
    csv_files = list(src_dir.rglob("*.csv"))
    images_dir = src_dir / "images"

    for cf in csv_files:
        try:
            with open(cf, "r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    image_id = row.get("ImageID")
                    if not image_id:
                        continue
                    img_path = None
                    for ext in VALID_IMAGE_EXTS:
                        candidate = images_dir / f"{image_id}{ext}"
                        if candidate.exists():
                            img_path = candidate
                            break
                    if img_path is None:
                        continue
                    try:
                        box = (float(row["XMin"]), float(row["YMin"]),
                               float(row["XMax"]), float(row["YMax"]))
                    except (KeyError, ValueError):
                        continue
                    results.setdefault(img_path, []).append(box)
        except Exception as e:
            logger.warning(f"  Skipping unreadable CSV {cf}: {e}")
    return results


def convert_pascal_voc_xml(src_dir: Path) -> dict:
    """NEON-style Pascal VOC XML with absolute-pixel <bndbox> coordinates."""
    results = {}
    images_dir = src_dir / "images"
    annotations_dir = src_dir / "annotations"
    xml_files = list(annotations_dir.rglob("*.xml")) if annotations_dir.exists() else []

    for xf in xml_files:
        try:
            tree = ET.parse(xf)
            root = tree.getroot()
        except ET.ParseError as e:
            logger.warning(f"  Skipping malformed XML {xf}: {e}")
            continue

        size_elem = root.find("size")
        if size_elem is None:
            continue
        img_w = float(size_elem.findtext("width", default="0"))
        img_h = float(size_elem.findtext("height", default="0"))
        if img_w <= 0 or img_h <= 0:
            continue

        img_path = None
        for ext in VALID_IMAGE_EXTS:
            candidate = images_dir / f"{xf.stem}{ext}"
            if candidate.exists():
                img_path = candidate
                break
        if img_path is None:
            continue

        boxes = []
        for obj in root.findall("object"):
            bndbox = obj.find("bndbox")
            if bndbox is None:
                continue
            try:
                xmin = float(bndbox.findtext("xmin")) / img_w
                ymin = float(bndbox.findtext("ymin")) / img_h
                xmax = float(bndbox.findtext("xmax")) / img_w
                ymax = float(bndbox.findtext("ymax")) / img_h
            except (TypeError, ValueError):
                continue
            boxes.append((xmin, ymin, xmax, ymax))
        if boxes:
            results[img_path] = boxes
    return results


CONVERTERS = {
    "auto_arborist_json": convert_auto_arborist_json,
    "open_images_csv": convert_open_images_csv,
    "pascal_voc_xml": convert_pascal_voc_xml,
}


# -----------------------------------------------------------------
# Main processing
# -----------------------------------------------------------------

def process_image_and_boxes(img_path: Path, boxes_norm: list, dst_images_dir: Path,
                             dst_labels_dir: Path, out_stem: str, min_box_size_px: int,
                             max_boxes_per_image: int) -> bool:
    try:
        with Image.open(img_path) as img:
            img = img.convert("RGB")
            if is_low_information(img):
                return False
            orig_w, orig_h = img.size

            if max(orig_w, orig_h) > MAX_LONG_EDGE:
                scale = MAX_LONG_EDGE / max(orig_w, orig_h)
                new_w, new_h = int(orig_w * scale), int(orig_h * scale)
                img = img.resize((new_w, new_h), Image.LANCZOS)
            final_w, final_h = img.size

            valid_yolo_boxes = []
            for (xmin, ymin, xmax, ymax) in boxes_norm[:max_boxes_per_image]:
                x_c, y_c, w, h = xyxy_norm_to_yolo(xmin, ymin, xmax, ymax)
                if is_valid_yolo_box(x_c, y_c, w, h, final_w, final_h, min_box_size_px):
                    valid_yolo_boxes.append((x_c, y_c, w, h))

            if not valid_yolo_boxes:
                return False

            dst_images_dir.mkdir(parents=True, exist_ok=True)
            dst_labels_dir.mkdir(parents=True, exist_ok=True)

            img.save(dst_images_dir / f"{out_stem}.jpg", format="JPEG", quality=92)
            with open(dst_labels_dir / f"{out_stem}.txt", "w") as f:
                for x_c, y_c, w, h in valid_yolo_boxes:
                    f.write(f"0 {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}\n")
        return True
    except Exception as e:
        logger.warning(f"  Skipping corrupt/unreadable image {img_path}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    raw_dir = Path(cfg["paths"]["raw_data_dir"])
    processed_dir = Path(cfg["paths"]["processed_data_dir"])
    min_box_size_px = cfg["data"]["min_box_size_px"]
    max_boxes_per_image = cfg["data"]["max_boxes_per_image"]

    if processed_dir.exists():
        logger.info(f"Clearing existing processed dir: {processed_dir}")
        shutil.rmtree(processed_dir)

    dst_images_dir = processed_dir / "images"
    dst_labels_dir = processed_dir / "labels"

    summary = {}
    for source_name, source_cfg in cfg["sources"].items():
        src_dir = raw_dir / source_cfg["raw_subdir"]
        fmt = source_cfg["format"]
        converter = CONVERTERS.get(fmt)

        if not src_dir.exists():
            logger.warning(f"[{source_name}] Missing raw dir (skipped): {src_dir}")
            summary[source_name] = 0
            continue
        if converter is None:
            logger.warning(f"[{source_name}] Unknown format '{fmt}' (skipped)")
            summary[source_name] = 0
            continue

        logger.info(f"[{source_name}] Converting with format '{fmt}'...")
        image_to_boxes = converter(src_dir)
        logger.info(f"[{source_name}] Found {len(image_to_boxes)} annotated images")

        count = 0
        for idx, (img_path, boxes) in enumerate(image_to_boxes.items()):
            out_stem = f"{source_name}_{idx:06d}"
            if process_image_and_boxes(
                Path(img_path), boxes, dst_images_dir, dst_labels_dir, out_stem,
                min_box_size_px, max_boxes_per_image,
            ):
                count += 1

        summary[source_name] = count
        logger.info(f"[{source_name}] Successfully processed: {count} images")

    total = sum(summary.values())
    logger.info("=" * 60)
    logger.info(f"Merge complete: {total} images across {len(summary)} sources")
    for source_name, count in summary.items():
        logger.info(f"  {source_name}: {count}")
    if total == 0:
        logger.warning(
            "No images were processed. Verify raw dataset directories match "
            "the structure documented in configs/config.yaml -> sources."
        )
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
