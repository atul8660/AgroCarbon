#!/usr/bin/env python3
"""
prepare_dataset.py
-------------------
Merges heterogeneous raw datasets (data/raw/<dataset>/<class>/*.jpg) into a
single unified health taxonomy (data/processed/<health_class>/*.jpg) as
defined in configs/config.yaml -> class_map.

Cleaning steps:
- Drops corrupt/unreadable images.
- Drops near-empty / near-solid images (e.g. blank backgrounds, scanner
  artifacts) via a low-variance filter — a common contamination issue in
  scraped datasets like PlantDoc.
- Deduplicates via exact content hash across merged sources.
- Converts to RGB JPEG, caps long edge for consistent disk usage while
  preserving lesion/spot-level detail.
- Label encoding is handled downstream in make_splits.py (canonical class
  list -> sorted integer labels), keeping this script purely about producing
  clean image files.

Usage:
    python data/prepare_dataset.py --config configs/config.yaml
"""

import argparse
import hashlib
import logging
import shutil
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

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
MAX_LONG_EDGE = 800          # px; preserves lesion/spot-level texture detail
MIN_STD_DEV = 5.0            # pixel-intensity std-dev threshold to reject blank/corrupt images


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def file_hash(path: Path, chunk_size: int = 65536) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def is_low_information(img: Image.Image) -> bool:
    """Flags near-blank / near-solid images (common scraping artifacts)."""
    arr = np.asarray(img.convert("L"), dtype=np.float32)
    return float(arr.std()) < MIN_STD_DEV


def resize_and_save(src_path: Path, dst_path: Path) -> bool:
    try:
        with Image.open(src_path) as img:
            img = img.convert("RGB")
            if is_low_information(img):
                logger.warning(f"  Rejecting low-information image: {src_path}")
                return False
            w, h = img.size
            if max(w, h) > MAX_LONG_EDGE:
                scale = MAX_LONG_EDGE / max(w, h)
                img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            img.save(dst_path, format="JPEG", quality=92)
        return True
    except Exception as e:
        logger.warning(f"  Skipping corrupt/unreadable image {src_path}: {e}")
        return False


def merge_sources(cfg: dict) -> dict:
    raw_dir = Path(cfg["paths"]["raw_data_dir"])
    processed_dir = Path(cfg["paths"]["processed_data_dir"])
    class_map = cfg["class_map"]
    min_per_class = cfg["data"]["min_images_per_class"]

    if processed_dir.exists():
        logger.info(f"Clearing existing processed dir: {processed_dir}")
        shutil.rmtree(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    summary = {}
    seen_hashes = set()

    for health_class, source_subdirs in class_map.items():
        out_dir = processed_dir / health_class
        count = 0
        idx = 0

        for subdir in source_subdirs:
            src_dir = raw_dir / subdir
            if not src_dir.exists():
                logger.warning(f"[{health_class}] Missing source dir (skipped): {src_dir}")
                continue

            image_paths = [
                p for p in src_dir.rglob("*")
                if p.suffix.lower() in VALID_EXTENSIONS and p.is_file()
            ]
            logger.info(f"[{health_class}] Found {len(image_paths)} images in {src_dir}")

            for src_path in image_paths:
                try:
                    h = file_hash(src_path)
                except Exception:
                    continue
                if h in seen_hashes:
                    continue  # exact duplicate across datasets
                seen_hashes.add(h)

                dst_path = out_dir / f"{health_class}_{idx:06d}.jpg"
                if resize_and_save(src_path, dst_path):
                    count += 1
                    idx += 1

        summary[health_class] = count
        status = "OK" if count >= min_per_class else "LOW-COUNT (flagged)"
        logger.info(f"[{health_class}] Total merged images: {count}  [{status}]")

    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    summary = merge_sources(cfg)

    total = sum(summary.values())
    n_classes = len(summary)
    flagged = [s for s, c in summary.items() if c < cfg["data"]["min_images_per_class"]]

    logger.info("=" * 60)
    logger.info(f"Merge complete: {total} images across {n_classes} health classes")
    if flagged:
        logger.warning(
            f"{len(flagged)} classes below min_images_per_class threshold: {flagged}"
        )
        logger.warning(
            "Nutrient_Deficient / Pest_Affected commonly land here until "
            "domain-specific data is added — see README section 6."
        )
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
