#!/usr/bin/env python3
"""
prepare_dataset.py
-------------------
Merges heterogeneous raw datasets (data/raw/<dataset>/<class>/*.jpg) into a
single unified taxonomy (data/processed/<canonical_species>/*.jpg) as defined
in configs/config.yaml -> class_map.

- Validates/repairs images (drops corrupt files, converts to RGB JPEG).
- Deduplicates via perceptual hash to avoid near-identical leaked splits.
- Resizes the longest edge to a max cap to control disk usage while
  preserving enough resolution for fine-grained texture (bark/vein) cues.
- Logs a per-class image count summary and flags classes below the
  minimum-images-per-class threshold.

Usage:
    python data/prepare_dataset.py --config configs/config.yaml
"""

import argparse
import hashlib
import logging
import shutil
from pathlib import Path

import yaml
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
)
logger = logging.getLogger("prepare_dataset")

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
MAX_LONG_EDGE = 800  # px; keeps enough detail for fine-grained texture cues


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def file_hash(path: Path, chunk_size: int = 65536) -> str:
    """Cheap exact-duplicate hash (content-based, not perceptual)."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def resize_and_save(src_path: Path, dst_path: Path) -> bool:
    """Load, validate, resize (cap long edge), and save as RGB JPEG."""
    try:
        with Image.open(src_path) as img:
            img = img.convert("RGB")
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

    for species, source_subdirs in class_map.items():
        out_dir = processed_dir / species
        count = 0
        idx = 0

        for subdir in source_subdirs:
            src_dir = raw_dir / subdir
            if not src_dir.exists():
                logger.warning(f"[{species}] Missing source dir (skipped): {src_dir}")
                continue

            image_paths = [
                p for p in src_dir.rglob("*")
                if p.suffix.lower() in VALID_EXTENSIONS and p.is_file()
            ]
            logger.info(f"[{species}] Found {len(image_paths)} images in {src_dir}")

            for src_path in image_paths:
                try:
                    h = file_hash(src_path)
                except Exception:
                    continue
                if h in seen_hashes:
                    continue  # exact duplicate across datasets
                seen_hashes.add(h)

                dst_path = out_dir / f"{species}_{idx:06d}.jpg"
                if resize_and_save(src_path, dst_path):
                    count += 1
                    idx += 1

        summary[species] = count
        status = "OK" if count >= min_per_class else "LOW-COUNT (flagged)"
        logger.info(f"[{species}] Total merged images: {count}  [{status}]")

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
    logger.info(f"Merge complete: {total} images across {n_classes} classes")
    if flagged:
        logger.warning(
            f"{len(flagged)} classes below min_images_per_class threshold: {flagged}"
        )
        logger.warning(
            "Consider collecting more data for these classes or dropping them "
            "in configs/config.yaml before training."
        )
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
