#!/usr/bin/env python3
"""
make_splits.py
---------------
Creates train/val/test splits from the unified processed dataset
(data/processed/images/*.jpg + data/processed/labels/*.txt), stratified by
source dataset (so, e.g., Open Images "complex scene" images aren't
concentrated only in train), and writes the YOLO-format directory layout
YOLOv11 expects, plus the dataset.yaml training config.

Output layout:
    data/splits/
        images/{train,val,test}/*.jpg
        labels/{train,val,test}/*.txt
        dataset.yaml

Usage:
    python data/make_splits.py --config configs/config.yaml
"""

import argparse
import logging
import shutil
from collections import defaultdict
from pathlib import Path

import yaml
from sklearn.model_selection import train_test_split

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
)
logger = logging.getLogger("make_splits")


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def source_of(image_path: Path) -> str:
    """Recovers the source dataset name from the '<source>_<idx>.jpg' filename
    produced by prepare_dataset.py, for stratified splitting."""
    stem = image_path.stem
    return stem.rsplit("_", 1)[0]


def stratified_split_by_source(image_paths: list, train_frac: float, val_frac: float,
                                test_frac: float, seed: int) -> tuple:
    by_source = defaultdict(list)
    for p in image_paths:
        by_source[source_of(p)].append(p)

    train, val, test = [], [], []
    for source_name, paths in by_source.items():
        if len(paths) < 3:
            # Too few to split meaningfully — put them all in train
            train.extend(paths)
            continue
        train_val, t = train_test_split(paths, test_size=test_frac, random_state=seed)
        relative_val = val_frac / (train_frac + val_frac)
        tr, v = train_test_split(train_val, test_size=relative_val, random_state=seed)
        train.extend(tr)
        val.extend(v)
        test.extend(t)

    return train, val, test


def copy_split(image_paths: list, labels_dir: Path, dst_images_dir: Path,
                dst_labels_dir: Path) -> int:
    dst_images_dir.mkdir(parents=True, exist_ok=True)
    dst_labels_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for img_path in image_paths:
        label_path = labels_dir / f"{img_path.stem}.txt"
        if not label_path.exists():
            continue
        shutil.copy2(img_path, dst_images_dir / img_path.name)
        shutil.copy2(label_path, dst_labels_dir / label_path.name)
        count += 1
    return count


def write_dataset_yaml(splits_dir: Path, classes: list) -> Path:
    dataset_yaml = {
        "path": str(splits_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {i: name for i, name in enumerate(classes)},
    }
    yaml_path = splits_dir / "dataset.yaml"
    with open(yaml_path, "w") as f:
        yaml.safe_dump(dataset_yaml, f, sort_keys=False)
    return yaml_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    processed_dir = Path(cfg["paths"]["processed_data_dir"])
    splits_dir = Path(cfg["paths"]["splits_dir"])

    images_dir = processed_dir / "images"
    labels_dir = processed_dir / "labels"

    image_paths = sorted(images_dir.glob("*.jpg"))
    if not image_paths:
        raise RuntimeError(
            f"No images found in {images_dir}. Run prepare_dataset.py first."
        )

    if splits_dir.exists():
        logger.info(f"Clearing existing splits dir: {splits_dir}")
        shutil.rmtree(splits_dir)

    train, val, test = stratified_split_by_source(
        image_paths,
        cfg["data"]["train_split"],
        cfg["data"]["val_split"],
        cfg["data"]["test_split"],
        cfg["project"]["seed"],
    )

    n_train = copy_split(train, labels_dir, splits_dir / "images" / "train", splits_dir / "labels" / "train")
    n_val = copy_split(val, labels_dir, splits_dir / "images" / "val", splits_dir / "labels" / "val")
    n_test = copy_split(test, labels_dir, splits_dir / "images" / "test", splits_dir / "labels" / "test")

    yaml_path = write_dataset_yaml(splits_dir, cfg["classes"])

    logger.info("=" * 60)
    logger.info(f"Train: {n_train} | Val: {n_val} | Test: {n_test}")
    logger.info(f"dataset.yaml written to: {yaml_path}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
