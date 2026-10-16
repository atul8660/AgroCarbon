#!/usr/bin/env python3
"""
dataset_stats.py
------------------
Reports dataset statistics for the split YOLO dataset: per-split image/box
counts, per-source composition, and a box-size distribution histogram — used
to sanity-check the merge before spending compute on training (e.g. catching
a source that's been reduced to near-nothing by label validation).

Usage:
    python data/dataset_stats.py --config configs/config.yaml
"""

import argparse
import logging
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
)
logger = logging.getLogger("dataset_stats")


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def source_of(stem: str) -> str:
    return stem.rsplit("_", 1)[0]


def collect_split_stats(images_dir: Path, labels_dir: Path) -> dict:
    box_widths, box_heights, box_areas = [], [], []
    boxes_per_image = []
    source_counts = Counter()

    for img_path in images_dir.glob("*.jpg"):
        label_path = labels_dir / f"{img_path.stem}.txt"
        source_counts[source_of(img_path.stem)] += 1
        if not label_path.exists():
            boxes_per_image.append(0)
            continue
        with open(label_path, "r") as f:
            lines = [ln.strip() for ln in f if ln.strip()]
        boxes_per_image.append(len(lines))
        for line in lines:
            parts = line.split()
            if len(parts) != 5:
                continue
            _, x_c, y_c, w, h = map(float, parts)
            box_widths.append(w)
            box_heights.append(h)
            box_areas.append(w * h)

    return {
        "num_images": len(list(images_dir.glob("*.jpg"))),
        "num_boxes": len(box_areas),
        "boxes_per_image": boxes_per_image,
        "box_widths": box_widths,
        "box_heights": box_heights,
        "box_areas": box_areas,
        "source_counts": dict(source_counts),
    }


def plot_box_size_histogram(all_stats: dict, out_path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for split_name, stats in all_stats.items():
        if stats["box_areas"]:
            axes[0].hist(stats["box_areas"], bins=40, alpha=0.5, label=split_name)
        if stats["boxes_per_image"]:
            axes[1].hist(stats["boxes_per_image"], bins=range(0, 20), alpha=0.5, label=split_name)

    axes[0].set_title("Normalized Box Area Distribution")
    axes[0].set_xlabel("box_area (normalized, w*h)")
    axes[0].legend()

    axes[1].set_title("Boxes per Image Distribution")
    axes[1].set_xlabel("num_boxes")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    logger.info(f"Saved box statistics plot: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    splits_dir = Path(cfg["paths"]["splits_dir"])
    plots_dir = Path(cfg["paths"]["plots_dir"])
    plots_dir.mkdir(parents=True, exist_ok=True)

    all_stats = {}
    for split_name in ("train", "val", "test"):
        images_dir = splits_dir / "images" / split_name
        labels_dir = splits_dir / "labels" / split_name
        if not images_dir.exists():
            logger.warning(f"Split dir missing (skipped): {images_dir}")
            continue
        stats = collect_split_stats(images_dir, labels_dir)
        all_stats[split_name] = stats

        logger.info("=" * 60)
        logger.info(f"[{split_name}] Images: {stats['num_images']} | Boxes: {stats['num_boxes']}")
        avg_boxes = np.mean(stats["boxes_per_image"]) if stats["boxes_per_image"] else 0
        logger.info(f"[{split_name}] Avg boxes/image: {avg_boxes:.2f}")
        logger.info(f"[{split_name}] Per-source image counts: {stats['source_counts']}")
        if stats["box_areas"]:
            areas = np.array(stats["box_areas"])
            logger.info(
                f"[{split_name}] Box area (normalized) — "
                f"min: {areas.min():.4f} | mean: {areas.mean():.4f} | max: {areas.max():.4f}"
            )

    if all_stats:
        plot_box_size_histogram(all_stats, plots_dir / "dataset_statistics.png")

    logger.info("=" * 60)
    logger.info("Dataset statistics complete.")


if __name__ == "__main__":
    main()
