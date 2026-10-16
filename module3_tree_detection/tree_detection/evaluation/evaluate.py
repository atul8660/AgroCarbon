#!/usr/bin/env python3
"""
evaluate.py
------------
Evaluates a trained YOLOv11 checkpoint on the held-out test split.

Produces:
- mAP@0.5, mAP@0.5:0.95
- Precision, Recall, F1 (derived from ultralytics' validator)
- PR curve (ultralytics-generated, copied to outputs/plots)
- Confusion matrix (ultralytics-generated, copied to outputs/plots)
- Inference speed (ms/image, FPS), measured directly on the test set

Usage:
    python evaluation/evaluate.py --config configs/config.yaml \
        --checkpoint outputs/checkpoints/best.pt
"""

import argparse
import json
import logging
import shutil
import sys
import time
from pathlib import Path

import yaml

sys.path.append(str(Path(__file__).resolve().parents[1]))

from models.model import load_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
)
logger = logging.getLogger("evaluate")


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def measure_inference_speed(model, image_paths: list, imgsz: int, device: str,
                             warmup: int = 5) -> dict:
    """Measures average inference latency/FPS over the test set (single-image
    batches, matching real serving conditions for Module 3 -> Module 4)."""
    sample = image_paths[:warmup]
    for p in sample:
        model.predict(source=str(p), imgsz=imgsz, device=device, verbose=False)

    timings = []
    for p in image_paths:
        start = time.perf_counter()
        model.predict(source=str(p), imgsz=imgsz, device=device, verbose=False)
        timings.append(time.perf_counter() - start)

    avg_ms = (sum(timings) / len(timings)) * 1000
    fps = 1000.0 / avg_ms if avg_ms > 0 else 0.0
    return {"avg_latency_ms": avg_ms, "fps": fps, "num_images": len(image_paths)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--split", type=str, default="test", choices=["val", "test"])
    parser.add_argument("--speed_sample_size", type=int, default=100,
                         help="Number of test images to time for inference speed")
    args = parser.parse_args()

    cfg = load_config(args.config)
    eval_cfg = cfg["evaluation"]
    device = "0" if cfg["project"]["device"] == "cuda" else "cpu"

    dataset_yaml = Path(cfg["paths"]["splits_dir"]) / "dataset.yaml"
    model = load_model(args.checkpoint)

    logger.info(f"Running ultralytics validation on '{args.split}' split...")
    plots_dir = Path(cfg["paths"]["plots_dir"])
    plots_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path(cfg["paths"]["log_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)

    metrics = model.val(
        data=str(dataset_yaml),
        split=args.split,
        conf=eval_cfg["conf_threshold"],
        iou=eval_cfg["iou_threshold"],
        plots=True,             # generates PR curve + confusion matrix PNGs
        project=str(Path(cfg["paths"]["output_dir"]) / "yolo_eval_runs"),
        name=args.split,
        exist_ok=True,
    )

    precision = float(metrics.box.mp)   # mean precision across classes
    recall = float(metrics.box.mr)      # mean recall across classes
    map50 = float(metrics.box.map50)
    map50_95 = float(metrics.box.map)
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    logger.info("=" * 70)
    logger.info(f"mAP@0.5:      {map50:.4f}")
    logger.info(f"mAP@0.5:0.95: {map50_95:.4f}")
    logger.info(f"Precision:    {precision:.4f}")
    logger.info(f"Recall:       {recall:.4f}")
    logger.info(f"F1:           {f1:.4f}")
    logger.info("=" * 70)

    # Copy ultralytics-generated PR curve + confusion matrix into our
    # standard plots_dir for consistency with Module 1/2's output layout.
    run_save_dir = Path(metrics.save_dir)
    for fname in ("PR_curve.png", "confusion_matrix.png", "confusion_matrix_normalized.png"):
        src = run_save_dir / fname
        if src.exists():
            dst = plots_dir / f"{args.split}_{fname}"
            shutil.copy2(src, dst)
            logger.info(f"Copied {src} -> {dst}")

    # Inference speed on a sample of the split (single-image latency, as
    # Module 3 will be called one scene image at a time in production).
    images_dir = Path(cfg["paths"]["splits_dir"]) / "images" / args.split
    image_paths = sorted(images_dir.glob("*.jpg"))[: args.speed_sample_size]
    speed_stats = {}
    if image_paths:
        logger.info(f"Measuring inference speed on {len(image_paths)} images...")
        speed_stats = measure_inference_speed(
            model, image_paths, cfg["model"]["input_size"], device
        )
        logger.info(
            f"Inference speed: {speed_stats['avg_latency_ms']:.2f} ms/image "
            f"({speed_stats['fps']:.1f} FPS) on device={device}"
        )

    metrics_summary = {
        "split": args.split,
        "map50": map50,
        "map50_95": map50_95,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "inference_speed": speed_stats,
    }
    metrics_path = log_dir / f"{args.split}_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics_summary, f, indent=2)
    logger.info(f"Saved metrics summary: {metrics_path}")


if __name__ == "__main__":
    main()
