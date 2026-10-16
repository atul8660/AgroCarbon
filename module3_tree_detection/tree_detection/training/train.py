#!/usr/bin/env python3
"""
train.py
---------
YOLOv11 transfer-learning entrypoint for the whole-tree detector.

Ultralytics' Trainer natively provides mixed precision (AMP, on by default
on CUDA), checkpoint saving (`last.pt` / `best.pt` every epoch), early
stopping (`patience`), and cosine LR scheduling — this script wires our
config.yaml into those mechanics and adds the project's standard logging /
resume-CLI conventions used in Module 1/2.

Usage:
    python training/train.py --config configs/config.yaml
    python training/train.py --config configs/config.yaml --resume outputs/checkpoints/last.pt
"""

import argparse
import logging
import shutil
import sys
from pathlib import Path

import yaml

sys.path.append(str(Path(__file__).resolve().parents[1]))

from models.model import build_model, load_model
from training.utils import set_seed, get_device

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
)
logger = logging.getLogger("train")


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--resume", type=str, default=None,
                         help="Path to a checkpoint (e.g. outputs/checkpoints/last.pt) to resume from")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["project"]["seed"])
    device = get_device(cfg["project"]["device"])
    logger.info(f"Using device: {device}")

    train_cfg = cfg["train"]
    aug_cfg = train_cfg["augmentation"]
    dataset_yaml = Path(cfg["paths"]["splits_dir"]) / "dataset.yaml"
    if not dataset_yaml.exists():
        raise FileNotFoundError(
            f"{dataset_yaml} not found. Run data/make_splits.py first."
        )

    checkpoint_dir = Path(cfg["paths"]["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    run_dir = Path(cfg["paths"]["output_dir"]) / "yolo_runs"

    if args.resume:
        logger.info(f"Resuming training from checkpoint: {args.resume}")
        model = load_model(args.resume)
        results = model.train(resume=True)
    else:
        model = build_model(cfg)
        logger.info(
            f"Starting YOLOv11 fine-tuning: {cfg['model']['architecture']} | "
            f"epochs={train_cfg['epochs']} | batch={train_cfg['batch_size']} | "
            f"imgsz={cfg['model']['input_size']}"
        )
        results = model.train(
            data=str(dataset_yaml),
            epochs=train_cfg["epochs"],
            batch=train_cfg["batch_size"],
            imgsz=cfg["model"]["input_size"],
            lr0=train_cfg["lr0"],
            lrf=train_cfg["lrf"],
            weight_decay=train_cfg["weight_decay"],
            warmup_epochs=train_cfg["warmup_epochs"],
            patience=train_cfg["patience"],          # early stopping
            amp=train_cfg["use_amp"],                # mixed precision
            cache=train_cfg.get("cache_data", False),# dataset caching (ram/disk/False)
            optimizer=train_cfg["optimizer"],
            seed=cfg["project"]["seed"],
            device=device,
            project=str(run_dir.parent),
            name=run_dir.name,
            exist_ok=True,
            # Augmentation (as required: rotation/flip handled by degrees+flipud/fliplr,
            # color jitter by hsv_h/s/v, mosaic/mixup for scale & context diversity)
            hsv_h=aug_cfg["hsv_h"],
            hsv_s=aug_cfg["hsv_s"],
            hsv_v=aug_cfg["hsv_v"],
            degrees=aug_cfg["degrees"],
            translate=aug_cfg["translate"],
            scale=aug_cfg["scale"],
            shear=aug_cfg["shear"],
            flipud=aug_cfg["flipud"],
            fliplr=aug_cfg["fliplr"],
            mosaic=aug_cfg["mosaic"],
            mixup=aug_cfg["mixup"],
        )

    # Ultralytics writes best.pt/last.pt under run_dir/weights/; mirror them
    # into our standard checkpoint_dir so evaluation/inference/export scripts
    # follow the same --checkpoint convention as Module 1/2.
    weights_dir = run_dir / "weights"
    for name in ("best.pt", "last.pt"):
        src = weights_dir / name
        if src.exists():
            dst = checkpoint_dir / name
            shutil.copy2(src, dst)
            logger.info(f"Copied {src} -> {dst}")

    logger.info("Training complete.")
    logger.info(f"Full ultralytics run artifacts (plots, logs, weights): {run_dir}")


if __name__ == "__main__":
    main()
