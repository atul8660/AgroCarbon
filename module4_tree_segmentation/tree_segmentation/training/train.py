#!/usr/bin/env python3
"""
train.py
---------
Main training entrypoint for the SegFormer-B5 tree segmentation model.

Features:
- Dataset + DataLoader with mixed-supervision-aware collation
- SegFormer-B5 fine-tuning from ADE20K-pretrained weights
- Combined cross-entropy + Dice loss, part-supervision-aware (see engine.py)
- Cosine LR schedule with linear warmup
- Automatic mixed precision (AMP)
- Early stopping on validation mean IoU (foreground classes)
- Full checkpoint save every epoch (`last.pt`) + on improvement (`best.pt`)
- `--resume <path>` restores model/optimizer/scheduler/scaler/epoch exactly
- TensorBoard scalar logging to outputs/tb_logs

Usage:
    python training/train.py --config configs/config.yaml
    python training/train.py --config configs/config.yaml --resume outputs/checkpoints/last.pt
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

sys.path.append(str(Path(__file__).resolve().parents[1]))

from data.dataset import TreeSegmentationDataset, get_train_transforms, get_eval_transforms
from models.model import build_model, count_parameters
from training.early_stopping import EarlyStopping
from training.engine import train_one_epoch, validate
from training.utils import set_seed, get_device, save_checkpoint, load_checkpoint

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
)
logger = logging.getLogger("train")


def load_config(config_path: str):
    return OmegaConf.load(config_path)


def build_dataloaders(cfg):
    splits_dir = cfg.paths.splits_dir
    image_size = cfg.data.image_size
    batch_size = cfg.train.batch_size
    num_workers = cfg.data.num_workers

    train_ds = TreeSegmentationDataset(
        f"{splits_dir}/train.csv", transforms=get_train_transforms(image_size)
    )
    val_ds = TreeSegmentationDataset(
        f"{splits_dir}/val.csv", transforms=get_eval_transforms(image_size)
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    return train_loader, val_loader, train_ds, val_ds


def build_scheduler(optimizer, warmup_epochs: int, total_epochs: int, steps_per_epoch: int):
    warmup_steps = warmup_epochs * steps_per_epoch
    total_steps = total_epochs * steps_per_epoch

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    return LambdaLR(optimizer, lr_lambda)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.project.seed)
    device = get_device(cfg.project.device)
    logger.info(f"Using device: {device}")

    train_loader, val_loader, train_ds, val_ds = build_dataloaders(cfg)
    num_classes = cfg.model.segformer.num_classes
    logger.info(f"Num classes: {num_classes} | Train: {len(train_ds)} | Val: {len(val_ds)}")

    model = build_model(cfg).to(device)
    logger.info(f"Model: SegFormer-B5 | Trainable params: {count_parameters(model):,}")

    class_weights = torch.tensor(list(cfg.train.class_weights), dtype=torch.float32).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay
    )
    scheduler = build_scheduler(
        optimizer, cfg.train.warmup_epochs, cfg.train.epochs, len(train_loader)
    )
    scaler = torch.amp.GradScaler(enabled=cfg.train.use_amp)
    early_stopper = EarlyStopping(patience=cfg.train.early_stopping_patience, mode="max")

    tb_log_dir = Path(cfg.paths.tb_log_dir)
    tb_log_dir.mkdir(parents=True, exist_ok=True)
    tb_writer = SummaryWriter(log_dir=str(tb_log_dir))

    start_epoch = 1
    best_metric = -np.inf
    history = []

    if args.resume:
        checkpoint = load_checkpoint(args.resume, model, optimizer, scheduler, scaler, device)
        start_epoch = checkpoint.get("epoch", 0) + 1
        best_metric = checkpoint.get("best_metric", -np.inf)
        early_stopper.best_score = best_metric
        history = checkpoint.get("history", [])
        logger.info(f"Resuming training from epoch {start_epoch}")

    checkpoint_dir = cfg.paths.checkpoint_dir
    log_dir = Path(cfg.paths.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(start_epoch, cfg.train.epochs + 1):
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, scaler, device, epoch,
            class_weights, num_classes, use_amp=cfg.train.use_amp,
            grad_clip_norm=cfg.train.grad_clip_norm, tb_writer=tb_writer,
        )
        val_metrics = validate(
            model, val_loader, device, epoch, class_weights, num_classes,
            use_amp=cfg.train.use_amp,
        )
        scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]
        logger.info(
            f"Epoch {epoch}/{cfg.train.epochs} | LR {current_lr:.2e} | "
            f"train_loss {train_metrics['loss']:.4f} | val_loss {val_metrics['loss']:.4f} | "
            f"val_mean_iou {val_metrics['mean_iou']:.4f} | "
            f"per_class_iou {['%.3f' % v for v in val_metrics['per_class_iou']]}"
        )

        tb_writer.add_scalar("train/loss", train_metrics["loss"], epoch)
        tb_writer.add_scalar("val/loss", val_metrics["loss"], epoch)
        tb_writer.add_scalar("val/mean_iou", val_metrics["mean_iou"], epoch)
        for c, iou in enumerate(val_metrics["per_class_iou"]):
            tb_writer.add_scalar(f"val/iou_class_{c}", iou, epoch)
        tb_writer.add_scalar("lr", current_lr, epoch)

        history.append({
            "epoch": epoch, "lr": current_lr,
            "train_loss": train_metrics["loss"],
            "val_loss": val_metrics["loss"],
            "val_mean_iou": val_metrics["mean_iou"],
            "val_per_class_iou": val_metrics["per_class_iou"],
        })
        with open(log_dir / "history.json", "w") as f:
            json.dump(history, f, indent=2)

        monitored_metric = val_metrics["mean_iou"]
        is_best = early_stopper.step(monitored_metric)
        if monitored_metric > best_metric:
            best_metric = monitored_metric

        checkpoint_state = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "best_metric": best_metric,
            "history": history,
            "num_classes": num_classes,
            "config": OmegaConf.to_container(cfg, resolve=True),
        }
        save_checkpoint(checkpoint_state, checkpoint_dir, "last.pt")
        if is_best:
            save_checkpoint(checkpoint_state, checkpoint_dir, "best.pt")
            logger.info(f"New best val_mean_iou: {best_metric:.4f} -> saved best.pt")

        if early_stopper.should_stop:
            logger.info(f"Early stopping triggered at epoch {epoch}.")
            break

    tb_writer.close()
    logger.info("Training complete.")


if __name__ == "__main__":
    main()
