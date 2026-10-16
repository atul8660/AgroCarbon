#!/usr/bin/env python3
"""
evaluate.py
------------
Evaluates a trained SegFormer checkpoint on the held-out test split.

Produces:
- Per-class IoU, mean IoU
- Dice score (per-class + mean)
- Boundary IoU (mask restricted to a thin band around the GT boundary —
  standard metric for segmentation edge quality, not just interior overlap)
- Precision, Recall, Pixel Accuracy
- Inference speed (ms/image, FPS)
- Visualization grid (image | GT mask | predicted mask overlay) for a sample
  of test images, saved as PNG

Usage:
    python evaluation/evaluate.py --config configs/config.yaml \
        --checkpoint outputs/checkpoints/best.pt
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parents[1]))

from data.dataset import TreeSegmentationDataset, get_eval_transforms, IMAGENET_MEAN, IMAGENET_STD
from models.model import build_model
from training.utils import get_device

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
)
logger = logging.getLogger("evaluate")


def load_config(config_path: str):
    return OmegaConf.load(config_path)


def compute_confusion_counts(pred: np.ndarray, gt: np.ndarray, num_classes: int) -> np.ndarray:
    mask = (gt >= 0) & (gt < num_classes)
    return np.bincount(
        num_classes * gt[mask].astype(int) + pred[mask].astype(int),
        minlength=num_classes ** 2,
    ).reshape(num_classes, num_classes)


def boundary_iou(pred: np.ndarray, gt: np.ndarray, dilation_px: int = 2) -> float:
    """Boundary IoU: restricts IoU computation to a thin band around the GT
    contour, so a mask can't score well just by getting the interior right
    while badly mismatching the edge (standard segmentation-quality metric)."""
    gt_u8 = (gt > 0).astype(np.uint8)
    pred_u8 = (pred > 0).astype(np.uint8)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilation_px + 1, 2 * dilation_px + 1))
    gt_boundary = cv2.dilate(gt_u8, kernel) - cv2.erode(gt_u8, kernel)
    band = gt_boundary > 0
    if band.sum() == 0:
        return float("nan")

    pred_band = pred_u8[band]
    gt_band = gt_u8[band]
    intersection = np.logical_and(pred_band, gt_band).sum()
    union = np.logical_or(pred_band, gt_band).sum()
    return float(intersection / union) if union > 0 else float("nan")


@torch.no_grad()
def evaluate_model(model, dataloader, device, num_classes: int, boundary_dilation_px: int):
    model.eval()
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    boundary_ious = []
    dice_sums = np.zeros(num_classes)
    dice_counts = np.zeros(num_classes)

    for images, masks, has_part in tqdm(dataloader, desc="Evaluating"):
        images_dev = images.to(device)
        logits = model(images_dev)
        preds = logits.argmax(dim=1).cpu().numpy()
        masks_np = masks.numpy()

        for b in range(images.size(0)):
            pred_b, mask_b = preds[b], masks_np[b]
            if not has_part[b]:
                pred_b = (pred_b > 0).astype(np.int64)
                mask_b = (mask_b > 0).astype(np.int64)
                confusion[:2, :2] += compute_confusion_counts(pred_b, mask_b, 2)
            else:
                confusion += compute_confusion_counts(pred_b, mask_b, num_classes)

            biou = boundary_iou(pred_b, mask_b, boundary_dilation_px)
            if not np.isnan(biou):
                boundary_ious.append(biou)

            eval_classes = num_classes if has_part[b] else 2
            for c in range(eval_classes):
                pred_c = (pred_b == c)
                gt_c = (mask_b == c)
                intersection = np.logical_and(pred_c, gt_c).sum()
                denom = pred_c.sum() + gt_c.sum()
                if denom > 0:
                    dice_sums[c] += (2 * intersection) / denom
                    dice_counts[c] += 1

    tp = np.diag(confusion)
    fp = confusion.sum(axis=0) - tp
    fn = confusion.sum(axis=1) - tp

    iou_per_class = tp / np.maximum(tp + fp + fn, 1)
    precision_per_class = tp / np.maximum(tp + fp, 1)
    recall_per_class = tp / np.maximum(tp + fn, 1)
    dice_per_class = dice_sums / np.maximum(dice_counts, 1)
    pixel_accuracy = tp.sum() / max(confusion.sum(), 1)

    return {
        "iou_per_class": iou_per_class.tolist(),
        "mean_iou": float(np.nanmean(iou_per_class[1:])),  # foreground classes only
        "dice_per_class": dice_per_class.tolist(),
        "mean_dice": float(np.nanmean(dice_per_class[1:])),
        "boundary_iou": float(np.mean(boundary_ious)) if boundary_ious else float("nan"),
        "precision_per_class": precision_per_class.tolist(),
        "recall_per_class": recall_per_class.tolist(),
        "pixel_accuracy": float(pixel_accuracy),
        "confusion_matrix": confusion.tolist(),
    }


def measure_inference_speed(model, dataloader, device, num_batches: int = 20) -> dict:
    model.eval()
    timings = []
    it = iter(dataloader)
    for _ in range(min(num_batches, len(dataloader))):
        try:
            images, _, _ = next(it)
        except StopIteration:
            break
        images = images.to(device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            model(images)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        timings.append(elapsed / images.size(0))

    avg_ms = float(np.mean(timings)) * 1000 if timings else 0.0
    fps = 1000.0 / avg_ms if avg_ms > 0 else 0.0
    return {"avg_latency_ms_per_image": avg_ms, "fps": fps}


def denormalize(image_tensor: torch.Tensor) -> np.ndarray:
    img = image_tensor.cpu().numpy().transpose(1, 2, 0)
    img = img * np.array(IMAGENET_STD) + np.array(IMAGENET_MEAN)
    return np.clip(img * 255, 0, 255).astype(np.uint8)


@torch.no_grad()
def save_visualizations(model, dataset, device, out_path: Path, num_samples: int = 6):
    fig, axes = plt.subplots(num_samples, 3, figsize=(9, 3 * num_samples))
    indices = np.linspace(0, len(dataset) - 1, num_samples, dtype=int)

    for row, idx in enumerate(indices):
        image, mask, _ = dataset[idx]
        logits = model(image.unsqueeze(0).to(device))
        pred = logits.argmax(dim=1).squeeze(0).cpu().numpy()

        img_np = denormalize(image)
        axes[row, 0].imshow(img_np)
        axes[row, 0].set_title("Image")
        axes[row, 1].imshow(mask.numpy(), cmap="viridis", vmin=0, vmax=2)
        axes[row, 1].set_title("Ground Truth")
        axes[row, 2].imshow(img_np)
        axes[row, 2].imshow(pred, cmap="viridis", alpha=0.5, vmin=0, vmax=2)
        axes[row, 2].set_title("Prediction Overlay")
        for ax in axes[row]:
            ax.axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()
    logger.info(f"Saved visualization grid: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--split", type=str, default="test", choices=["val", "test"])
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = get_device(cfg.project.device)
    num_classes = cfg.model.segformer.num_classes

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = build_model(cfg).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    logger.info(f"Loaded checkpoint from epoch {checkpoint.get('epoch')} "
                f"(best_metric={checkpoint.get('best_metric'):.4f})")

    dataset = TreeSegmentationDataset(
        f"{cfg.paths.splits_dir}/{args.split}.csv",
        transforms=get_eval_transforms(cfg.data.image_size),
    )
    dataloader = DataLoader(dataset, batch_size=cfg.train.batch_size, shuffle=False,
                             num_workers=cfg.data.num_workers, pin_memory=True)

    plots_dir = Path(cfg.paths.plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path(cfg.paths.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    metrics = evaluate_model(model, dataloader, device, num_classes,
                              cfg.evaluation.boundary_iou_dilation_px)
    speed = measure_inference_speed(model, dataloader, device)

    logger.info("=" * 70)
    logger.info(f"Mean IoU (foreground):  {metrics['mean_iou']:.4f}")
    logger.info(f"Per-class IoU:          {['%.3f' % v for v in metrics['iou_per_class']]}")
    logger.info(f"Mean Dice (foreground): {metrics['mean_dice']:.4f}")
    logger.info(f"Boundary IoU:           {metrics['boundary_iou']:.4f}")
    logger.info(f"Pixel Accuracy:         {metrics['pixel_accuracy']:.4f}")
    logger.info(f"Precision per class:    {['%.3f' % v for v in metrics['precision_per_class']]}")
    logger.info(f"Recall per class:       {['%.3f' % v for v in metrics['recall_per_class']]}")
    logger.info(f"Inference speed:        {speed['avg_latency_ms_per_image']:.2f} ms/image "
                f"({speed['fps']:.1f} FPS)")
    logger.info("=" * 70)

    save_visualizations(model, dataset, device, plots_dir / f"{args.split}_visualization.png")

    metrics_summary = {"split": args.split, **metrics, "inference_speed": speed}
    metrics_path = log_dir / f"{args.split}_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics_summary, f, indent=2)
    logger.info(f"Saved metrics summary: {metrics_path}")


if __name__ == "__main__":
    main()
