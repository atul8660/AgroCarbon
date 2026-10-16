#!/usr/bin/env python3
"""
evaluate.py
------------
Evaluates a trained checkpoint on the held-out test split.

Produces:
- Overall accuracy
- Per-class precision / recall / F1 (sklearn classification_report)
- Macro and weighted F1
- Confusion matrix (raw counts + normalized, saved as PNG + CSV)
- JSON metrics summary saved to outputs/logs/test_metrics.json

Usage:
    python evaluation/evaluate.py --config configs/config.yaml \
        --checkpoint outputs/checkpoints/best.pt
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import yaml
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parents[1]))

from data.dataset import TreeSpeciesDataset, get_eval_transforms
from models.model import build_model
from training.utils import get_device

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
)
logger = logging.getLogger("evaluate")


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


@torch.no_grad()
def run_inference(model, dataloader, device):
    model.eval()
    all_preds, all_labels, all_confidences = [], [], []

    for images, labels in tqdm(dataloader, desc="Evaluating on test set"):
        images = images.to(device, non_blocking=True)
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)
        confidences, preds = probs.max(dim=1)

        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.tolist())
        all_confidences.extend(confidences.cpu().tolist())

    return np.array(all_preds), np.array(all_labels), np.array(all_confidences)


def plot_confusion_matrix(cm: np.ndarray, class_names: list, out_path: Path,
                           normalize: bool = False):
    if normalize:
        cm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-9)
        fmt = ".2f"
        title = "Normalized Confusion Matrix"
    else:
        fmt = "d"
        title = "Confusion Matrix (counts)"

    fig_size = max(8, 0.4 * len(class_names))
    plt.figure(figsize=(fig_size, fig_size))
    sns.heatmap(
        cm, annot=len(class_names) <= 40, fmt=fmt, cmap="Blues",
        xticklabels=class_names, yticklabels=class_names, cbar=True,
    )
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    logger.info(f"Saved confusion matrix plot: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--split", type=str, default="test", choices=["val", "test"])
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = get_device(cfg["project"]["device"])

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    label_map = checkpoint["label_map"]  # {label_int: species_name}
    num_classes = checkpoint["num_classes"]
    class_names = [label_map[i] for i in range(num_classes)]

    model = build_model(cfg, num_classes).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    logger.info(f"Loaded checkpoint from epoch {checkpoint.get('epoch')} "
                f"(best_metric={checkpoint.get('best_metric'):.4f})")

    splits_dir = cfg["paths"]["splits_dir"]
    image_size = cfg["data"]["image_size"]
    test_ds = TreeSpeciesDataset(
        f"{splits_dir}/{args.split}.csv", transforms=get_eval_transforms(image_size)
    )
    test_loader = DataLoader(
        test_ds, batch_size=cfg["train"]["batch_size"], shuffle=False,
        num_workers=cfg["data"]["num_workers"], pin_memory=True,
    )
    logger.info(f"Evaluating on {args.split} split: {len(test_ds)} images")

    preds, labels, confidences = run_inference(model, test_loader, device)

    accuracy = accuracy_score(labels, preds)
    precision_macro = precision_score(labels, preds, average="macro", zero_division=0)
    recall_macro = recall_score(labels, preds, average="macro", zero_division=0)
    f1_macro = f1_score(labels, preds, average="macro", zero_division=0)
    f1_weighted = f1_score(labels, preds, average="weighted", zero_division=0)

    report = classification_report(
        labels, preds, target_names=class_names, zero_division=0, output_dict=True
    )
    report_text = classification_report(
        labels, preds, target_names=class_names, zero_division=0
    )

    logger.info("=" * 70)
    logger.info(f"Accuracy:            {accuracy:.4f}")
    logger.info(f"Precision (macro):   {precision_macro:.4f}")
    logger.info(f"Recall (macro):      {recall_macro:.4f}")
    logger.info(f"F1 (macro):          {f1_macro:.4f}")
    logger.info(f"F1 (weighted):       {f1_weighted:.4f}")
    logger.info(f"Mean prediction confidence: {confidences.mean():.4f}")
    logger.info("=" * 70)
    print(report_text)

    plots_dir = Path(cfg["paths"]["plots_dir"])
    plots_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path(cfg["paths"]["log_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)

    cm = confusion_matrix(labels, preds, labels=list(range(num_classes)))
    np.savetxt(log_dir / f"{args.split}_confusion_matrix.csv", cm, fmt="%d", delimiter=",")
    plot_confusion_matrix(cm, class_names, plots_dir / f"{args.split}_confusion_matrix.png",
                           normalize=False)
    plot_confusion_matrix(cm, class_names, plots_dir / f"{args.split}_confusion_matrix_normalized.png",
                           normalize=True)

    metrics_summary = {
        "split": args.split,
        "num_samples": len(labels),
        "accuracy": accuracy,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "mean_confidence": float(confidences.mean()),
        "per_class_report": report,
    }
    with open(log_dir / f"{args.split}_metrics.json", "w") as f:
        json.dump(metrics_summary, f, indent=2)
    logger.info(f"Saved metrics summary: {log_dir / f'{args.split}_metrics.json'}")


if __name__ == "__main__":
    main()
