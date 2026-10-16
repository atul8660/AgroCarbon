"""
utils.py
---------
Shared training utilities: reproducibility seeding, checkpoint save/load
(full training state for exact resume), and running-average metric tracking.
"""

import logging
import random
from pathlib import Path

import numpy as np
import torch

logger = logging.getLogger("train_utils")


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False  # keep True only if reproducibility > speed
    torch.backends.cudnn.benchmark = True


def get_device(preferred: str = "cuda") -> torch.device:
    if preferred == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if preferred == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class AverageMeter:
    """Tracks running mean of a scalar metric (e.g. loss) across a training epoch."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0.0
        self.count = 0
        self.avg = 0.0

    def update(self, value: float, n: int = 1):
        self.sum += value * n
        self.count += n
        self.avg = self.sum / max(self.count, 1)


def save_checkpoint(state: dict, checkpoint_dir: str, filename: str) -> None:
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    path = Path(checkpoint_dir) / filename
    torch.save(state, path)
    logger.info(f"Checkpoint saved: {path}")


def load_checkpoint(checkpoint_path: str, model, optimizer=None, scheduler=None,
                     scaler=None, device=None) -> dict:
    """Loads a checkpoint saved by save_checkpoint and restores full training state.

    Returns the checkpoint dict so the caller can read back epoch/best-metric.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device or "cpu", weights_only=False)

    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    if scaler is not None and checkpoint.get("scaler_state_dict") is not None:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])

    logger.info(
        f"Resumed from {checkpoint_path} "
        f"(epoch={checkpoint.get('epoch')}, best_metric={checkpoint.get('best_metric')})"
    )
    return checkpoint
