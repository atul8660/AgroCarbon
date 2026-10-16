"""
utils.py
---------
Shared training utilities: reproducibility seeding and device resolution.
Checkpointing, AMP, early stopping, and LR scheduling for YOLOv11 are handled
internally by ultralytics' Trainer (configured via train.py), so this module
stays intentionally small — unlike Module 1/2's custom PyTorch loops, we're
not hand-rolling those mechanics here, only wiring config into them.
"""

import logging
import random

import numpy as np
import torch

logger = logging.getLogger("train_utils")


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(preferred: str = "cuda") -> str:
    """Returns a device string in the format ultralytics expects
    ('cpu', '0', '0,1', etc.)."""
    if preferred == "cpu":
        return "cpu"
    if torch.cuda.is_available():
        return "0"
    logger.warning("CUDA not available — falling back to CPU (training will be slow).")
    return "cpu"
