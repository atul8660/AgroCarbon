"""
engine.py
----------
Core train_one_epoch / validate loops for the 3-class (background/canopy/
trunk) SegFormer model.

Loss handling for mixed supervision: samples flagged has_part_supervision=False
(no real trunk ground truth — see data/prepare_dataset.py) would corrupt
training if scored against a 3-class target where "trunk" is silently absent.
For those samples we collapse the model's canopy+trunk logits into a single
foreground logit via log-sum-exp and compute a 2-class (background/foreground)
loss instead — the model is only ever supervised on the canopy/trunk
distinction where that distinction is actually known to be correct.

Validation mean IoU is computed per-class, but trunk IoU is only accumulated
over part-supervised validation samples, for the same reason.
"""

import logging

import torch
import torch.nn.functional as F
from tqdm import tqdm

from training.utils import AverageMeter

logger = logging.getLogger("engine")


def dice_loss_multiclass(logits: torch.Tensor, targets: torch.Tensor, num_classes: int,
                          eps: float = 1e-6) -> torch.Tensor:
    probs = F.softmax(logits, dim=1)
    targets_onehot = F.one_hot(targets, num_classes=num_classes).permute(0, 3, 1, 2).float()
    dims = (0, 2, 3)
    intersection = (probs * targets_onehot).sum(dims)
    union = probs.sum(dims) + targets_onehot.sum(dims)
    dice_per_class = (2 * intersection + eps) / (union + eps)
    return 1 - dice_per_class.mean()


def compute_loss(logits: torch.Tensor, masks: torch.Tensor, has_part_supervision: torch.Tensor,
                  class_weights: torch.Tensor, num_classes: int) -> torch.Tensor:
    device = logits.device
    losses = []

    part_idx = has_part_supervision
    if part_idx.any():
        ce = F.cross_entropy(logits[part_idx], masks[part_idx], weight=class_weights)
        dice = dice_loss_multiclass(logits[part_idx], masks[part_idx], num_classes)
        losses.append(ce + dice)

    binary_idx = ~has_part_supervision
    if binary_idx.any():
        l = logits[binary_idx]
        bg_logit = l[:, 0:1]
        fg_logit = torch.logsumexp(l[:, 1:num_classes], dim=1, keepdim=True)
        collapsed_logits = torch.cat([bg_logit, fg_logit], dim=1)
        target = (masks[binary_idx] > 0).long()
        collapsed_weights = torch.stack([
            class_weights[0], class_weights[1:num_classes].mean()
        ]).to(device)
        ce2 = F.cross_entropy(collapsed_logits, target, weight=collapsed_weights)
        losses.append(ce2)

    if not losses:
        return torch.tensor(0.0, device=device, requires_grad=True)
    return sum(losses) / len(losses)


@torch.no_grad()
def compute_iou_per_class(preds: torch.Tensor, masks: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Returns a [num_classes] tensor of per-class IoU for this batch (NaN
    for classes absent from both pred and target, to be nan-safe-averaged
    by the caller)."""
    ious = torch.zeros(num_classes)
    for c in range(num_classes):
        pred_c = (preds == c)
        mask_c = (masks == c)
        intersection = (pred_c & mask_c).sum().float()
        union = (pred_c | mask_c).sum().float()
        ious[c] = intersection / union if union > 0 else float("nan")
    return ious


def train_one_epoch(model, dataloader, optimizer, scaler, device, epoch: int,
                     class_weights: torch.Tensor, num_classes: int,
                     use_amp: bool = True, grad_clip_norm: float = 1.0, tb_writer=None):
    model.train()
    loss_meter = AverageMeter()

    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [train]", leave=False)
    for step, (images, masks, has_part) in enumerate(pbar):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        has_part = has_part.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits = model(images)
            loss = compute_loss(logits, masks, has_part, class_weights, num_classes)

        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()

        loss_meter.update(loss.item(), n=images.size(0))
        pbar.set_postfix(loss=f"{loss_meter.avg:.4f}")

        if tb_writer is not None:
            global_step = (epoch - 1) * len(dataloader) + step
            tb_writer.add_scalar("train/batch_loss", loss.item(), global_step)

    return {"loss": loss_meter.avg}


@torch.no_grad()
def validate(model, dataloader, device, epoch: int, class_weights: torch.Tensor,
             num_classes: int, use_amp: bool = True):
    model.eval()
    loss_meter = AverageMeter()
    iou_sums = torch.zeros(num_classes)
    iou_counts = torch.zeros(num_classes)

    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [val]", leave=False)
    for images, masks, has_part in pbar:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        has_part_device = has_part.to(device, non_blocking=True)

        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits = model(images)
            loss = compute_loss(logits, masks, has_part_device, class_weights, num_classes)
        loss_meter.update(loss.item(), n=images.size(0))

        preds = logits.argmax(dim=1)
        for b in range(images.size(0)):
            sample_num_classes = num_classes if has_part[b] else 2
            pred_b, mask_b = preds[b], masks[b]
            if not has_part[b]:
                # Evaluate background-vs-foreground only for non-part-supervised samples
                pred_b = (pred_b > 0).long()
                mask_b = (mask_b > 0).long()
            ious = compute_iou_per_class(pred_b, mask_b, sample_num_classes)
            for c in range(sample_num_classes):
                if not torch.isnan(ious[c]):
                    # class 1 in the 2-class collapsed case maps to "any foreground",
                    # which we attribute to canopy (index 1) for aggregate reporting
                    target_class = c if has_part[b] else (0 if c == 0 else 1)
                    iou_sums[target_class] += ious[c]
                    iou_counts[target_class] += 1

        pbar.set_postfix(loss=f"{loss_meter.avg:.4f}")

    per_class_iou = iou_sums / iou_counts.clamp(min=1)
    mean_iou_foreground = per_class_iou[1:].mean().item()  # canopy + trunk, excludes background

    return {
        "loss": loss_meter.avg,
        "mean_iou": mean_iou_foreground,
        "per_class_iou": per_class_iou.tolist(),
    }
