"""
engine.py
----------
Core train_one_epoch / validate loops, with automatic mixed precision (AMP),
gradient clipping, macro-F1 tracking, and optional TensorBoard scalar logging
(writer is passed in from train.py; engine stays writer-agnostic if None).
"""

import logging

import torch
from sklearn.metrics import f1_score
from tqdm import tqdm

from training.utils import AverageMeter

logger = logging.getLogger("engine")


def train_one_epoch(model, dataloader, optimizer, criterion, scaler, device,
                     epoch: int, use_amp: bool = True, grad_clip_norm: float = 5.0,
                     tb_writer=None):
    model.train()
    loss_meter = AverageMeter()
    all_preds, all_labels = [], []

    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [train]", leave=False)
    for step, (images, labels) in enumerate(pbar):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=device.type, enabled=use_amp):
            outputs = model(images)
            loss = criterion(outputs, labels)

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
        preds = outputs.argmax(dim=1)
        all_preds.extend(preds.detach().cpu().tolist())
        all_labels.extend(labels.detach().cpu().tolist())

        pbar.set_postfix(loss=f"{loss_meter.avg:.4f}")

        if tb_writer is not None:
            global_step = (epoch - 1) * len(dataloader) + step
            tb_writer.add_scalar("train/batch_loss", loss.item(), global_step)

    train_acc = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
    train_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    return {"loss": loss_meter.avg, "accuracy": train_acc, "macro_f1": train_f1}


@torch.no_grad()
def validate(model, dataloader, criterion, device, epoch: int, use_amp: bool = True):
    model.eval()
    loss_meter = AverageMeter()
    all_preds, all_labels = [], []

    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [val]", leave=False)
    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.autocast(device_type=device.type, enabled=use_amp):
            outputs = model(images)
            loss = criterion(outputs, labels)

        loss_meter.update(loss.item(), n=images.size(0))
        preds = outputs.argmax(dim=1)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

        pbar.set_postfix(loss=f"{loss_meter.avg:.4f}")

    val_acc = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
    val_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    return {"loss": loss_meter.avg, "accuracy": val_acc, "macro_f1": val_f1}
