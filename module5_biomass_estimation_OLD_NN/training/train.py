import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import logging
from tqdm import tqdm
from omegaconf import OmegaConf

from models.model import BiomassEstimator
from data.dataset import BiomassDataset

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_loss_fn(cfg):
    if cfg.loss.type == "huber":
        return nn.HuberLoss(delta=cfg.loss.huber_delta)
    elif cfg.loss.type == "mae":
        return nn.L1Loss()
    else:
        return nn.MSELoss()

def train_one_epoch(model, loader, optimizer, loss_fn, device, scaler, cfg):
    model.train()
    total_loss = 0
    total_mae = 0
    
    pbar = tqdm(loader, desc="Training")
    for batch in pbar:
        images = batch['image'].to(device)
        handcrafted = batch['handcrafted'].to(device)
        species = batch['species'].to(device)
        health = batch['health'].to(device)
        labels = batch['label'].to(device)
        
        optimizer.zero_grad()
        
        with torch.cuda.amp.autocast(enabled=cfg.train.use_amp):
            preds = model(images, handcrafted, species, health)
            loss = loss_fn(preds, labels)
            
        scaler.scale(loss).backward()
        
        if cfg.train.grad_clip_norm > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip_norm)
            
        scaler.step(optimizer)
        scaler.update()
        
        mae = torch.abs(preds - labels).mean().item()
        total_loss += loss.item()
        total_mae += mae
        
        pbar.set_postfix({"Loss": loss.item(), "MAE": mae})
        
    return total_loss / len(loader), total_mae / len(loader)

@torch.no_grad()
def validate(model, loader, loss_fn, device, cfg):
    model.eval()
    total_loss = 0
    total_mae = 0
    
    pbar = tqdm(loader, desc="Validating")
    for batch in pbar:
        images = batch['image'].to(device)
        handcrafted = batch['handcrafted'].to(device)
        species = batch['species'].to(device)
        health = batch['health'].to(device)
        labels = batch['label'].to(device)
        
        with torch.cuda.amp.autocast(enabled=cfg.train.use_amp):
            preds = model(images, handcrafted, species, health)
            loss = loss_fn(preds, labels)
            
        mae = torch.abs(preds - labels).mean().item()
        total_loss += loss.item()
        total_mae += mae
        
    return total_loss / len(loader), total_mae / len(loader)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()
    
    cfg = OmegaConf.load(args.config)
    os.makedirs(cfg.paths.checkpoint_dir, exist_ok=True)
    
    device = torch.device(cfg.project.device if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    
    model = BiomassEstimator(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.train.use_amp)
    loss_fn = get_loss_fn(cfg)
    
    # Placeholders for actual data paths
    train_dataset = BiomassDataset(f"{cfg.paths.splits_dir}/train.csv", cfg.paths.raw_data_dir, mode="train", image_size=cfg.data.image_size)
    val_dataset = BiomassDataset(f"{cfg.paths.splits_dir}/val.csv", cfg.paths.raw_data_dir, mode="val", image_size=cfg.data.image_size)
    
    train_loader = DataLoader(train_dataset, batch_size=cfg.train.batch_size, shuffle=True, num_workers=cfg.data.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=cfg.train.batch_size * 2, shuffle=False, num_workers=cfg.data.num_workers)
    
    best_mae = float('inf')
    early_stop_count = 0
    
    for epoch in range(1, cfg.train.epochs + 1):
        logging.info(f"Epoch {epoch}/{cfg.train.epochs}")
        train_loss, train_mae = train_one_epoch(model, train_loader, optimizer, loss_fn, device, scaler, cfg)
        val_loss, val_mae = validate(model, val_loader, loss_fn, device, cfg)
        
        scheduler.step(val_mae)
        
        logging.info(f"Train Loss: {train_loss:.4f} | Train MAE: {train_mae:.2f} kg")
        logging.info(f"Val Loss: {val_loss:.4f} | Val MAE: {val_mae:.2f} kg")
        
        if val_mae < best_mae:
            best_mae = val_mae
            early_stop_count = 0
            save_path = os.path.join(cfg.paths.checkpoint_dir, "best.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_mae': val_mae
            }, save_path)
            logging.info(f"Saved new best model to {save_path} (MAE: {val_mae:.2f} kg)")
        else:
            early_stop_count += 1
            
        if early_stop_count >= cfg.train.early_stopping_patience:
            logging.info("Early stopping triggered.")
            break

if __name__ == "__main__":
    main()
