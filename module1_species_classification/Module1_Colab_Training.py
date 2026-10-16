#!/usr/bin/env python3
"""
=============================================================================
 Module 1 — Tree Species Classification (EfficientNet-B3)
 Google Colab Training Script
 
 ► Mount Google Drive
 ► Auto-find datasets & zip files from Module1_Traning folder
 ► Train with AMP, early stopping, cosine LR
 ► Save last.pt & best.pt checkpoints to Drive
 ► Print full training & testing metrics + confusion matrix
=============================================================================

HOW TO USE (paste each section into a separate Colab cell):
  Cell 1 → SETUP & INSTALL
  Cell 2 → MOUNT DRIVE & FIND DATA
  Cell 3 → PREPARE DATASET
  Cell 4 → TRAIN
  Cell 5 → TEST EVALUATION & METRICS
"""

# =====================================================================
# ██████  CELL 1: SETUP & INSTALL
# =====================================================================
# !pip install -q torch torchvision timm albumentations opencv-python-headless \
#     scikit-learn matplotlib seaborn pandas tqdm PyYAML Pillow

import os, sys, glob, shutil, json, random, logging, hashlib, warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import timm
import cv2
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score
)
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s")
logger = logging.getLogger("module1")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🖥️  Device: {device}")
if device.type == 'cuda':
    print(f"🎮  GPU: {torch.cuda.get_device_name(0)}")
    print(f"💾  VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")


# =====================================================================
# ██████  CELL 2: MOUNT DRIVE & FIND DATA
# =====================================================================
from google.colab import drive
drive.mount('/content/drive')

# --- Auto-find Module1_Traning folder ---
DRIVE_BASE = '/content/drive/MyDrive'
MODULE1_DIR = None

# Search for the folder (handles typos like Traning/Training)
for pattern in ['Module1_Tran*', 'module1_tran*', 'Module1_Train*']:
    found = glob.glob(os.path.join(DRIVE_BASE, '**', pattern), recursive=True)
    # Also check direct children
    found += glob.glob(os.path.join(DRIVE_BASE, pattern))
    if found:
        MODULE1_DIR = found[0]
        break

if MODULE1_DIR is None:
    # Fallback: search entire drive
    for root, dirs, files in os.walk(DRIVE_BASE):
        for d in dirs:
            if 'module1' in d.lower() and 'train' in d.lower():
                MODULE1_DIR = os.path.join(root, d)
                break
        if MODULE1_DIR:
            break

if MODULE1_DIR:
    print(f"✅ Found Module1 folder: {MODULE1_DIR}")
else:
    print("⚠️  Could not auto-find Module1_Traning folder. Listing drive root:")
    for item in os.listdir(DRIVE_BASE)[:30]:
        print(f"   {item}")
    MODULE1_DIR = input("📂 Enter the full path to your Module1 training folder: ").strip()

# --- List contents ---
print(f"\n📁 Contents of {MODULE1_DIR}:")
for item in sorted(os.listdir(MODULE1_DIR)):
    full = os.path.join(MODULE1_DIR, item)
    size = os.path.getsize(full) / (1024*1024) if os.path.isfile(full) else 0
    tag = f"[DIR]" if os.path.isdir(full) else f"[FILE {size:.1f} MB]"
    print(f"   {tag} {item}")

# --- Find and extract ZIP files ---
WORK_DIR = '/content/module1_data'
os.makedirs(WORK_DIR, exist_ok=True)

zip_files = glob.glob(os.path.join(MODULE1_DIR, '*.zip'))
print(f"\n📦 Found {len(zip_files)} zip files:")
for zf in zip_files:
    print(f"   → {os.path.basename(zf)}")
    extract_dir = os.path.join(WORK_DIR, Path(zf).stem)
    if not os.path.exists(extract_dir):
        os.system(f'unzip -q "{zf}" -d "{extract_dir}"')
        print(f"     Extracted to {extract_dir}")
    else:
        print(f"     Already extracted")

# --- Also find non-zip dataset directories ---
for item in os.listdir(MODULE1_DIR):
    full = os.path.join(MODULE1_DIR, item)
    if os.path.isdir(full) and not item.startswith('.'):
        dest = os.path.join(WORK_DIR, item)
        if not os.path.exists(dest):
            os.symlink(full, dest)
            print(f"🔗 Linked dataset dir: {item}")

# --- Auto-discover dataset directories (find folders with image subfolders) ---
DATASET_DIRS = []
for root, dirs, files in os.walk(WORK_DIR):
    # Check if this directory has subdirectories with images
    has_image_subdirs = False
    for d in dirs:
        subdir = os.path.join(root, d)
        imgs = glob.glob(os.path.join(subdir, '*.jpg')) + \
               glob.glob(os.path.join(subdir, '*.jpeg')) + \
               glob.glob(os.path.join(subdir, '*.png'))
        if len(imgs) > 5:
            has_image_subdirs = True
            break
    if has_image_subdirs:
        DATASET_DIRS.append(root)

print(f"\n🌳 Found {len(DATASET_DIRS)} dataset directories:")
for dd in DATASET_DIRS:
    classes = [d for d in os.listdir(dd) if os.path.isdir(os.path.join(dd, d))]
    print(f"   {dd} → {len(classes)} classes")


# =====================================================================
# ██████  CELL 3: PREPARE DATASET (merge + split)
# =====================================================================
PROCESSED_DIR = '/content/processed_data'
SPLITS_DIR = '/content/splits'
CHECKPOINT_DIR = '/content/checkpoints'
PLOTS_DIR = '/content/plots'
LOGS_DIR = '/content/logs'

for d in [PROCESSED_DIR, SPLITS_DIR, CHECKPOINT_DIR, PLOTS_DIR, LOGS_DIR]:
    os.makedirs(d, exist_ok=True)

# --- Merge all datasets into unified structure ---
VALID_EXT = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
MAX_EDGE = 800
seen_hashes = set()
class_counts = {}

def file_hash(path):
    h = hashlib.md5()
    with open(path, 'rb') as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

print("🔄 Merging datasets into unified structure...")
for dataset_dir in DATASET_DIRS:
    for class_dir in sorted(Path(dataset_dir).iterdir()):
        if not class_dir.is_dir() or class_dir.name.startswith('.'):
            continue
        species = class_dir.name.replace(' ', '_').replace('-', '_')
        out_dir = Path(PROCESSED_DIR) / species
        out_dir.mkdir(parents=True, exist_ok=True)
        
        existing = len(list(out_dir.glob('*.jpg')))
        idx = existing
        count = 0
        
        for img_path in class_dir.rglob('*'):
            if img_path.suffix.lower() not in VALID_EXT or not img_path.is_file():
                continue
            try:
                h = file_hash(img_path)
                if h in seen_hashes:
                    continue
                seen_hashes.add(h)
                
                with Image.open(img_path) as img:
                    img = img.convert('RGB')
                    w, h_px = img.size
                    if max(w, h_px) > MAX_EDGE:
                        scale = MAX_EDGE / max(w, h_px)
                        img = img.resize((int(w*scale), int(h_px*scale)), Image.LANCZOS)
                    dst = out_dir / f"{species}_{idx:06d}.jpg"
                    img.save(dst, format='JPEG', quality=92)
                    idx += 1
                    count += 1
            except Exception:
                continue
        
        class_counts[species] = class_counts.get(species, 0) + count

# Summary
print(f"\n{'='*60}")
print(f"📊 Dataset Summary:")
print(f"{'='*60}")
total = 0
for species in sorted(class_counts.keys()):
    cnt = len(list(Path(PROCESSED_DIR, species).glob('*.jpg')))
    class_counts[species] = cnt
    total += cnt
    status = "✅" if cnt >= 20 else "⚠️ LOW"
    print(f"   {status} {species:25s} → {cnt:5d} images")
print(f"{'='*60}")
print(f"   Total: {total} images across {len(class_counts)} classes")

# --- Stratified Split ---
records = []
for class_dir in sorted(Path(PROCESSED_DIR).iterdir()):
    if class_dir.is_dir():
        for img in class_dir.glob('*.jpg'):
            records.append({'filepath': str(img), 'species': class_dir.name})

df = pd.DataFrame(records)
MIN_PER_CLASS = 3
counts = df['species'].value_counts()
valid_classes = counts[counts >= MIN_PER_CLASS].index
dropped = set(counts.index) - set(valid_classes)
if dropped:
    print(f"⚠️  Dropping classes with <{MIN_PER_CLASS} images: {sorted(dropped)}")
df = df[df['species'].isin(valid_classes)].reset_index(drop=True)

train_val_df, test_df = train_test_split(df, test_size=0.15, stratify=df['species'], random_state=42)
rel_val = 0.15 / 0.85
train_df, val_df = train_test_split(train_val_df, test_size=rel_val, stratify=train_val_df['species'], random_state=42)

classes = sorted(df['species'].unique())
label_map = {cls: idx for idx, cls in enumerate(classes)}
for split_df in (train_df, val_df, test_df):
    split_df['label'] = split_df['species'].map(label_map)

train_df.to_csv(f'{SPLITS_DIR}/train.csv', index=False)
val_df.to_csv(f'{SPLITS_DIR}/val.csv', index=False)
test_df.to_csv(f'{SPLITS_DIR}/test.csv', index=False)
pd.DataFrame({'label': list(label_map.values()), 'species': list(label_map.keys())}).sort_values('label').to_csv(f'{SPLITS_DIR}/label_map.csv', index=False)

NUM_CLASSES = len(classes)
print(f"\n✅ Splits created: Train={len(train_df)} | Val={len(val_df)} | Test={len(test_df)} | Classes={NUM_CLASSES}")


# =====================================================================
# ██████  CELL 4: TRAIN
# =====================================================================

# --- Config ---
CONFIG = {
    'image_size': 300,
    'backbone': 'efficientnet_b3',
    'pretrained': True,
    'drop_rate': 0.3,
    'drop_path_rate': 0.2,
    'epochs': 60,
    'batch_size': 32,
    'lr': 3e-4,
    'weight_decay': 1e-4,
    'label_smoothing': 0.1,
    'warmup_epochs': 3,
    'early_stopping_patience': 8,
    'use_amp': True,
    'grad_clip_norm': 5.0,
    'num_workers': 2,
}

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# --- Transforms ---
train_transforms = A.Compose([
    A.RandomResizedCrop(size=(CONFIG['image_size'], CONFIG['image_size']), scale=(0.7, 1.0)),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.3),
    A.Rotate(limit=180, border_mode=cv2.BORDER_REFLECT_101, p=0.7),
    A.OneOf([
        A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
        A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=25, val_shift_limit=15),
    ], p=0.7),
    A.OneOf([A.GaussianBlur(blur_limit=(3,5)), A.MotionBlur(blur_limit=5)], p=0.2),
    A.CoarseDropout(num_holes_range=(1,4), hole_height_range=(0.05,0.15), hole_width_range=(0.05,0.15), p=0.3),
    A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ToTensorV2(),
])

eval_transforms = A.Compose([
    A.Resize(height=CONFIG['image_size'], width=CONFIG['image_size']),
    A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ToTensorV2(),
])

# --- Dataset ---
class TreeDataset(Dataset):
    def __init__(self, csv_path, transforms=None):
        self.df = pd.read_csv(csv_path)
        self.transforms = transforms
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = cv2.imread(row['filepath'])
        if image is None:
            image = np.zeros((300,300,3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transforms:
            image = self.transforms(image=image)['image']
        return image, int(row['label'])
    def get_class_counts(self):
        n = self.df['label'].nunique()
        counts = np.zeros(n, dtype=np.int64)
        for lbl, cnt in self.df['label'].value_counts().items():
            counts[lbl] = cnt
        return counts

# --- Model ---
def build_model(num_classes):
    try:
        backbone = timm.create_model(CONFIG['backbone'], pretrained=CONFIG['pretrained'],
            num_classes=num_classes, drop_rate=CONFIG['drop_rate'], drop_path_rate=CONFIG['drop_path_rate'])
    except TypeError:
        backbone = timm.create_model(CONFIG['backbone'], pretrained=CONFIG['pretrained'],
            num_classes=num_classes, drop_rate=CONFIG['drop_rate'])
    return backbone

# --- DataLoaders ---
train_ds = TreeDataset(f'{SPLITS_DIR}/train.csv', train_transforms)
val_ds = TreeDataset(f'{SPLITS_DIR}/val.csv', eval_transforms)

class_counts_arr = train_ds.get_class_counts()
class_weights = 1.0 / np.maximum(class_counts_arr, 1)
sample_weights = train_ds.df['label'].map(lambda l: class_weights[l]).values
sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

train_loader = DataLoader(train_ds, batch_size=CONFIG['batch_size'], sampler=sampler,
                          num_workers=CONFIG['num_workers'], pin_memory=True, drop_last=True)
val_loader = DataLoader(val_ds, batch_size=CONFIG['batch_size'], shuffle=False,
                        num_workers=CONFIG['num_workers'], pin_memory=True)

print(f"📦 Train: {len(train_ds)} | Val: {len(val_ds)} | Classes: {NUM_CLASSES}")

# --- Build model, optimizer, scheduler ---
model = build_model(NUM_CLASSES).to(device)
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"🧠 Model: {CONFIG['backbone']} | Params: {total_params:,}")

criterion = nn.CrossEntropyLoss(label_smoothing=CONFIG['label_smoothing'])
optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG['lr'], weight_decay=CONFIG['weight_decay'])

warmup_steps = CONFIG['warmup_epochs'] * len(train_loader)
total_steps = CONFIG['epochs'] * len(train_loader)
def lr_lambda(step):
    if step < warmup_steps:
        return step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * (1.0 + np.cos(np.pi * progress))
scheduler = LambdaLR(optimizer, lr_lambda)
scaler = torch.amp.GradScaler(enabled=CONFIG['use_amp'])

# --- Training Loop ---
best_metric = -np.inf
patience_counter = 0
history = []

print(f"\n{'='*70}")
print(f"🚀 Starting Training — {CONFIG['epochs']} epochs")
print(f"{'='*70}\n")

for epoch in range(1, CONFIG['epochs'] + 1):
    # ---- Train ----
    model.train()
    train_loss_sum, train_n = 0.0, 0
    train_preds, train_labels = [], []
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{CONFIG['epochs']} [train]", leave=False)
    for images, labels in pbar:
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=CONFIG['use_amp']):
            outputs = model(images)
            loss = criterion(outputs, labels)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG['grad_clip_norm'])
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        
        train_loss_sum += loss.item() * images.size(0)
        train_n += images.size(0)
        train_preds.extend(outputs.argmax(1).cpu().tolist())
        train_labels.extend(labels.cpu().tolist())
        pbar.set_postfix(loss=f"{train_loss_sum/train_n:.4f}")
    
    train_loss = train_loss_sum / train_n
    train_acc = sum(p==l for p,l in zip(train_preds, train_labels)) / len(train_labels)
    train_f1 = f1_score(train_labels, train_preds, average='macro', zero_division=0)
    
    # ---- Validate ----
    model.eval()
    val_loss_sum, val_n = 0.0, 0
    val_preds, val_labels = [], []
    
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc=f"Epoch {epoch}/{CONFIG['epochs']} [val]", leave=False):
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, enabled=CONFIG['use_amp']):
                outputs = model(images)
                loss = criterion(outputs, labels)
            val_loss_sum += loss.item() * images.size(0)
            val_n += images.size(0)
            val_preds.extend(outputs.argmax(1).cpu().tolist())
            val_labels.extend(labels.cpu().tolist())
    
    val_loss = val_loss_sum / val_n
    val_acc = sum(p==l for p,l in zip(val_preds, val_labels)) / len(val_labels)
    val_f1 = f1_score(val_labels, val_preds, average='macro', zero_division=0)
    
    lr_now = optimizer.param_groups[0]['lr']
    print(f"Epoch {epoch:3d}/{CONFIG['epochs']} | LR {lr_now:.2e} | "
          f"Train Loss {train_loss:.4f} Acc {train_acc:.4f} F1 {train_f1:.4f} | "
          f"Val Loss {val_loss:.4f} Acc {val_acc:.4f} F1 {val_f1:.4f}")
    
    history.append({
        'epoch': epoch, 'lr': lr_now,
        'train_loss': train_loss, 'train_acc': train_acc, 'train_f1': train_f1,
        'val_loss': val_loss, 'val_acc': val_acc, 'val_f1': val_f1,
    })
    with open(f'{LOGS_DIR}/history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    # ---- Checkpoint ----
    ckpt = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'best_metric': best_metric,
        'history': history,
        'num_classes': NUM_CLASSES,
        'label_map': {idx: cls for cls, idx in label_map.items()},
        'config': CONFIG,
    }
    torch.save(ckpt, f'{CHECKPOINT_DIR}/last.pt')
    
    if val_f1 > best_metric + 1e-4:
        best_metric = val_f1
        patience_counter = 0
        torch.save(ckpt, f'{CHECKPOINT_DIR}/best.pt')
        print(f"   ⭐ New best val_F1: {best_metric:.4f} → saved best.pt")
    else:
        patience_counter += 1
        print(f"   ⏳ No improvement {patience_counter}/{CONFIG['early_stopping_patience']}")
        if patience_counter >= CONFIG['early_stopping_patience']:
            print(f"   🛑 Early stopping at epoch {epoch}")
            break

print(f"\n✅ Training complete! Best val_F1: {best_metric:.4f}")

# --- Copy checkpoints to Drive ---
DRIVE_SAVE = os.path.join(MODULE1_DIR, 'checkpoints')
os.makedirs(DRIVE_SAVE, exist_ok=True)
shutil.copy2(f'{CHECKPOINT_DIR}/last.pt', f'{DRIVE_SAVE}/last.pt')
if os.path.exists(f'{CHECKPOINT_DIR}/best.pt'):
    shutil.copy2(f'{CHECKPOINT_DIR}/best.pt', f'{DRIVE_SAVE}/best.pt')
print(f"💾 Checkpoints saved to Drive: {DRIVE_SAVE}")


# =====================================================================
# ██████  CELL 5: TEST EVALUATION & METRICS
# =====================================================================
print(f"\n{'='*70}")
print(f"📊 EVALUATING ON TEST SET")
print(f"{'='*70}\n")

# Load best checkpoint
best_ckpt = torch.load(f'{CHECKPOINT_DIR}/best.pt', map_location=device, weights_only=False)
model_eval = build_model(NUM_CLASSES).to(device)
model_eval.load_state_dict(best_ckpt['model_state_dict'])
model_eval.eval()

inv_label_map = best_ckpt['label_map']  # {idx: species}
class_names = [inv_label_map[i] for i in range(NUM_CLASSES)]

test_ds = TreeDataset(f'{SPLITS_DIR}/test.csv', eval_transforms)
test_loader = DataLoader(test_ds, batch_size=CONFIG['batch_size'], shuffle=False,
                         num_workers=CONFIG['num_workers'], pin_memory=True)

all_preds, all_labels, all_confs = [], [], []
with torch.no_grad():
    for images, labels in tqdm(test_loader, desc="Testing"):
        images = images.to(device, non_blocking=True)
        outputs = model_eval(images)
        probs = torch.softmax(outputs, dim=1)
        confs, preds = probs.max(dim=1)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.tolist())
        all_confs.extend(confs.cpu().tolist())

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)
all_confs = np.array(all_confs)

# --- Metrics ---
accuracy = accuracy_score(all_labels, all_preds)
prec_macro = precision_score(all_labels, all_preds, average='macro', zero_division=0)
rec_macro = recall_score(all_labels, all_preds, average='macro', zero_division=0)
f1_macro = f1_score(all_labels, all_preds, average='macro', zero_division=0)
f1_weighted = f1_score(all_labels, all_preds, average='weighted', zero_division=0)

print(f"\n{'='*50}")
print(f"  TEST RESULTS")
print(f"{'='*50}")
print(f"  Accuracy:          {accuracy:.4f}  ({accuracy*100:.2f}%)")
print(f"  Precision (macro): {prec_macro:.4f}")
print(f"  Recall (macro):    {rec_macro:.4f}")
print(f"  F1 (macro):        {f1_macro:.4f}")
print(f"  F1 (weighted):     {f1_weighted:.4f}")
print(f"  Mean Confidence:   {all_confs.mean():.4f}")
print(f"{'='*50}\n")

# --- Classification Report ---
report_text = classification_report(all_labels, all_preds, target_names=class_names, zero_division=0)
report_dict = classification_report(all_labels, all_preds, target_names=class_names, zero_division=0, output_dict=True)
print("📋 Per-Class Classification Report:")
print(report_text)

# --- Confusion Matrix ---
cm = confusion_matrix(all_labels, all_preds, labels=list(range(NUM_CLASSES)))

# Raw counts
fig_size = max(8, 0.5 * NUM_CLASSES)
plt.figure(figsize=(fig_size, fig_size))
sns.heatmap(cm, annot=(NUM_CLASSES <= 30), fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix (Counts)')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig(f'{PLOTS_DIR}/test_confusion_matrix.png', dpi=150)
plt.show()
print(f"✅ Saved: {PLOTS_DIR}/test_confusion_matrix.png")

# Normalized
cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-9)
plt.figure(figsize=(fig_size, fig_size))
sns.heatmap(cm_norm, annot=(NUM_CLASSES <= 30), fmt='.2f', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.title('Normalized Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig(f'{PLOTS_DIR}/test_confusion_matrix_normalized.png', dpi=150)
plt.show()
print(f"✅ Saved: {PLOTS_DIR}/test_confusion_matrix_normalized.png")

# --- Training Curves ---
hist = pd.DataFrame(history)
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].plot(hist['epoch'], hist['train_loss'], label='Train Loss', marker='o', ms=3)
axes[0].plot(hist['epoch'], hist['val_loss'], label='Val Loss', marker='s', ms=3)
axes[0].set_title('Loss Curves'); axes[0].legend(); axes[0].set_xlabel('Epoch'); axes[0].grid(True)

axes[1].plot(hist['epoch'], hist['train_acc'], label='Train Acc', marker='o', ms=3)
axes[1].plot(hist['epoch'], hist['val_acc'], label='Val Acc', marker='s', ms=3)
axes[1].set_title('Accuracy Curves'); axes[1].legend(); axes[1].set_xlabel('Epoch'); axes[1].grid(True)

axes[2].plot(hist['epoch'], hist['train_f1'], label='Train F1', marker='o', ms=3)
axes[2].plot(hist['epoch'], hist['val_f1'], label='Val F1', marker='s', ms=3)
axes[2].set_title('Macro F1 Curves'); axes[2].legend(); axes[2].set_xlabel('Epoch'); axes[2].grid(True)

plt.tight_layout()
plt.savefig(f'{PLOTS_DIR}/training_curves.png', dpi=150)
plt.show()
print(f"✅ Saved: {PLOTS_DIR}/training_curves.png")

# --- Save all metrics to JSON ---
metrics_summary = {
    'test_samples': len(all_labels),
    'accuracy': accuracy,
    'precision_macro': prec_macro,
    'recall_macro': rec_macro,
    'f1_macro': f1_macro,
    'f1_weighted': f1_weighted,
    'mean_confidence': float(all_confs.mean()),
    'per_class': report_dict,
    'training_epochs': len(history),
    'best_val_f1': best_metric,
}
with open(f'{LOGS_DIR}/test_metrics.json', 'w') as f:
    json.dump(metrics_summary, f, indent=2)

# --- Copy everything to Drive ---
DRIVE_RESULTS = os.path.join(MODULE1_DIR, 'results')
os.makedirs(DRIVE_RESULTS, exist_ok=True)
for src in [f'{PLOTS_DIR}/test_confusion_matrix.png',
            f'{PLOTS_DIR}/test_confusion_matrix_normalized.png',
            f'{PLOTS_DIR}/training_curves.png',
            f'{LOGS_DIR}/test_metrics.json',
            f'{LOGS_DIR}/history.json']:
    if os.path.exists(src):
        shutil.copy2(src, DRIVE_RESULTS)

print(f"\n{'='*70}")
print(f"🎉 ALL DONE! Results saved to: {DRIVE_RESULTS}")
print(f"   Checkpoints: {DRIVE_SAVE}")
print(f"{'='*70}")
