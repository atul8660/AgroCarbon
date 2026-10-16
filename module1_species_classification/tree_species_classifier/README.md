# Tree Species Classification — Module 1
### AI-Powered Tree Carbon Assessment System

Production-ready PyTorch pipeline that classifies tree species from a close-up
image of a **leaf and/or bark**. This model is Module 1 of a larger carbon
estimation system (species → allometric equation → biomass → carbon).

---

## 1. Datasets

| Dataset | Species | Images | Modality | Why |
|---|---|---|---|---|
| **BarkVisionAI** | 13 (India) | 156,001 | Bark | Largest Indian bark dataset; captures real bark texture variability across seasons/devices |
| **Indian Medicinal Leaves Dataset** (Mendeley/Kaggle, Pushpa & Rani) | 80 (India) | ~6,900 | Leaf | Smartphone-captured, matches real field-photo conditions |
| **Plant Leaves for Image Classification** (Mendeley/Kaggle) | 12 (India-common) | 4,502 | Leaf | High-res, healthy + diseased leaves of common Indian trees (Mango, Jamun, Guava, Arjun, Chinar...) |
| **BarkNet 1.0** (Université Laval) | 23 | 23,000+ | Bark | Largest general bark benchmark; boosts bark-feature generalization |
| **LeafSnap** | 185 | 23,000+ | Leaf | Broadens leaf-shape generalization beyond Indian species |

All datasets are merged into one unified `ImageFolder`-style taxonomy defined
in `configs/config.yaml` under `class_map`. This lets you add/remove datasets
or species without touching any code.

> Note: Some sources require manual acceptance of a license/download form
> (Kaggle, Mendeley). `data/download_datasets.sh` uses the Kaggle API where
> possible and documents manual steps where it isn't.

---

## 2. Folder Structure

```
tree_species_classifier/
├── configs/
│   └── config.yaml              # single source of truth for paths, classes, hyperparams
├── data/
│   ├── download_datasets.sh     # fetch + extract raw datasets
│   ├── prepare_dataset.py       # merge raw sources -> data/processed/<species>/*.jpg
│   ├── make_splits.py           # stratified train/val/test split -> CSV manifests
│   └── dataset.py               # PyTorch Dataset + Albumentations transforms
├── models/
│   └── model.py                 # model factory (EfficientNet-B3 default, swappable)
├── training/
│   ├── utils.py                 # seed, checkpoint save/load, AverageMeter
│   ├── early_stopping.py
│   ├── engine.py                # train_one_epoch / validate
│   └── train.py                 # main training entrypoint (resume-capable)
├── evaluation/
│   └── evaluate.py              # accuracy/precision/recall/F1/confusion matrix on test split
├── inference/
│   └── predict.py               # single-image inference -> species + confidence
├── export/
│   └── export_onnx.py           # export + verify ONNX model
├── requirements.txt
└── README.md
```

## 3. Quickstart

```bash
pip install -r requirements.txt

# 1. Download & organize raw data
bash data/download_datasets.sh

# 2. Merge into unified class folders
python data/prepare_dataset.py --config configs/config.yaml

# 3. Create stratified splits
python data/make_splits.py --config configs/config.yaml

# 4. Train
python training/train.py --config configs/config.yaml

# 5. Resume if interrupted
python training/train.py --config configs/config.yaml --resume outputs/checkpoints/last.pt

# 6. Evaluate on held-out test set
python evaluation/evaluate.py --config configs/config.yaml --checkpoint outputs/checkpoints/best.pt

# 7. Run inference on a single image
python inference/predict.py --checkpoint outputs/checkpoints/best.pt --image /path/to/leaf.jpg

# 8. Export to ONNX
python export/export_onnx.py --checkpoint outputs/checkpoints/best.pt --output outputs/model.onnx
```

## 4. Design Notes

- **Model**: EfficientNet-B3 (timm, ImageNet-pretrained) — best accuracy/latency
  trade-off for fine-grained texture classification (leaf venation, bark ridges)
  at a size deployable on-device/server without heavy GPU. Backbone is
  swappable via `configs/config.yaml -> model.backbone` (e.g. `convnext_tiny`,
  `vit_small_patch16_224`) with zero code changes.
- **Loss**: label-smoothed cross-entropy (handles noisy crowd-sourced labels).
- **Class imbalance**: `WeightedRandomSampler` derived from inverse class frequency.
- **Augmentation**: Albumentations pipeline simulating field conditions —
  rotation/flip (leaves photographed at any angle), color jitter (lighting),
  coarse dropout (occlusion by other leaves/branches), blur (camera shake).
- **Mixed precision**: `torch.cuda.amp` for ~2x training speedup on modern GPUs.
- **Early stopping**: monitored on validation macro-F1 (robust to class imbalance).
- **Checkpointing**: full training state (model, optimizer, scheduler, scaler,
  epoch, best metric) saved every epoch (`last.pt`) and on improvement (`best.pt`),
  enabling exact resume.
