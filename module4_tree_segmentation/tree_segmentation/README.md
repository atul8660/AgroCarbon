# Tree Segmentation — Module 4
### AI-Powered Tree Carbon Assessment System (Multi-View Computer Vision)

Production-grade semantic segmentation of the single tree cropped by
Module 3. Produces a binary tree mask, canopy/trunk sub-masks, an RGBA
cutout, a validated polygon, and rich pixel/geometry metadata — all
consumed directly by Module 5 (Structural Feature Extraction) with **no
further preprocessing**.

Follows the same coding standards / folder philosophy as Modules 1–3, with
two segmentation-specific additions (`postprocessing/`, `analysis/`)
justified in the design notes below. This module is fully independent — it
does not import from or modify Modules 1–3, and its only input is
`cropped_tree.jpg`.

---

## 1. Pipeline Position

```
cropped_tree.jpg (from Module 3, untouched)
        │
        ▼
Module 4 — Tree Segmentation (THIS MODULE)
        │  tree_mask.png, tree_rgba.png, canopy_mask.png, trunk_mask.png,
        │  tree_polygon.json, mask_metadata.json
        ▼
Module 5 — Structural Feature Extraction → Module 6 — Biomass → Module 7 — Carbon
```

## 2. Model Choice: SegFormer-B5 (production) + SAM2 (supporting role)

Full comparison against Mask2Former and a SAM2-only design is in the
project design notes. Summary:

- **SegFormer-B5** (`nvidia/mit-b5` encoder) fine-tuned as a **3-class
  semantic segmenter** (background / canopy / trunk) is the model actually
  served in production. One forward pass gives tree + canopy + trunk masks
  together, with per-pixel confidence for quality scoring. Fast, single
  clean ONNX graph, fine-tunes well on our realistic dataset scale
  (thousands–tens of thousands of masks), unlike Mask2Former which wants
  COCO-panoptic-scale data to earn its extra machinery.
- **SAM2** is *not* the served model. It's used for (a) offline box-prompted
  pseudo-label generation to bootstrap the training set, and (b) an
  optional, config-gated zero-shot fallback when SegFormer's quality score
  is low. This keeps default inference latency and ONNX deployment simple
  while still getting SAM2's zero-shot strength where it actually helps.

## 3. Output Contract

Success:
```json
{
  "tree_mask": "outputs/results/tree_0001/tree_mask.png",
  "tree_rgba": "outputs/results/tree_0001/tree_rgba.png",
  "canopy_mask": "outputs/results/tree_0001/canopy_mask.png",
  "trunk_mask": "outputs/results/tree_0001/trunk_mask.png",
  "polygon": "outputs/results/tree_0001/tree_polygon.json",
  "coverage": 0.95,
  "quality": 0.97,
  "ready_for_module5": true,
  "metadata": "outputs/results/tree_0001/mask_metadata.json"
}
```

Rejected / low-quality segmentation:
```json
{
  "error": "MASK_QUALITY_REJECTED",
  "reason": "coverage_too_low",
  "coverage": 0.03,
  "quality": 0.21,
  "ready_for_module5": false
}
```

## 4. Datasets

Datasets requested by name were checked against real public sources; where
a requested name doesn't correspond to a verifiable public dataset, the
closest real, verified alternative is used instead and the substitution is
called out explicitly — no invented datasets.

| Requested | Used | Images | License | Purpose | Advantages | Disadvantages |
|---|---|---|---|---|---|---|
| TreeSeg | **TreeSeg (Roboflow Universe, ITULab)** — a real, verified instance-segmentation dataset under this name | 161 images | CC BY 4.0 (Roboflow Universe default) | Small, clean tree instance-mask seed set | Ready-to-use polygon/mask annotations, no conversion needed | Very small — training-support only, not sufficient alone |
| TreeMask | **No verifiable public dataset by this exact name was found.** Substituted with additional Roboflow Universe tree-segmentation projects (e.g. `urban-trees`, CC BY 4.0, 27 images) as a small supplementary source | ~27 images | CC BY 4.0 | Supplementary mask diversity | Free, ready-to-use | Tiny; noisy label quality typical of community-uploaded sets |
| CityTrees | **No verifiable public dataset by this exact name was found.** Substituted with the real **Urban Street Tree dataset** (Yang et al. 2023, ScienceDirect/GitHub) | 41,467 images total, 22,872 annotated | Publicly released for research (see dataset site for full terms) | **Primary source** — has dedicated leaf/tree/trunk/branch/flower/fruit polygon sub-datasets across 10 city scenes, 50 species | Exactly matches our canopy-vs-trunk labeling need; fine-grained polygon annotations; real street conditions | Large download; per-organ splits must be merged carefully; license terms should be re-checked before commercial use |
| NEON | **NEON Tree Evaluation** (Weecology/NEON) | 30,975 boxes across 22 sites | Open academic release | Auxiliary — box-to-mask pseudo-labeling via SAM2, aerial canopy-shape prior only | Large, high-quality boxes, wide site diversity | **Boxes only, not pixel masks** — must be converted via SAM2-prompted pseudo-labeling; aerial view doesn't match ground-level ground-truth framing |
| Open Images Segmentation | **Open Images V7, "Tree" class, instance-segmentation subset** | Segmentation masks cover 350 of 600 boxable classes across ~944k images total (Tree-class subset extracted at download time) | Annotations CC BY 4.0 / images CC BY 2.0 | Complex real-scene diversity, hard negatives (non-tree green foliage, hedges, etc.) | Large, diverse, high mask quality (~84% mIoU per Google's QA process) | Must verify "Tree" is populated in the 350 masked classes at download time — not guaranteed for every boxable class |

## 5. Folder Structure

```
tree_segmentation/
├── configs/
│   └── config.yaml              # OmegaConf/Hydra-style hierarchical config
├── data/
│   ├── download_datasets.sh
│   ├── prepare_dataset.py       # clean, validate, polygon<->mask conversion, resize
│   ├── make_splits.py           # split + dataset stats
│   ├── sam2_pseudo_labeling.py  # SAM2 box-prompted pseudo-mask bootstrapping
│   └── dataset.py                # SegmentationDataset + Albumentations transforms
├── models/
│   └── model.py                  # SegFormer-B5 factory + SAM2 wrapper
├── training/
│   ├── utils.py
│   ├── early_stopping.py
│   ├── engine.py                 # train/val loop, TensorBoard
│   └── train.py                  # resume-capable entrypoint
├── postprocessing/
│   └── postprocess.py            # morphology, CCA, hole fill, polygon approx, contours
├── analysis/
│   └── tree_analysis.py          # coverage/occupancy/orientation/occlusion/shadow metrics
├── evaluation/
│   └── evaluate.py               # IoU/Dice/BoundaryIoU/P/R/PixelAcc/mIoU/speed/viz
├── inference/
│   └── segment.py                # TreeSegmenter: the Module 5-facing API
├── export/
│   └── export_onnx.py
├── requirements.txt
└── README.md
```

*(`postprocessing/` and `analysis/` are new relative to Modules 1–3 because
segmentation inherently produces artifacts — masks, polygons, geometric
statistics — that classification/detection don't. They follow the same
config-driven, docstring-first, single-responsibility style as every other
module folder.)*

## 6. Quickstart

```bash
pip install -r requirements.txt

bash data/download_datasets.sh
python data/prepare_dataset.py --config configs/config.yaml
python data/sam2_pseudo_labeling.py --config configs/config.yaml   # optional, expands training set
python data/make_splits.py --config configs/config.yaml

python training/train.py --config configs/config.yaml
python training/train.py --config configs/config.yaml --resume outputs/checkpoints/last.pt

python evaluation/evaluate.py --config configs/config.yaml --checkpoint outputs/checkpoints/best.pt

python inference/segment.py --checkpoint outputs/checkpoints/best.pt --image cropped_tree.jpg

python export/export_onnx.py --checkpoint outputs/checkpoints/best.pt --output outputs/model.onnx
```
