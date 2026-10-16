# Whole Tree Detection — Module 3
### AI-Powered Tree Carbon Assessment System (Multi-View Computer Vision)

Production-ready YOLOv11 pipeline that detects trees in complex outdoor
scenes and hands off exactly **one** clean, high-resolution, padded crop of
the primary tree to Module 4 (Segmentation).

Follows the same coding standards / folder architecture as Modules 1–2.
This module is fully independent — it does not import from or modify
Module 1 or Module 2.

---

## 1. Pipeline Position

```
Whole Tree Image
      │
      ▼
Module 3 — Tree Detection (THIS MODULE)
      │  bbox, confidence, cropped_tree.jpg, mask_placeholder.png
      ▼
Module 4 — Tree Segmentation
      │
      ▼
Module 5 — Structural Feature Extraction → Module 6 — Biomass → Module 7 — Carbon
```

Module 3 is a **single-class detector** ("tree"). Its only job is: find every
tree-shaped object in a complex scene (buildings/vehicles/people/farmland/
forest all present), pick the one instance that should be treated as *the*
subject tree, and produce a crop clean enough that Module 4 never has to
guess which object it's segmenting.

## 2. Output Contract

Success:
```json
{
  "bbox": [x1, y1, x2, y2],
  "confidence": 0.97,
  "cropped_tree": "outputs/crops/tree_0001.jpg",
  "module4_input": "outputs/crops/tree_0001.jpg",
  "mask_placeholder": "outputs/crops/tree_0001_mask_placeholder.png",
  "selection_rule": "single_detection",
  "metadata": { "...": "..." }
}
```

No tree found:
```json
{
  "error": "NO_TREE_DETECTED",
  "bbox": null,
  "confidence": 0.0,
  "cropped_tree": null,
  "module4_input": null
}
```

## 3. Datasets

| Dataset | Images | License | Role in this pipeline |
|---|---|---|---|
| **Auto Arborist** | 2.5M+ trees, aerial + street-level imagery for 1M trees | Mixed per-city open-data licenses (verify per source city) | Primary source of real urban street-level "whole tree in context" scenes with usable bbox regions |
| **Google Open Images V7 (`Tree` class)** | 9.2M images total (Tree-class subset extracted via OIDv7 toolkit) | Annotations CC BY 4.0, images CC BY 2.0 | Primary source of *complex scenes* — trees alongside buildings, cars, people, farmland — and of hard negatives |
| **NEON Tree Evaluation** | 30,975 hand-verified boxes across 22 sites + 10,000+ training crowns | Open academic release (Weecology/NEON) | Auxiliary aerial-canopy pretraining set for tree-shape generalization only (not ground-level framing) |
| **iNaturalist** | Millions of tree photos | Mixed CC (mostly CC BY-NC) | Raw image source for diversity/hard-negative mining only — **has no ready-made detection boxes**, not used as an annotated source |

See design rationale (advantages/limitations) in the project conversation /
top-level docs — summarized briefly in each dataset's comment block in
`data/download_datasets.sh`.

## 4. Folder Structure

```
tree_detection/
├── configs/
│   └── config.yaml              # dataset.yaml gen params, training hp, selection-rule weights
├── data/
│   ├── download_datasets.sh
│   ├── prepare_dataset.py       # clean, validate, convert all sources -> unified YOLO txt labels
│   ├── make_splits.py           # stratified-by-source train/val/test split + dataset.yaml
│   └── dataset_stats.py         # class balance, box-size histogram, per-source counts
├── models/
│   └── model.py                 # thin wrapper around ultralytics YOLO for a consistent interface
├── training/
│   ├── utils.py
│   └── train.py                 # YOLOv11 fine-tuning entrypoint (resume-capable)
├── evaluation/
│   └── evaluate.py              # mAP50/mAP50-95/P/R/F1/PR curve/confusion matrix/inference speed
├── inference/
│   └── detect.py                # TreeDetector: load_model/detect_tree/select_primary/crop_tree/predict_api
├── export/
│   └── export_onnx.py
├── requirements.txt
└── README.md
```

## 5. Quickstart

```bash
pip install -r requirements.txt

# 1. Download & organize raw data
bash data/download_datasets.sh

# 2. Clean, validate, convert to unified YOLO-format labels
python data/prepare_dataset.py --config configs/config.yaml

# 3. Stratified split + generate dataset.yaml for YOLO training
python data/make_splits.py --config configs/config.yaml

# 4. Dataset statistics (class balance, box sizes, per-source counts)
python data/dataset_stats.py --config configs/config.yaml

# 5. Train (transfer learning from YOLOv11 COCO checkpoint)
python training/train.py --config configs/config.yaml

# 6. Resume if interrupted
python training/train.py --config configs/config.yaml --resume outputs/checkpoints/last.pt

# 7. Evaluate (mAP/P/R/F1/PR curve/confusion matrix/inference speed)
python evaluation/evaluate.py --config configs/config.yaml --checkpoint outputs/checkpoints/best.pt

# 8. Run detection -> primary-tree selection -> crop for Module 4
python inference/detect.py --checkpoint outputs/checkpoints/best.pt --image /path/to/scene.jpg

# 9. Export to ONNX
python export/export_onnx.py --checkpoint outputs/checkpoints/best.pt --output outputs/model.onnx
```

## 6. Primary Tree Selection Rule

Documented in `configs/config.yaml -> selection` and implemented in
`inference/detect.py -> TreeDetector.select_primary_tree()`:

- **0 detections** ≥ `min_confidence` → structured error response, `NO_TREE_DETECTED`, confidence 0.0. Never passes an unusable image to Module 4.
- **1 detection** → used directly.
- **>1 detections** → composite score per candidate:

  ```
  score = area_weight * normalized_bbox_area
        + center_weight * (1 - normalized_distance_from_image_center)
  ```

  Defaults: `area_weight = 0.7`, `center_weight = 0.3` (config-tunable). This
  favors the tree that is both the dominant subject (large box) and centrally
  framed (how whole-tree photos are typically composed), while still letting
  a very large but off-center tree win over a small centered one. All
  candidate scores are included in `metadata.candidates` for auditability.

## 7. Crop Contract for Module 4

- `crop_padding_ratio` (default 0.15) adds padding around the bbox on all
  sides before cropping, so branches/canopy edges are never clipped.
- Crop is clipped to image bounds, then upsampled (Lanczos) if it falls under
  `min_crop_resolution` to guarantee Module 4 receives sufficient detail.
- A bbox-derived elliptical **mask placeholder** is saved alongside the crop
  — explicitly not a real segmentation, just a documented, overwritable
  starting point/sanity check for Module 4.
