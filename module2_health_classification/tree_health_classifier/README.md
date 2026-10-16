# Tree Health Classification — Module 2
### AI-Powered Tree Carbon Assessment System

Production-ready PyTorch pipeline that classifies the **health condition** of
a tree from a close-up image of a **leaf** (and optionally bark). Output
feeds directly into the carbon-estimation pipeline: unhealthy trees get an
adjusted biomass/carbon-sequestration estimate downstream.

Follows the exact same coding standards / folder architecture as Module 1
(Species Classification). This module is fully independent — it does not
import from or modify Module 1.

---

## 1. Output Contract

```json
{
  "health": "Healthy",
  "confidence": 0.97
}
```

Classes: `Healthy`, `Diseased`, `Nutrient_Deficient`, `Pest_Affected`, `Unknown`.
`Unknown` is not a trained class — it's returned by the inference layer
whenever the top prediction's confidence falls below `inference.confidence_threshold`
in `configs/config.yaml`, since a fine-grained health model should not force
a confident-sounding label on an out-of-distribution or ambiguous image.

## 2. Datasets

| Dataset | Scope | Images | License | Why |
|---|---|---|---|---|
| **PlantVillage** | 14 crop species, 38 disease/healthy classes | ~54,000 | Free for research; check terms for commercial use | Largest clean healthy-vs-diseased corpus; used to build a strong general disease-texture prior |
| **PlantDoc** | 13 species, up to 27 classes (incl. healthy) | ~2,598 | CC BY 4.0 | Real-world, in-field images (not lab background) — closest match to how field photos will actually look |
| **Plant Pathology 2021 (FGVC8)** | Apple: healthy, scab, rust, powdery mildew, frog-eye leaf spot, complex | ~23,249 | Kaggle competition terms (research use) | High-resolution, expert-annotated, real field backgrounds, strong disease-symptom detail |

**Known gap (documented, not papered over):** no public, tree-specific,
close-up dataset exists for `Nutrient_Deficient` or `Pest_Affected` at the
scale of PlantVillage. The taxonomy and pipeline fully support these classes
today — `data/prepare_dataset.py` will merge any data you drop into
`data/raw/<source>/<class>/`. Until domain-specific data is collected, expect
`Healthy` and `Diseased` to be your strongest classes, with `Nutrient_Deficient`
and `Pest_Affected` needing supplementary data collection before deployment
(see README section 6, "Extending the taxonomy").

## 3. Folder Structure

```
tree_health_classifier/
├── configs/
│   └── config.yaml
├── data/
│   ├── download_datasets.sh
│   ├── prepare_dataset.py
│   ├── make_splits.py
│   └── dataset.py
├── models/
│   └── model.py
├── training/
│   ├── utils.py
│   ├── early_stopping.py
│   ├── engine.py            # includes TensorBoard scalar logging
│   └── train.py
├── evaluation/
│   └── evaluate.py          # accuracy/precision/recall/F1/confusion matrix/ROC
├── inference/
│   └── predict.py           # predict_api() -> {"health":..., "confidence":...}
├── export/
│   └── export_onnx.py
├── requirements.txt
└── README.md
```

## 4. Quickstart

```bash
pip install -r requirements.txt

# 1. Download & organize raw data
bash data/download_datasets.sh

# 2. Merge into unified health taxonomy
python data/prepare_dataset.py --config configs/config.yaml

# 3. Create stratified splits
python data/make_splits.py --config configs/config.yaml

# 4. Train (TensorBoard logs to outputs/tb_logs)
python training/train.py --config configs/config.yaml
tensorboard --logdir outputs/tb_logs

# 5. Resume if interrupted
python training/train.py --config configs/config.yaml --resume outputs/checkpoints/last.pt

# 6. Evaluate (accuracy/precision/recall/F1/confusion matrix/ROC/report)
python evaluation/evaluate.py --config configs/config.yaml --checkpoint outputs/checkpoints/best.pt

# 7. Single-image inference
python inference/predict.py --checkpoint outputs/checkpoints/best.pt --image /path/to/leaf.jpg

# 8. Export to ONNX
python export/export_onnx.py --checkpoint outputs/checkpoints/best.pt --output outputs/model.onnx
```

## 5. Design Notes

- **Model**: EfficientNet-B3 (timm, ImageNet-pretrained), same rationale as
  Module 1 — best accuracy/latency trade-off for fine-grained texture
  classification (lesions, discoloration, spotting), and keeps both modules
  at comparable inference cost when run together in the pipeline. Swappable
  via `configs/config.yaml -> model.backbone`.
- **Loss**: label-smoothed cross-entropy; class-weighted via `WeightedRandomSampler`
  to handle the severe imbalance between PlantVillage's ~54k images and
  PlantDoc's ~2.6k images.
- **Confidence-gated `Unknown`**: the model is never trained on an "Unknown"
  class (there's nothing coherent to sample); instead `inference/predict.py`
  applies a confidence threshold at serving time.
- **TensorBoard**: scalar logging (loss, accuracy, macro-F1, LR) per epoch to
  `outputs/tb_logs`, in addition to the JSON history file used in Module 1.
- **ROC curve**: one-vs-rest multi-class ROC with per-class AUC, saved as PNG.

## 6. Extending the taxonomy

To add `Nutrient_Deficient` / `Pest_Affected` training data:
1. Drop images into `data/raw/<your_source>/<class_name>/`.
2. Add an entry to `class_map` in `configs/config.yaml` pointing the
   canonical class (e.g. `Pest_Affected`) at that new source subfolder.
3. Re-run `prepare_dataset.py` and `make_splits.py` — no code changes needed.
