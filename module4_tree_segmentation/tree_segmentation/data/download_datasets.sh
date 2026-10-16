#!/usr/bin/env bash
# =============================================================
# Download & extract all raw datasets for tree segmentation.
#
# See README section 4 for full purpose/license/advantages/limitations,
# including honest notes on where a requested dataset name ("TreeMask",
# "CityTrees") had no verifiable public source and was substituted.
# =============================================================
set -euo pipefail

RAW_DIR="data/raw"
mkdir -p "${RAW_DIR}"
cd "${RAW_DIR}"

echo "=============================================================="
echo "[1/4] TreeSeg (Roboflow Universe, ITULab) — 161 images"
echo "=============================================================="
mkdir -p treeseg_roboflow
echo "  Manual step (Roboflow account + API key required for CLI export):"
echo "  1) https://universe.roboflow.com/itulab/treeseg"
echo "  2) Export in 'COCO Segmentation' format"
echo "  3) Extract into ${RAW_DIR}/treeseg_roboflow/ (images/ + _annotations.coco.json)"

echo "=============================================================="
echo "[2/4] Urban Street Tree dataset (Yang et al. 2023) — 22,872 annotated"
echo "=============================================================="
mkdir -p urban_street_tree
echo "  Manual step (used here in place of an unverifiable 'CityTrees' dataset):"
echo "  1) https://ytt917251944.github.io/dataset_jekyll/"
echo "  2) Download the 'tree' and 'trunk' organ sub-datasets (polygon annotations)"
echo "  3) Extract into ${RAW_DIR}/urban_street_tree/<organ>/images/ + <organ>/annotations/"
echo "  4) Re-check the dataset site's license terms before any commercial use"

echo "=============================================================="
echo "[3/4] Open Images V7 — 'Tree' class segmentation subset"
echo "=============================================================="
mkdir -p open_images_tree_seg
if command -v python3 &> /dev/null && python3 -c "import openimages" &> /dev/null; then
    echo "  openimages toolkit detected — downloading Tree class segmentation masks..."
    python3 -m openimages.download --classes Tree --limit 10000 \
        --dataset-dir open_images_tree_seg --annotation-format segmentation
else
    echo "  Manual/CLI steps:"
    echo "  1) pip install openimages"
    echo "  2) python3 -m openimages.download --classes Tree --limit 10000 \\"
    echo "       --dataset-dir ${RAW_DIR}/open_images_tree_seg --annotation-format segmentation"
    echo "  NOTE: verify 'Tree' is among the 350 classes with mask coverage before relying on this source"
    echo "  (see: https://storage.googleapis.com/openimages/web/factsfigures_v7.html)"
fi

echo "=============================================================="
echo "[4/4] NEON Tree Evaluation — boxes only, converted via SAM2 later"
echo "=============================================================="
mkdir -p neon_raw
echo "  Manual step (same source used in Module 3):"
echo "  1) git clone https://github.com/weecology/NeonTreeEvaluation.git neon_raw/repo"
echo "  2) After downloading, run data/sam2_pseudo_labeling.py to convert NEON"
echo "     bounding boxes into pseudo-masks (writes to ${RAW_DIR}/neon_pseudo_labeled/)"

cd - > /dev/null
echo ""
echo "Done. Verify each raw dataset directory before running prepare_dataset.py"
echo "Expected structure per source is documented in configs/config.yaml -> sources"
