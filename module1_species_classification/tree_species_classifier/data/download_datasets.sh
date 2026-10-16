#!/usr/bin/env bash
# =============================================================
# Download & extract all raw datasets for tree species classification.
#
# Requires: kaggle CLI configured with ~/.kaggle/kaggle.json
# (https://www.kaggle.com/docs/api -> "Create New API Token")
#
# Some sources (Mendeley, Figshare) don't have a stable CLI download,
# so those steps are documented as manual downloads with the exact
# URL to fetch and the target extraction path.
# =============================================================
set -euo pipefail

RAW_DIR="data/raw"
mkdir -p "${RAW_DIR}"
cd "${RAW_DIR}"

echo "=============================================================="
echo "[1/5] Plant Leaves for Image Classification (Kaggle)"
echo "=============================================================="
mkdir -p plant_leaves
if command -v kaggle &> /dev/null; then
    kaggle datasets download -d csafrit2/plant-leaves-for-image-classification \
        -p plant_leaves --unzip
else
    echo "  kaggle CLI not found. Manual steps:"
    echo "  1) https://www.kaggle.com/datasets/csafrit2/plant-leaves-for-image-classification"
    echo "  2) Download & unzip into ${RAW_DIR}/plant_leaves/"
fi

echo "=============================================================="
echo "[2/5] Indian Medicinal Leaves Dataset (Kaggle mirror of Mendeley)"
echo "=============================================================="
mkdir -p medicinal_leaves
if command -v kaggle &> /dev/null; then
    kaggle datasets download -d aryashah2k/indian-medicinal-leaves-dataset \
        -p medicinal_leaves --unzip
else
    echo "  kaggle CLI not found. Manual steps:"
    echo "  1) https://www.kaggle.com/datasets/aryashah2k/indian-medicinal-leaves-dataset"
    echo "  2) Download & unzip into ${RAW_DIR}/medicinal_leaves/"
fi

echo "=============================================================="
echo "[3/5] BarkVisionAI — 13 Indian tree species bark images"
echo "=============================================================="
mkdir -p barkvisionai
echo "  Manual step (no direct CLI mirror):"
echo "  1) https://github.com/Forest-Economy-Alliance/BarkVisionAI"
echo "  2) Follow the repo's data-access instructions (Figshare DOI: 10.6084/m9.figshare.28427246)"
echo "  3) Extract species subfolders into ${RAW_DIR}/barkvisionai/<species>/"

echo "=============================================================="
echo "[4/5] BarkNet 1.0 — 23 species bark images (Universite Laval)"
echo "=============================================================="
mkdir -p barknet
echo "  Manual step:"
echo "  1) https://github.com/ulaval-damas/tree-bark-classification"
echo "  2) Download the dataset archive per repo instructions"
echo "  3) Extract into ${RAW_DIR}/barknet/<species>/"

echo "=============================================================="
echo "[5/5] LeafSnap — 185 species leaf dataset"
echo "=============================================================="
mkdir -p leafsnap
echo "  Manual step:"
echo "  1) http://leafsnap.com/dataset/"
echo "  2) Download & extract into ${RAW_DIR}/leafsnap/<species>/"

cd - > /dev/null
echo ""
echo "Done. Verify each raw dataset directory before running prepare_dataset.py"
echo "Expected structure: data/raw/<dataset_name>/<species_or_class>/*.jpg"
