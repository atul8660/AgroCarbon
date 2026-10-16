#!/usr/bin/env bash
# =============================================================
# Download & extract all raw datasets for tree health classification.
#
# Requires: kaggle CLI configured with ~/.kaggle/kaggle.json
# (https://www.kaggle.com/docs/api -> "Create New API Token")
# =============================================================
set -euo pipefail

RAW_DIR="data/raw"
mkdir -p "${RAW_DIR}"
cd "${RAW_DIR}"

echo "=============================================================="
echo "[1/3] PlantVillage — 38 classes, ~54,000 images"
echo "=============================================================="
mkdir -p plantvillage
if command -v kaggle &> /dev/null; then
    kaggle datasets download -d abdallahalidev/plantvillage-dataset \
        -p plantvillage --unzip
else
    echo "  kaggle CLI not found. Manual steps:"
    echo "  1) https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset"
    echo "  2) Download & unzip into ${RAW_DIR}/plantvillage/"
fi

echo "=============================================================="
echo "[2/3] PlantDoc — 13 species, up to 27 classes, ~2,598 images"
echo "=============================================================="
mkdir -p plantdoc
echo "  Manual step (official repo, CC BY 4.0):"
echo "  1) https://github.com/pratikkayal/PlantDoc-Dataset"
echo "  2) git clone https://github.com/pratikkayal/PlantDoc-Dataset.git plantdoc_raw"
echo "  3) Reorganize into ${RAW_DIR}/plantdoc/<class_name>/*.jpg"
echo "     (the classification-folder version is also mirrored on Kaggle,"
echo "     search 'PlantDoc' for a pre-split classification variant)"

echo "=============================================================="
echo "[3/3] Plant Pathology 2021 - FGVC8 — apple foliar disease, ~23,249 images"
echo "=============================================================="
mkdir -p plant_pathology_2021
if command -v kaggle &> /dev/null; then
    kaggle competitions download -c plant-pathology-2021-fgvc8 \
        -p plant_pathology_2021
    (cd plant_pathology_2021 && unzip -o plant-pathology-2021-fgvc8.zip)
else
    echo "  kaggle CLI not found. Manual steps:"
    echo "  1) https://www.kaggle.com/competitions/plant-pathology-2021-fgvc8/data"
    echo "     (must accept competition rules to download)"
    echo "  2) Download & unzip into ${RAW_DIR}/plant_pathology_2021/"
    echo "  3) Use train.csv label column to sort images into"
    echo "     ${RAW_DIR}/plant_pathology_2021/<class_name>/*.jpg"
fi

echo "=============================================================="
echo "[optional] Custom domain data for Nutrient_Deficient / Pest_Affected"
echo "=============================================================="
mkdir -p custom/nutrient_deficient custom/pest_affected
echo "  Place field-collected or curated images here:"
echo "  ${RAW_DIR}/custom/nutrient_deficient/*.jpg"
echo "  ${RAW_DIR}/custom/pest_affected/*.jpg"

cd - > /dev/null
echo ""
echo "Done. Verify each raw dataset directory before running prepare_dataset.py"
echo "Expected structure: data/raw/<dataset_name>/<class>/*.jpg"
