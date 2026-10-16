#!/usr/bin/env bash
# =============================================================
# Download & extract all raw datasets for whole-tree detection.
#
# Dataset notes (see README section 3 for full advantages/limitations):
#   - Auto Arborist: mixed per-city open-data licenses, verify each city's
#     terms before commercial use. Primary source of street-level "tree in
#     urban context" scenes with usable bbox regions.
#   - Open Images V7 "Tree" class: CC BY 4.0 annotations / CC BY 2.0 images.
#     Primary source of complex, multi-object outdoor scenes.
#   - NEON Tree Evaluation: open academic release. Aerial canopy view only —
#     used as an auxiliary tree-shape pretraining set, not the main source.
# =============================================================
set -euo pipefail

RAW_DIR="data/raw"
mkdir -p "${RAW_DIR}"
cd "${RAW_DIR}"

echo "=============================================================="
echo "[1/3] Auto Arborist — 2.5M+ trees, street-level + aerial imagery"
echo "=============================================================="
mkdir -p auto_arborist
echo "  Manual step (no single-archive CLI download):"
echo "  1) https://google.github.io/auto-arborist/"
echo "  2) git clone https://github.com/google/auto-arborist.git auto_arborist_repo"
echo "  3) Follow the repo's instructions to pull per-city TFRecord/JSON shards"
echo "     and per-tree imagery for the cities you want to include."
echo "  4) Export per-tree records (with xmin/xmax/ymin/ymax fields) as JSON"
echo "     into ${RAW_DIR}/auto_arborist/<city>/*.json + matching images/"
echo "  5) Double-check each city's license (listed in the repo) before use."

echo "=============================================================="
echo "[2/3] Open Images V7 — 'Tree' class subset"
echo "=============================================================="
mkdir -p open_images_tree
if command -v python3 &> /dev/null && python3 -c "import openimages" &> /dev/null; then
    echo "  openimages toolkit detected — downloading Tree class..."
    python3 -m openimages.download --classes Tree --limit 20000 \
        --dataset-dir open_images_tree
else
    echo "  Manual/CLI steps (Common Visual Data Foundation mirror):"
    echo "  1) pip install openimages"
    echo "  2) python3 -m openimages.download --classes Tree --limit 20000 \\"
    echo "       --dataset-dir ${RAW_DIR}/open_images_tree"
    echo "     (or use the official downloader: https://storage.googleapis.com/openimages/web/download.html"
    echo "     and filter train-annotations-bbox.csv for LabelName == the 'Tree' MID)"
fi

echo "=============================================================="
echo "[3/3] NEON Tree Evaluation — 30,975 boxes, 22 sites (aerial RGB)"
echo "=============================================================="
mkdir -p neon_tree_eval
echo "  Manual step:"
echo "  1) https://github.com/weecology/NeonTreeEvaluation"
echo "  2) git clone https://github.com/weecology/NeonTreeEvaluation.git neon_tree_eval_repo"
echo "  3) Copy evaluation/ (images) and annotations/ (Pascal-VOC XML) into"
echo "     ${RAW_DIR}/neon_tree_eval/images/ and ${RAW_DIR}/neon_tree_eval/annotations/"
echo "  Static mirror also available at: https://zenodo.org/record/5914554"

cd - > /dev/null
echo ""
echo "Done. Verify each raw dataset directory before running prepare_dataset.py"
echo "Expected structure per source is documented in configs/config.yaml -> sources"
