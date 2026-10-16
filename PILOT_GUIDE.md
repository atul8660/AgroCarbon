# Pilot Training Plan — Mango & Neem

This is a scoped-down, fast-to-run version of Modules 1-4, meant to prove
the whole pipeline works end-to-end before investing in full multi-species,
multi-dataset training. It changes **data volume and class scope only** —
none of the model architectures, training loops, or inference contracts
described in each module's own README change.

## How to use these files

Each `config.pilot.yaml` goes into the matching module's existing `configs/`
folder, alongside (not replacing) the full `config.yaml`:

```
tree_carbon_assessment_system/
├── module1_species_classification/configs/config.pilot.yaml
├── module2_health_classification/configs/config.pilot.yaml
├── module3_tree_detection/configs/config.pilot.yaml
└── module4_tree_segmentation/configs/config.pilot.yaml
```

Then run every script with `--config configs/config.pilot.yaml` instead of
the default, e.g.:

```bash
cd module1_species_classification
python data/prepare_dataset.py --config configs/config.pilot.yaml
python data/make_splits.py --config configs/config.pilot.yaml
python training/train.py --config configs/config.pilot.yaml
```

Pilot runs write to `outputs_pilot/` and `data/processed_pilot/` /
`data/splits_pilot/` — separate from the full-run paths — so you can later
run the full config without the pilot run's artifacts getting in the way.

## What's scoped down, and why

| Module | What changed | Why |
|---|---|---|
| **1 — Species** | `class_map` → Mango + Neem only. Epochs 60→25, batch 32→16. | 2 classes minimum for a classifier to mean anything; both species already have real source folders mapped. |
| **2 — Health** | `class_map` → Healthy + Diseased only (dropped Nutrient_Deficient/Pest_Affected, which had no real data anyway). Epochs 50→20. | Faster pilot; those 2 classes were already flagged as data-scarce in the full config. **Not species-matched to Mango/Neem** — see the caveat in that module's pilot config header. |
| **3 — Detection** | Only `open_images_tree` source needs populating (Auto Arborist/NEON can stay empty). Model `yolo11m`→`yolo11n`, image size 960→640, epochs 100→40. | Detector is single-class/species-agnostic; this is purely about faster iteration and skipping the harder manual dataset setups for now. |
| **4 — Segmentation** | Only `treeseg_roboflow` source needs populating (161 ready-to-use images, no other sources needed). Encoder `mit-b5`→`mit-b0`, image size 640→512, epochs 80→30. SAM2 pseudo-labeling disabled. | Segmenter is also species-agnostic; smallest real dataset gets you a working pipeline fastest. **Trunk masks will be unreliable in this pilot** — TreeSeg has no trunk polygons, so trunk supervision is effectively absent until Urban Street Tree data is added later. |

## Minimum data to collect for the pilot

- **Module 1**: ~30-50 images each of Mango and Neem leaves (Kaggle's "Plant
  Leaves for Image Classification" and "Indian Medicinal Leaves Dataset"
  both have Mango and Neem — see Module 1's `download_datasets.sh`).
- **Module 2**: ~30-50 images each of a healthy/diseased pair from
  PlantVillage (e.g. `Apple___healthy` vs `Apple___Apple_scab` — one of the
  larger, cleaner classes in that dataset).
- **Module 3**: ~100-150 whole-tree images with boxes from Open Images'
  `Tree` class (via the `openimages` pip toolkit, see Module 3's
  `download_datasets.sh`).
- **Module 4**: the full TreeSeg Roboflow dataset (161 images) — just
  download it whole, no subsetting needed.

## After the pilot proves out

Once all four modules train, evaluate, and run inference cleanly on this
small scope, the natural next steps (not part of this pilot) are:
1. Expand Module 1/2's `class_map` back toward the full species/health list.
2. Populate Module 3/4's additional sources (Auto Arborist, Urban Street
   Tree, NEON) for better generalization and real trunk supervision.
3. Switch Module 3/4 back to their larger architectures (`yolo11m`,
   `mit-b5`) once you're optimizing for accuracy rather than iteration speed.
4. Re-run each module against its full `config.yaml`.

None of this requires code changes — only swapping which config file you
pass to each script, exactly as documented in each module's own README.
