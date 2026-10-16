"""
=============================================================================
 Module 1 — Tree Species Classification — FIXED Kaggle Training
 Paste each "CELL" section into a separate Kaggle notebook cell.
=============================================================================
"""

# =====================================================================
# CELL 1: SETUP — Extract codebase & install deps
# =====================================================================
import os, glob

zip_files = glob.glob('/kaggle/input/**/*.zip', recursive=True)
target = [z for z in zip_files if 'tree_species' in z]

if not target:
    print('Checking if codebase was already unpacked by Kaggle...')
    req_files = glob.glob('/kaggle/input/**/requirements.txt', recursive=True)
    if req_files:
        code_dir = os.path.dirname(req_files[0])
        os.system(f"cp -r '{code_dir}'/* /kaggle/working/")
    else:
        print('ERROR: Could not find your codebase files.')
else:
    zip_path = target[0]
    print(f'Found ZIP at: {zip_path}')
    os.system(f'unzip -q "{zip_path}" -d .')

print('Installing requirements...')
os.system('pip install -q -r requirements.txt')


# =====================================================================
# CELL 2: MAP DATASETS — Fixed paths & species name mapping
# =====================================================================
import os, glob, shutil

# ---------------------------------------------------------------
# CORRECT Kaggle mount paths (datasets/<username>/<dataset-name>/)
# ---------------------------------------------------------------
PLANT_LEAVES_BASE = None
MEDICINAL_BASE = None

# Auto-find Plant Leaves dataset
for p in glob.glob('/kaggle/input/**/Plants_2', recursive=True):
    PLANT_LEAVES_BASE = p
    break
if not PLANT_LEAVES_BASE:
    for p in glob.glob('/kaggle/input/**/plant-leaves*', recursive=True):
        if os.path.isdir(p):
            PLANT_LEAVES_BASE = p
            break

# Auto-find Indian Medicinal Leaves dataset
for p in glob.glob('/kaggle/input/**/Indian Medicinal Leaves Image Datasets', recursive=True):
    MEDICINAL_BASE = p
    break
if not MEDICINAL_BASE:
    for p in glob.glob('/kaggle/input/**/indian-medicinal*', recursive=True):
        if os.path.isdir(p):
            MEDICINAL_BASE = p
            break

print(f"Plant Leaves base: {PLANT_LEAVES_BASE}")
print(f"Medicinal base:    {MEDICINAL_BASE}")

# ---------------------------------------------------------------
# Species name mapping: source folder pattern -> target species
# Plant Leaves has "{Species} healthy/diseased (PXa/PXb)" format
# We merge healthy+diseased into one species folder
# ---------------------------------------------------------------
PLANT_LEAVES_MAPPING = {
    # source substring (case-insensitive match) -> target species
    'Mango':             'Mango',
    'Gauva':             'Guava',       # note: source spells it "Gauva"
    'Guava':             'Guava',
    'Alstonia':          'AlstoniaScholaris',
    'Jamun':             'Jamun',
    'Arjun':             'Arjun',
    'Pongamia':          'Pongamia',
    'Jatropha':          'Jatropha',
    'Lemon':             'Lemon',
    'Pomegranate':       'Pomegranate',
    'Chinar':            'Chinar',
    'Basil':             'Basil',
}

# Medicinal plant dataset mapping
MEDICINAL_PLANT_MAPPING = {
    'Mango':      'Mango',
    'Neem':       'Neem',
    'Arjun':      'Arjun',
    'Curry_Leaf': 'CurryLeaf',
    'Curry Leaf': 'CurryLeaf',
}

# Medicinal leaf dataset mapping
MEDICINAL_LEAF_MAPPING = {
    'Guava':      'Guava',
    'Eucalyptus': 'Eucalyptus',
    'Neem':       'Neem',
}

def match_species(folder_name, mapping):
    """Match a folder name to a target species using the mapping dict."""
    for key, target in mapping.items():
        if key.lower() in folder_name.lower():
            return target
    return None

def copy_images_from_dir(src_dir, target_species, raw_subdir):
    """Copy all images from src_dir into data/raw/{raw_subdir}/{target_species}/"""
    dst = f'data/raw/{raw_subdir}/{target_species}'
    os.makedirs(dst, exist_ok=True)
    count = 0
    for ext in ('*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG'):
        for img in glob.glob(os.path.join(src_dir, ext)):
            shutil.copy2(img, dst)
            count += 1
    return count

# Clean old raw data
if os.path.exists('data/raw'):
    shutil.rmtree('data/raw')
os.makedirs('data/raw/plant_leaves', exist_ok=True)
os.makedirs('data/raw/medicinal_leaves', exist_ok=True)

total_copied = 0

# --- 1) Plant Leaves dataset (has train/valid/test splits with healthy/diseased) ---
if PLANT_LEAVES_BASE:
    print("\n📦 Processing Plant Leaves dataset...")
    for split in ['train', 'valid', 'test', 'Train', 'Valid', 'Test']:
        split_dir = os.path.join(PLANT_LEAVES_BASE, split)
        if not os.path.isdir(split_dir):
            continue
        for folder in sorted(os.listdir(split_dir)):
            folder_path = os.path.join(split_dir, folder)
            if not os.path.isdir(folder_path):
                continue
            species = match_species(folder, PLANT_LEAVES_MAPPING)
            if species:
                n = copy_images_from_dir(folder_path, species, 'plant_leaves')
                if n > 0:
                    print(f"   ✅ {folder:45s} → {species:20s} ({n} images)")
                total_copied += n
            else:
                # Check for images anyway
                imgs = glob.glob(os.path.join(folder_path, '*.[jJpP][pPnN][gGgG]'))
                if imgs:
                    print(f"   ⚠️  UNMAPPED: {folder} ({len(imgs)} images)")
else:
    print("❌ Plant Leaves dataset NOT FOUND")

# --- 2) Indian Medicinal - Plant dataset ---
if MEDICINAL_BASE:
    print("\n📦 Processing Medicinal Plant dataset...")
    plant_dir = os.path.join(MEDICINAL_BASE, 'Medicinal plant dataset')
    if not os.path.isdir(plant_dir):
        # Try other possible names
        for name in os.listdir(MEDICINAL_BASE):
            if 'plant' in name.lower() and os.path.isdir(os.path.join(MEDICINAL_BASE, name)):
                plant_dir = os.path.join(MEDICINAL_BASE, name)
                break

    if os.path.isdir(plant_dir):
        for folder in sorted(os.listdir(plant_dir)):
            folder_path = os.path.join(plant_dir, folder)
            if not os.path.isdir(folder_path):
                continue
            species = match_species(folder, MEDICINAL_PLANT_MAPPING)
            if species:
                n = copy_images_from_dir(folder_path, species, 'medicinal_leaves')
                if n > 0:
                    print(f"   ✅ {folder:45s} → {species:20s} ({n} images)")
                total_copied += n
    else:
        print(f"   ⚠️  Medicinal plant dataset dir not found in {MEDICINAL_BASE}")

    # --- 3) Indian Medicinal - Leaf dataset ---
    print("\n📦 Processing Medicinal Leaf dataset...")
    leaf_dir = os.path.join(MEDICINAL_BASE, 'Medicinal Leaf dataset')
    if not os.path.isdir(leaf_dir):
        for name in os.listdir(MEDICINAL_BASE):
            if 'leaf' in name.lower() and os.path.isdir(os.path.join(MEDICINAL_BASE, name)):
                leaf_dir = os.path.join(MEDICINAL_BASE, name)
                break

    if os.path.isdir(leaf_dir):
        for folder in sorted(os.listdir(leaf_dir)):
            folder_path = os.path.join(leaf_dir, folder)
            if not os.path.isdir(folder_path):
                continue
            species = match_species(folder, MEDICINAL_LEAF_MAPPING)
            if species:
                n = copy_images_from_dir(folder_path, species, 'medicinal_leaves')
                if n > 0:
                    print(f"   ✅ {folder:45s} → {species:20s} ({n} images)")
                total_copied += n
    else:
        print(f"   ⚠️  Medicinal leaf dataset dir not found in {MEDICINAL_BASE}")
else:
    print("❌ Medicinal Leaves dataset NOT FOUND")

print(f"\n{'='*60}")
print(f"📋 Total images copied: {total_copied}")
print(f"{'='*60}")

# --- Also list ALL folders found in kaggle input for debugging ---
print("\n🔍 Full /kaggle/input tree (depth=3):")
os.system("find /kaggle/input -maxdepth 4 -type d | head -80")


# =====================================================================
# CELL 2.5: VERIFY COUNTS — Review before training!
# =====================================================================
import os

print("=" * 65)
print("📊  STEP 2: PER-SPECIES IMAGE COUNT VERIFICATION")
print("=" * 65)

grand_total = 0
species_counts = {}

for raw_subdir in ['plant_leaves', 'medicinal_leaves']:
    raw_path = f'data/raw/{raw_subdir}'
    if not os.path.isdir(raw_path):
        continue
    print(f"\n📁 data/raw/{raw_subdir}/")
    for species_dir in sorted(os.listdir(raw_path)):
        full = os.path.join(raw_path, species_dir)
        if not os.path.isdir(full):
            continue
        count = len([f for f in os.listdir(full)
                     if f.lower().endswith(('.jpg','.jpeg','.png'))])
        species_counts[species_dir] = species_counts.get(species_dir, 0) + count
        tag = "✅" if count >= 20 else "⚠️"
        print(f"   {tag} {species_dir:25s} → {count:5d} images")

print(f"\n{'='*65}")
print(f"📊 MERGED TOTALS (what config.yaml species will get):")
print(f"{'='*65}")
for sp in sorted(species_counts.keys()):
    total = species_counts[sp]
    grand_total += total
    tag = "✅" if total >= 20 else "⚠️ LOW"
    print(f"   {tag} {sp:25s} → {total:5d} images (merged)")

print(f"\n   GRAND TOTAL: {grand_total} images across {len(species_counts)} species")
print(f"{'='*65}")
print("\n⏸️  REVIEW THESE COUNTS. If they look good, proceed to Cell 3.")
print("   Ideal: 20+ images per species, 14 species total")
print("   (Peepal, Banyan, Teak, Sal, Mahogany should NOT appear — no data)")


# =====================================================================
# CELL 3: UPDATE CONFIG, PREPARE, SPLIT, TRAIN (run AFTER approving counts)
# =====================================================================
import yaml, os

# --- Update config.yaml: drop species with no data ---
with open('configs/config.yaml', 'r') as f:
    cfg = yaml.safe_load(f)

# New class_map with ONLY species that have real data
cfg['class_map'] = {
    'Mango':            ['plant_leaves/Mango', 'medicinal_leaves/Mango'],
    'Jamun':            ['plant_leaves/Jamun'],
    'Guava':            ['plant_leaves/Guava', 'medicinal_leaves/Guava'],
    'Arjun':            ['plant_leaves/Arjun', 'medicinal_leaves/Arjun'],
    'Chinar':           ['plant_leaves/Chinar'],
    'Pongamia':         ['plant_leaves/Pongamia'],
    'Jatropha':         ['plant_leaves/Jatropha'],
    'Lemon':            ['plant_leaves/Lemon'],
    'Pomegranate':      ['plant_leaves/Pomegranate'],
    'Basil':            ['plant_leaves/Basil'],
    'AlstoniaScholaris':['plant_leaves/AlstoniaScholaris'],
    'Neem':             ['medicinal_leaves/Neem'],
    'CurryLeaf':        ['medicinal_leaves/CurryLeaf'],
    'Eucalyptus':       ['medicinal_leaves/Eucalyptus'],
    # DROPPED: Peepal, Banyan, Teak, Sal, Mahogany (no data in either dataset)
}

with open('configs/config.yaml', 'w') as f:
    yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)

print(f"✅ Updated config.yaml: {len(cfg['class_map'])} species")
for sp in sorted(cfg['class_map'].keys()):
    print(f"   → {sp}: {cfg['class_map'][sp]}")

# --- Run the full pipeline ---
print("\n" + "="*60)
print("🔄 Step 1: Preparing dataset (merge + deduplicate)...")
print("="*60)
os.system('python data/prepare_dataset.py')

print("\n" + "="*60)
print("🔄 Step 2: Creating stratified train/val/test splits...")
print("="*60)
os.system('python data/make_splits.py')

print("\n" + "="*60)
print("🚀 Step 3: Training EfficientNet-B3...")
print("="*60)
os.system('python training/train.py')


# =====================================================================
# CELL 4: EVALUATE & METRICS
# =====================================================================
print('=' * 60)
print('📊 Running Evaluation on Test Set...')
os.system('python evaluation/evaluate.py --checkpoint outputs/checkpoints/best.pt')

print('\n💾 Copying results to Kaggle outputs...')
os.system('cp outputs/plots/test_confusion_matrix.png /kaggle/working/ 2>/dev/null || true')
os.system('cp outputs/plots/test_confusion_matrix_normalized.png /kaggle/working/ 2>/dev/null || true')
os.system('cp outputs/logs/test_metrics.json /kaggle/working/ 2>/dev/null || true')
os.system('cp outputs/checkpoints/best.pt /kaggle/working/ 2>/dev/null || true')
os.system('cp outputs/checkpoints/last.pt /kaggle/working/ 2>/dev/null || true')
print('✅ Done! Check /kaggle/working/ for outputs.')
