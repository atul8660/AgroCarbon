"""
Download best.pt from your Kaggle notebook output.

SETUP (one-time):
1. Go to https://www.kaggle.com → Your Profile → Settings → API → "Create New Token"
2. It downloads kaggle.json — place it at: C:/Users/<YourName>/.kaggle/kaggle.json
3. pip install kaggle

HOW TO RUN:
    python download_best_pt.py
"""

import os
import shutil
import subprocess
import sys

# ─────────────────────────────────────────────
# CONFIGURE THESE TWO VALUES
KAGGLE_USERNAME = "YOUR_KAGGLE_USERNAME"   # e.g. "atulkumar"
NOTEBOOK_SLUG   = "YOUR_NOTEBOOK_SLUG"     # e.g. "kaggle-module1-training"
# ─────────────────────────────────────────────

# Where to save the file locally
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
DEST_PT    = os.path.join(OUTPUT_DIR,
             "tree_species_classifier", "outputs_pilot", "checkpoints", "best.pt")

KERNEL_ID = f"{KAGGLE_USERNAME}/{NOTEBOOK_SLUG}"

def run(cmd):
    print(f">> {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr)
    return result.returncode

def main():
    # 1. Check kaggle is installed
    try:
        import kaggle  # noqa: F401
    except ImportError:
        print("Installing kaggle package...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "kaggle", "-q"])

    # 2. Pull notebook output into a temp folder
    temp_dir = os.path.join(OUTPUT_DIR, "_kaggle_output_tmp")
    os.makedirs(temp_dir, exist_ok=True)

    print(f"\nDownloading output from kernel: {KERNEL_ID}")
    rc = run(f'kaggle kernels output "{KERNEL_ID}" --path "{temp_dir}"')

    if rc != 0:
        print("\n❌ Download failed. Check your KAGGLE_USERNAME and NOTEBOOK_SLUG above.")
        print("   Make sure your kaggle.json is at C:/Users/<YourName>/.kaggle/kaggle.json")
        return

    # 3. Find best.pt in the downloaded output
    pt_file = None
    for root, _, files in os.walk(temp_dir):
        for f in files:
            if f == "best.pt":
                pt_file = os.path.join(root, f)
                break

    if not pt_file:
        print("\n⚠️  best.pt not found in kernel output. Files downloaded:")
        for root, _, files in os.walk(temp_dir):
            for f in files:
                print("  ", os.path.join(root, f))
        return

    # 4. Copy to the correct location in the project
    os.makedirs(os.path.dirname(DEST_PT), exist_ok=True)
    shutil.copy2(pt_file, DEST_PT)
    size_mb = os.path.getsize(DEST_PT) / (1024 * 1024)
    print(f"\n✅ best.pt saved to:\n   {DEST_PT}")
    print(f"   Size: {size_mb:.1f} MB")

    # 5. Also copy to backend models directory
    backend_models = os.path.join(
        OUTPUT_DIR, "..", "backend", "models", "module1_species_classifier.pt"
    )
    backend_models = os.path.normpath(backend_models)
    if os.path.isdir(os.path.dirname(backend_models)):
        shutil.copy2(pt_file, backend_models)
        print(f"✅ Also copied to backend:\n   {backend_models}")

    # 6. Cleanup temp folder
    shutil.rmtree(temp_dir)
    print("\nDone!")

if __name__ == "__main__":
    main()
