#!/usr/bin/env python3
"""
test_m2_adapter.py
-------------------
Direct test of Module2Adapter with the real health-classification model.
Loads a known diseased-leaf image and prints the predicted label + confidence.

Usage:
    python test_m2_adapter.py
"""

import sys
import os
import cv2

# Ensure project root is on path
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

# --- Test image path (known diseased leaf, previously verified) ---
TEST_IMAGE = r"C:\Users\atull\Downloads\0021_0001.JPG"


def main():
    print("=" * 60)
    print("Module2Adapter — Real Model Test")
    print("=" * 60)

    # 1. Verify test image exists
    if not os.path.isfile(TEST_IMAGE):
        print(f"ERROR: Test image not found: {TEST_IMAGE}")
        sys.exit(1)
    print(f"[OK] Test image found: {TEST_IMAGE}")

    # 2. Load the image as numpy array (simulating what the pipeline does)
    image = cv2.imread(TEST_IMAGE)
    if image is None:
        print(f"ERROR: cv2.imread failed on {TEST_IMAGE}")
        sys.exit(1)
    print(f"[OK] Image loaded: shape={image.shape}, dtype={image.dtype}")

    # 3. Instantiate Module2Adapter — this MUST load the real model or raise
    print("\n--- Initialising Module2Adapter (real model, no mock) ---")
    from backend.app.pipeline.blackbox_adapters import Module2Adapter

    try:
        adapter = Module2Adapter()
    except RuntimeError as e:
        print(f"FATAL: Module2Adapter failed to initialise:\n  {e}")
        sys.exit(1)
    print("[OK] Module2Adapter initialised with real checkpoint\n")

    # 4. Run prediction
    print("--- Running predict() ---")
    label, confidence = adapter.predict(image)

    print(f"\n{'='*60}")
    print(f"  Predicted label : {label}")
    print(f"  Confidence      : {confidence:.4f} ({confidence:.2%})")
    print(f"{'='*60}")

    # 5. Sanity checks
    assert isinstance(label, str), f"Label should be str, got {type(label)}"
    assert isinstance(confidence, float), f"Confidence should be float, got {type(confidence)}"
    assert label in ("Healthy", "Diseased", "Unknown"), f"Unexpected label: {label}"
    assert 0.0 <= confidence <= 1.0, f"Confidence out of range: {confidence}"

    print("\n[OK] All assertions passed — real model is working correctly.")


if __name__ == "__main__":
    main()
