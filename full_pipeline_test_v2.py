"""
Full end-to-end pipeline test: photo(s) in -> rupee carbon credit value out.

Chains all 6 real backend adapters in sequence:
  Module 3 (detect)  -> Module 4 (segment) -> Module 1 (species)
  -> Module 2 (health) -> Module 5 (biomass) -> Module 6 (carbon/INR)

Run this from the project root:
  D:\\Capston Zip Files\\tree_carbon_assessment_system

Usage:
  python full_pipeline_test.py <whole_tree_photo_path> <leaf_photo_path>
"""

import sys
import os
import numpy as np
import cv2

sys.path.insert(0, '.')

from backend.app.pipeline.blackbox_adapters import (
    Module1Adapter, Module2Adapter, Module3Adapter,
    Module4Adapter, Module5Adapter, Module6Adapter,
)


def main(tree_photo_path: str, leaf_photo_path: str):
    print("=" * 70)
    print("FULL PIPELINE TEST")
    print("=" * 70)

    # ---- Load images ----
    tree_img_bgr = cv2.imread(tree_photo_path)
    if tree_img_bgr is None:
        raise FileNotFoundError(f"Could not read tree photo: {tree_photo_path}")
    tree_img_rgb = cv2.cvtColor(tree_img_bgr, cv2.COLOR_BGR2RGB)

    leaf_img_bgr = cv2.imread(leaf_photo_path)
    if leaf_img_bgr is None:
        raise FileNotFoundError(f"Could not read leaf photo: {leaf_photo_path}")
    leaf_img_rgb = cv2.cvtColor(leaf_img_bgr, cv2.COLOR_BGR2RGB)

    # ---- Module 3: Tree Detection ----
    print("\n[Module 3] Detecting tree...")
    m3 = Module3Adapter()
    cropped = m3.predict(tree_img_rgb)
    print(f"  Crop shape: {cropped.shape}")

    # ---- Module 4: Segmentation ----
    print("\n[Module 4] Segmenting tree...")
    m4 = Module4Adapter()
    seg_result = m4.predict(cropped)
    coverage = seg_result.get("coverage")
    quality = seg_result.get("quality")
    ready = seg_result.get("ready_for_module5")
    print(f"  Coverage: {coverage}")
    print(f"  Quality: {quality}")
    print(f"  Ready for Module 5: {ready}")

    if not ready:
        print("\n  STOPPING: segmentation not ready for Module 5 (quality gate failed).")
        return

    # ---- Module 1: Species Classification ----
    print("\n[Module 1] Classifying species...")
    m1 = Module1Adapter()
    species, species_conf = m1.predict(leaf_img_rgb, filename=os.path.basename(leaf_photo_path))
    print(f"  Species: {species} ({species_conf*100:.2f}%)")

    # ---- Module 2: Health Classification ----
    print("\n[Module 2] Classifying health...")
    m2 = Module2Adapter()
    health, health_conf = m2.predict(leaf_img_rgb)
    print(f"  Health: {health} ({health_conf*100:.2f}%)")

    # ---- Module 5: Biomass Estimation ----
    print("\n[Module 5] Estimating biomass...")
    m5 = Module5Adapter(config_path=None, checkpoint_path=None)
    try:
        biomass_result = m5.predict(
            image_path=tree_photo_path,
            tree_mask=seg_result.get("tree_mask"),
            canopy_mask=seg_result.get("canopy_mask"),
            trunk_mask=seg_result.get("trunk_mask"),
            polygon=seg_result.get("polygon"),
            species=species,
            health=health,
        )
        print(f"  Result: {biomass_result}")
    except ValueError as e:
        print(f"  SPECIES NOT SUPPORTED: {e}")
        print("\n" + "=" * 70)
        print("FINAL RESULT: Species identified, but carbon calculation")
        print("unavailable for this species in the current pilot.")
        print("=" * 70)
        return

    # ---- Module 6: Carbon Credit Value (INR) ----
    print("\n[Module 6] Calculating carbon credit value...")
    os.makedirs("pipeline_test_output", exist_ok=True)
    m6 = Module6Adapter(config_path="module6_carbon_analytics/configs/config.yaml")
    carbon_result = m6.process(
        biomass_kg=biomass_result.get("agb_kg"),
        species=species,
        health=health,
        confs={
            "biomass": 0.85,
            "species": species_conf,
            "health": health_conf,
        },
        pdf_path="pipeline_test_output/test_report.pdf",
        csv_path="pipeline_test_output/test_report.csv",
    )
    print(f"  Result: {carbon_result}")

    print("\n" + "=" * 70)
    print("FINAL RESULT")
    print("=" * 70)
    print(carbon_result)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python full_pipeline_test.py <tree_photo_path> <leaf_photo_path>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
