import os
import sys
import numpy as np
import cv2
import torch
import warnings

# Suppress warnings for clean output
warnings.filterwarnings('ignore')

# Set paths to allow imports across modules
sys.path.append(os.path.abspath('module5_biomass_estimation'))
sys.path.append(os.path.abspath('module6_carbon_analytics'))

from module5_biomass_estimation.inference.predict import BiomassPredictor
from module6_carbon_analytics.api.engine import CarbonEngine
from module6_carbon_analytics.api.schemas import BiomassInput
from module6_carbon_analytics.exports.pdf_generator import generate_pdf_report

def main():
    print("--- Capstone AI Pipeline Integration Test ---")
    
    # 1. Ensure checkpoint exists
    checkpoint_path = "module5_biomass_estimation/outputs/checkpoints/best.pt"
    if not os.path.exists(checkpoint_path):
        print(f"Error: Could not find model checkpoint at {checkpoint_path}")
        return
        
    print("[1] Loading Biomass Estimator Model (ConvNeXt-Tiny) ...")
    try:
        predictor = BiomassPredictor(
            config_path="module5_biomass_estimation/configs/config.yaml",
            checkpoint_path=checkpoint_path
        )
    except Exception as e:
        print(f"Error loading predictor: {e}")
        return
    
    print("[2] Creating synthetic test data (mimicking M1-M4 outputs)...")
    # Fake image (256x256 RGB)
    fake_img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    cv2.imwrite("test_tree_integration.jpg", fake_img)
    
    # Fake geometric properties
    fake_mask = np.ones((256, 256), dtype=np.uint8)
    fake_polygon = [[10, 10], [10, 50], [50, 50], [50, 10]]
    
    print("[3] Running Biomass AI Inference (Module 5)...")
    result = predictor.predict(
        image_path="test_tree_integration.jpg",
        tree_mask=fake_mask,
        canopy_mask=fake_mask,
        trunk_mask=fake_mask,
        polygon=fake_polygon,
        species_id=1,  # Mock label for Neem
        health_id=0    # Mock label for Healthy
    )
    
    biomass_kg = result["predicted_biomass_kg"]
    print(f"    -> Neural Network predicted AGB: {biomass_kg} kg")
    
    print("[4] Executing IPCC Carbon Analytics Engine (Module 6)...")
    engine = CarbonEngine("module6_carbon_analytics/configs/config.yaml") 
    
    # Pack the input exactly as the FastAPI endpoint expects in production
    input_data = BiomassInput(
        predicted_biomass_kg=biomass_kg,
        species="Neem",
        health="Healthy",
        biomass_confidence=result["confidence"],
        species_confidence=0.92,
        health_confidence=0.88,
        gps_coordinates="19.0760 N, 72.8777 E"
    )
    
    pdf_path = os.path.abspath("Final_Carbon_Report.pdf")
    csv_path = os.path.abspath("Carbon_Log.csv")
    
    report = engine.process(input_data, pdf_path, csv_path)
    
    print("[5] Generating Professional PDF Report...")
    generate_pdf_report(report, pdf_path)
    
    print(f"\n✅ SUCCESS! Pipeline completed end-to-end.")
    print(f"👉 Please open {os.path.abspath(pdf_path)} to see your generated Carbon Impact Report!")
    
if __name__ == "__main__":
    main()
