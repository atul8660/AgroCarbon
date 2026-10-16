import sys
import datetime
import torch
import cv2

print("=== MODULE 1: Species Classification ===")
try:
    ckpt1 = torch.load(r'module1_species_classification/tree_species_classifier/outputs_pilot/checkpoints/best_13_species_model.pt', map_location='cpu', weights_only=False)
    print(type(ckpt1))
    
    sys.path.insert(0, 'backend')
    from app.pipeline.blackbox_adapters import Module1Adapter
    adapter1 = Module1Adapter()
    img1 = cv2.imread(r'module1_species_classification/tree_species_classifier/data/processed_pilot/Mango/Mango_000000.jpg')
    if img1 is None:
        print("ERROR: Could not load Mango image")
    else:
        res1 = adapter1.predict(img1)
        print(f"Predicted species: {res1.get('species')}, confidence: {res1.get('confidence')}")
except Exception as e:
    print(f"Module 1 Error: {e}")

sys.path.pop(0)

print("\n=== MODULE 2: Health Classification ===")
try:
    ckpt2 = torch.load(r'module2_health_classification/tree_health_classifier/outputs/checkpoints/best.pt', map_location='cpu', weights_only=False)
    print('num_classes:', ckpt2.get('num_classes'))
    print('label_map:', ckpt2.get('label_map'))
    
    sys.path.insert(0, 'backend')
    from app.pipeline.blackbox_adapters import Module2Adapter
    adapter2 = Module2Adapter()
    img2 = cv2.imread(r'module1_species_classification/tree_species_classifier/data/processed_pilot/Mango/Mango_000000.jpg')
    res2 = adapter2.predict(img2)
    print(f"Predicted health: {res2.get('health_status')}, confidence: {res2.get('confidence')}")
except Exception as e:
    print(f"Module 2 Error: {e}")

sys.path.pop(0)

print("\n=== MODULE 3: Tree Detection ===")
try:
    ckpt3 = torch.load(r'module3_tree_detection/tree_detection/outputs/checkpoints/best.pt', map_location='cpu', weights_only=False)
    print('Class names:', ckpt3['model'].names)
    
    sys.path.insert(0, 'backend')
    from app.pipeline.blackbox_adapters import Module3Adapter
    adapter3 = Module3Adapter()
    img3 = cv2.imread('test_tree.jpg')
    if img3 is None:
        print("ERROR: test_tree.jpg not found for Module 3")
    else:
        res3 = adapter3.predict(img3)
        print(f"Crop shape: {res3.get('tree_crop').shape if res3.get('tree_crop') is not None else 'None'}")
except Exception as e:
    print(f"Module 3 Error: {e}")

sys.path.pop(0)

print("\n=== MODULE 4: Tree Segmentation ===")
try:
    ckpt4 = torch.load(r'module4_tree_segmentation/tree_segmentation/outputs/checkpoints/best.pt', map_location='cpu', weights_only=False)
    print('epoch:', ckpt4.get('epoch'))
    print('num_classes:', ckpt4.get('num_classes'))
    
    sys.path.insert(0, 'backend')
    from app.pipeline.blackbox_adapters import Module4Adapter
    adapter4 = Module4Adapter()
    if img3 is not None and res3.get('tree_crop') is not None:
        res4 = adapter4.predict(res3.get('tree_crop'))
        print(f"coverage: {res4.get('canopy_coverage_percent')}")
        print(f"quality: {res4.get('quality_score')}")
        print(f"ready_for_module5: {res4.get('ready_for_module5')}")
    else:
        print("Skipped M4 inference due to missing M3 crop")
except Exception as e:
    print(f"Module 4 Error: {e}")

sys.path.pop(0)

print("\n=== MODULE 6: Carbon Credit Value ===")
try:
    sys.path.insert(0, '.')
    from module6_carbon_analytics.api.engine import CarbonEngine
    from module6_carbon_analytics.api.schemas import BiomassInput
    
    engine = CarbonEngine(config_path='module6_carbon_analytics/configs/config.yaml')
    data = BiomassInput(predicted_biomass_kg=341.81, biomass_confidence=0.85, species='Mango', species_confidence=0.9757, health='Healthy', health_confidence=0.9185, timestamp=datetime.datetime.now())
    result = engine.process(data, pdf_path='verify.pdf', csv_path='verify.csv')
    print(f"Carbon Engine Result: credits={result.estimated_credits} value={result.estimated_value_inr}")
except Exception as e:
    print(f"Module 6 Error: {e}")
