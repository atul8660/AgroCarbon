import sys
sys.path.insert(0, '.')
import numpy as np
import cv2
from backend.app.pipeline.blackbox_adapters import Module4Adapter

m4 = Module4Adapter()

img_path = r'D:\Capston Zip Files\tree_carbon_assessment_system\module3_tree_detection\tree_detection\outputs\crops\test_tree.jpg'
crop = cv2.imread(img_path)

with open('m4_clean.txt', 'w') as f:
    f.write('--- MODULE 4: SEGMENTATION ---\n')
    try:
        seg_result = m4.predict(crop)
        f.write(f'Coverage: {seg_result.get("coverage", 0.0)*100:.2f}%\n')
        f.write(f'Quality: {seg_result.get("quality", 0.0):.4f}\n')
        f.write(f'Ready for Module 5: {seg_result.get("ready_for_module5", False)}\n')
    except Exception as e:
        f.write(f'Exception: {e}')
