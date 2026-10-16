import sys
sys.path.insert(0, '.')
import numpy as np
import cv2
from backend.app.pipeline.blackbox_adapters import Module3Adapter, Module4Adapter

m3 = Module3Adapter()
m4 = Module4Adapter()

img_path = r'D:\Capston Zip Files\tree_carbon_assessment_system\module3_tree_detection\tree_detection\outputs\crops\test_tree.jpg'
image = cv2.imread(img_path)
if image is None:
    print('Failed to read image:', img_path)
    sys.exit(1)

print('--- MODULE 3: DETECTION ---')
crop = m3.predict(image)
print(f'Detected tree! Crop dimensions: {crop.shape[1]}x{crop.shape[0]} (WxH)')

print('\n--- MODULE 4: SEGMENTATION ---')
seg_result = m4.predict(crop)
print(f'Coverage: {seg_result.get("coverage", 0.0)*100:.2f}%')
print(f'Quality: {seg_result.get("quality", 0.0):.4f}')
print(f'Ready for Module 5: {seg_result.get("ready_for_module5", False)}')
