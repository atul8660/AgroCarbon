import sys
import cv2
sys.path.insert(0, '.')
from backend.app.pipeline.blackbox_adapters import Module1Adapter, Module3Adapter, Module5Adapter

img_path = r'C:\Users\atull\Downloads\test_tree.jpg'
print('\n?? STARTING CAPSTONE PIPELINE ??\n')

# 1. Image Loading
print('Loading image...')
img_rgb = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)

# 2. Module 1: Species Classification
print('--> [Module 1] Classifying Species...')
m1 = Module1Adapter()
# FIXED: Passing the actual image array (img_rgb), not the string path
m1_res = m1.predict(img_rgb)

# Safely extract species whether it returns a dictionary or a direct string/tuple
if isinstance(m1_res, dict):
    species = m1_res.get('species', 'Unknown')
elif isinstance(m1_res, tuple):
    species = m1_res[0]
else:
    species = str(m1_res)

print(f'    Result: {species} Detected')

# 3. Module 3: Tree Detection
print('--> [Module 3] Detecting Tree & Cropping...')
m3 = Module3Adapter()
crop = m3.predict(img_rgb)
print(f'    Result: Tree localized successfully! (Crop size: {crop.shape})')

# 4. Module 5: Biomass & Carbon Calculation
print('--> [Module 5] Calculating Biomass & Carbon...')
m5 = Module5Adapter()
final_res = m5.predict(
    image_path=img_path, 
    tree_mask=None, canopy_mask=None, trunk_mask=None, polygon=None, 
    species=species, health='Healthy'
)

# 5. Final Output
print('\n? FINAL CARBON ASSESSMENT REPORT ?')
print('-' * 45)
if isinstance(final_res, dict):
    for key, value in final_res.items():
        print(f'{key.replace("_", " ").title().rjust(20)} : {value}')
else:
    print(final_res)
print('-' * 45 + '\n')
