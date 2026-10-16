import sys, time, os
sys.path.insert(0, '.')
import numpy as np
import cv2
from backend.app.pipeline.blackbox_adapters import Module3Adapter, Module4Adapter

img = cv2.imread(r'module3_tree_detection/tree_detection/test_tree.jpg')
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

m3 = Module3Adapter()
cropped = m3.predict(img_rgb)

# Test ORIGINAL fp32 checkpoint, timed
m4_original = Module4Adapter()
start = time.time()
result_original = m4_original.predict(cropped)
time_original = time.time() - start
print('=== ORIGINAL (fp32) ===')
print('Coverage:', result_original.get('coverage'))
print('Quality:', result_original.get('quality'))
print('Inference time (seconds):', round(time_original, 2))

print('')

# modify adapter path temporally
Module4Adapter._CHECKPOINT_REL = os.path.join(
    "module4_tree_segmentation", "tree_segmentation",
    "outputs", "checkpoints", "best_fp16.pt"
)

# Test fp16 checkpoint, timed
m4_fp16 = Module4Adapter()
start = time.time()
result_fp16 = m4_fp16.predict(cropped)
time_fp16 = time.time() - start

print('=== PRUNED (fp16) ===')
print('Coverage:', result_fp16.get('coverage'))
print('Quality:', result_fp16.get('quality'))
print('Inference time (seconds):', round(time_fp16, 2))
