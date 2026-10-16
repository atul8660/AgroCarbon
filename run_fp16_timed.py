import sys, time, os
sys.path.insert(0, '.')
import cv2
from backend.app.pipeline.blackbox_adapters import Module3Adapter, Module4Adapter

img = cv2.imread(r'module3_tree_detection/tree_detection/test_tree.jpg')
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

m3 = Module3Adapter()
cropped = m3.predict(img_rgb)

Module4Adapter._CHECKPOINT_REL = os.path.join(
    'module4_tree_segmentation', 'tree_segmentation',
    'outputs', 'checkpoints', 'best_fp16.pt'
)
m4 = Module4Adapter()
start = time.time()
r = m4.predict(cropped)
t = time.time() - start
print('=== FP16 (pruned) ===')
print('Coverage:', r.get('coverage'))
print('Quality:', r.get('quality'))
print('Inference time (seconds):', round(t, 2))
