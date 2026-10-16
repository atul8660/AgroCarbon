import torch
import os
import sys

root = r'd:\Capston Zip Files\tree_carbon_assessment_system'
os.chdir(root)
ckpt_path = r'module4_tree_segmentation/tree_segmentation/outputs/checkpoints/best.pt'
out_path = r'module4_tree_segmentation/tree_segmentation/outputs/checkpoints/best_fp16.pt'

print(f"Original size: {os.path.getsize(ckpt_path) / (1024*1024):.2f} MB")

# Load and cast weights
ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
sd = ckpt['model_state_dict']
for k, v in sd.items():
    if v.is_floating_point():
        sd[k] = v.half()
ckpt['model_state_dict'] = sd
torch.save(ckpt, out_path)

print(f"FP16 size: {os.path.getsize(out_path) / (1024*1024):.2f} MB")

# Replace Module4Adapter logic safely to test
print("\n--- Testing FP16 Checkpoint via Adapter ---")
import backend.app.pipeline.blackbox_adapters as adapters
# force the adapter to use the new fp16 checkpoint
adapters.Module4Adapter._CHECKPOINT_REL = os.path.join(
    "module4_tree_segmentation", "tree_segmentation",
    "outputs", "checkpoints", "best_fp16.pt"
)

import cv2
img_bgr = cv2.imread(r'module3_tree_detection/tree_detection/test_tree.jpg')
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

m3 = adapters.Module3Adapter()
cropped_rgb = m3.predict(img_rgb)

m4 = adapters.Module4Adapter()
res = m4.predict(cropped_rgb)
print(f"FP16 Coverage: {res.get('coverage')}")
print(f"FP16 Quality: {res.get('quality')}")
print(f"FP16 Ready for Module 5: {res.get('ready_for_module5')}")
