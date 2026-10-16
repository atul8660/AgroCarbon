"""INT8 Dynamic Quantization for Module 4 SegFormer-B5 checkpoint."""
import sys, os, time, torch
os.environ['OMP_NUM_THREADS'] = '2'  # limit threads to prevent hanging

ROOT = r'd:\Capston Zip Files\tree_carbon_assessment_system'
os.chdir(ROOT)
sys.path.insert(0, '.')

m4_root = os.path.join(ROOT, 'module4_tree_segmentation', 'tree_segmentation')
if m4_root not in sys.path:
    sys.path.insert(0, m4_root)

ORIG = os.path.join(m4_root, 'outputs', 'checkpoints', 'best.pt')
INT8 = os.path.join(m4_root, 'outputs', 'checkpoints', 'best_int8.pt')

print(f'Original size: {os.path.getsize(ORIG)/(1024*1024):.2f} MB')

# ── Step 1: Load checkpoint, strip training-only keys ──────────────
ckpt = torch.load(ORIG, map_location='cpu', weights_only=False)
for k in ['optimizer_state_dict', 'scheduler_state_dict', 'scaler_state_dict', 'history']:
    ckpt.pop(k, None)

# ── Step 2: Build the SegFormer model ──────────────────────────────
from inference.segment import TreeSegmenter
config_path = os.path.join(m4_root, 'configs', 'config.yaml')
segmenter = TreeSegmenter(checkpoint_path=ORIG, config_path=config_path)

# Access the underlying nn.Module
model = segmenter.model  # the actual SegFormer nn.Module
model.eval()

# ── Step 3: Apply INT8 dynamic quantization to all Linear layers ───
quantized_model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)

# Save quantized state dict back into checkpoint
ckpt['model_state_dict'] = quantized_model.state_dict()
torch.save(ckpt, INT8)
print(f'INT8 size: {os.path.getsize(INT8)/(1024*1024):.2f} MB')

# ── Step 4: Timed inference using Module4Adapter pointed at INT8 ───
print('\n--- Timed inference with INT8 checkpoint ---')
from backend.app.pipeline.blackbox_adapters import Module3Adapter, Module4Adapter
import cv2

img = cv2.imread('module3_tree_detection/tree_detection/test_tree.jpg')
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

m3 = Module3Adapter()
cropped = m3.predict(img_rgb)
print(f'Crop shape: {cropped.shape}')

Module4Adapter._CHECKPOINT_REL = os.path.join(
    'module4_tree_segmentation', 'tree_segmentation',
    'outputs', 'checkpoints', 'best_int8.pt'
)
m4 = Module4Adapter()
start = time.time()
r = m4.predict(cropped)
t = time.time() - start

print('=== INT8 RESULTS ===')
print('Coverage:', r.get('coverage'))
print('Quality:', r.get('quality'))
print('Ready:', r.get('ready_for_module5'))
print('Inference time (seconds):', round(t, 2))
