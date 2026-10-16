#!/usr/bin/env python3
"""
export_onnx.py
----------------
Exports a trained YOLOv11 checkpoint to ONNX for deployment, using
ultralytics' built-in exporter (handles NMS-compatible graph construction
correctly, unlike a hand-rolled torch.onnx.export for detection heads), then
verifies the exported model loads and runs via onnxruntime.

Usage:
    python export/export_onnx.py --checkpoint outputs/checkpoints/best.pt \
        --output outputs/model.onnx
"""

import argparse
import logging
import shutil
import sys
from pathlib import Path

import onnx
import onnxruntime as ort
import yaml

sys.path.append(str(Path(__file__).resolve().parents[1]))

from models.model import load_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
)
logger = logging.getLogger("export_onnx")


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def export_to_onnx(model, image_size: int, output_path: str, opset_version: int = 17) -> Path:
    exported_path = model.export(
        format="onnx",
        imgsz=image_size,
        opset=opset_version,
        dynamic=True,          # dynamic batch axis
        simplify=True,
    )
    exported_path = Path(exported_path)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if exported_path.resolve() != output_path.resolve():
        shutil.copy2(exported_path, output_path)
    logger.info(f"ONNX model exported to: {output_path}")

    onnx_model = onnx.load(str(output_path))
    onnx.checker.check_model(onnx_model)
    logger.info("ONNX model structure check passed.")
    return output_path


def verify_onnx_runs(onnx_path: Path, image_size: int) -> bool:
    import numpy as np

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    dummy_input = np.random.rand(1, 3, image_size, image_size).astype("float32")

    try:
        outputs = session.run(None, {input_name: dummy_input})
        logger.info(
            f"Verification PASSED — ONNX model ran successfully, "
            f"output shape(s): {[o.shape for o in outputs]}"
        )
        return True
    except Exception as e:
        logger.warning(f"Verification FAILED — {e}")
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output", type=str, default="outputs/model.onnx")
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()

    cfg = load_config(args.config)
    image_size = cfg["model"]["input_size"]

    model = load_model(args.checkpoint)
    onnx_path = export_to_onnx(model, image_size, args.output, opset_version=args.opset)
    verify_onnx_runs(onnx_path, image_size)

    logger.info(
        f"Export complete. Architecture: {cfg['model']['architecture']} | "
        f"Input size: {image_size}x{image_size} | Classes: {cfg['classes']}"
    )


if __name__ == "__main__":
    main()
