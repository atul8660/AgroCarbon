#!/usr/bin/env python3
"""
export_onnx.py
----------------
Exports a trained checkpoint to ONNX for deployment (server or edge), and
verifies numerical equivalence against the original PyTorch model on random
input plus a real sample image if provided.

Usage:
    python export/export_onnx.py --checkpoint outputs/checkpoints/best.pt \
        --output outputs/model.onnx

    python export/export_onnx.py --checkpoint outputs/checkpoints/best.pt \
        --output outputs/model.onnx --verify_image /path/to/leaf.jpg
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
import yaml

sys.path.append(str(Path(__file__).resolve().parents[1]))

from data.dataset import get_eval_transforms
from models.model import build_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
)
logger = logging.getLogger("export_onnx")


def load_model_from_checkpoint(checkpoint_path: str):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = checkpoint["config"]
    num_classes = checkpoint["num_classes"]
    label_map = checkpoint["label_map"]

    model = build_model(cfg, num_classes)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, cfg, num_classes, label_map


def export_to_onnx(model: torch.nn.Module, image_size: int, output_path: str,
                    opset_version: int = 17) -> None:
    dummy_input = torch.randn(1, 3, image_size, image_size)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "logits": {0: "batch_size"},
        },
    )
    logger.info(f"ONNX model exported to: {output_path}")

    # Structural validation
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    logger.info("ONNX model structure check passed.")


def verify_onnx_vs_pytorch(model: torch.nn.Module, onnx_path: str, image_size: int,
                            verify_image: str = None, atol: float = 1e-3) -> bool:
    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    if verify_image:
        import cv2
        image = cv2.imread(verify_image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        transforms = get_eval_transforms(image_size)
        tensor = transforms(image=image)["image"].unsqueeze(0)
        logger.info(f"Verifying against real image: {verify_image}")
    else:
        tensor = torch.randn(1, 3, image_size, image_size)
        logger.info("Verifying against random dummy input")

    with torch.no_grad():
        torch_output = model(tensor).numpy()

    onnx_output = session.run(None, {input_name: tensor.numpy()})[0]

    max_diff = np.abs(torch_output - onnx_output).max()
    is_close = np.allclose(torch_output, onnx_output, atol=atol)

    logger.info(f"Max absolute difference (PyTorch vs ONNX logits): {max_diff:.6f}")
    if is_close:
        logger.info(f"Verification PASSED (within atol={atol})")
    else:
        logger.warning(f"Verification FAILED — outputs diverge beyond atol={atol}")

    return is_close


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output", type=str, default="outputs/model.onnx")
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--verify_image", type=str, default=None,
                         help="Optional real image path for numerical verification")
    args = parser.parse_args()

    model, cfg, num_classes, label_map = load_model_from_checkpoint(args.checkpoint)
    image_size = cfg["data"]["image_size"]

    export_to_onnx(model, image_size, args.output, opset_version=args.opset)
    verify_onnx_vs_pytorch(model, args.output, image_size, verify_image=args.verify_image)

    # Save label map alongside the ONNX model so deployment doesn't need the
    # original PyTorch checkpoint to map class indices back to species names.
    label_map_path = Path(args.output).with_suffix(".labels.json")
    with open(label_map_path, "w") as f:
        json.dump({str(k): v for k, v in label_map.items()}, f, indent=2)
    logger.info(f"Label map saved to: {label_map_path}")

    logger.info(
        f"Export complete. Model: {cfg['model']['backbone']} | "
        f"Classes: {num_classes} | Input size: {image_size}x{image_size}"
    )


if __name__ == "__main__":
    main()
