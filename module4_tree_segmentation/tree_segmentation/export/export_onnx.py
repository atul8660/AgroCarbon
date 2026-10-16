#!/usr/bin/env python3
"""
export_onnx.py
----------------
Exports a trained SegFormer-B5 checkpoint to ONNX for deployment, and
verifies numerical equivalence against the original PyTorch model.

Note: this exports the SegFormer production model only. SAM2 (used for
pseudo-labeling / optional inference fallback) is not exported here — see
README section 2 for why it isn't the served model.

Usage:
    python export/export_onnx.py --checkpoint outputs/checkpoints/best.pt \
        --output outputs/model.onnx
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from omegaconf import OmegaConf

sys.path.append(str(Path(__file__).resolve().parents[1]))

from models.model import build_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
)
logger = logging.getLogger("export_onnx")


def load_config(config_path: str):
    return OmegaConf.load(config_path)


def load_model_from_checkpoint(checkpoint_path: str, cfg):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = build_model(cfg)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


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

    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    logger.info("ONNX model structure check passed.")


def verify_onnx_vs_pytorch(model: torch.nn.Module, onnx_path: str, image_size: int,
                            atol: float = 1e-3) -> bool:
    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    tensor = torch.randn(1, 3, image_size, image_size)

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
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output", type=str, default="outputs/model.onnx")
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()

    cfg = load_config(args.config)
    image_size = cfg.data.image_size

    model = load_model_from_checkpoint(args.checkpoint, cfg)
    export_to_onnx(model, image_size, args.output, opset_version=args.opset)
    verify_onnx_vs_pytorch(model, args.output, image_size)

    logger.info(
        f"Export complete. Model: SegFormer-B5 | Classes: {cfg.model.segformer.num_classes} | "
        f"Input size: {image_size}x{image_size}"
    )


if __name__ == "__main__":
    main()
