#!/usr/bin/env python3
"""
segment.py
-----------
TreeSegmenter: the single entrypoint Module 5 (and any other caller) should
import. Given only `cropped_tree.jpg` (Module 3's output), produces every
Module 4 deliverable: tree/canopy/trunk masks, RGBA cutout, polygon, and a
rich metadata bundle, with a documented quality-gate that can reject a bad
segmentation rather than silently handing Module 5 garbage.

Methods:
    load_model()      -- loads the SegFormer checkpoint (+ optional SAM2 fallback)
    segment_tree()     -- raw model inference -> per-pixel class prediction + confidence
    extract_canopy()    -- isolates the canopy sub-mask
    generate_polygon()   -- Shapely polygon -> GeoJSON-style dict
    generate_metadata()  -- full mask_metadata.json content
    predict_api()         -- single call: segment -> postprocess -> analyze -> quality-gate -> JSON contract

Usage:
    python inference/segment.py --checkpoint outputs/checkpoints/best.pt \
        --image cropped_tree.jpg
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from omegaconf import OmegaConf

sys.path.append(str(Path(__file__).resolve().parents[1]))

from data.dataset import get_eval_transforms, IMAGENET_MEAN, IMAGENET_STD
from models.model import build_model, SAM2Wrapper, remap_checkpoint_keys
from postprocessing.postprocess import postprocess_mask, connected_component_analysis
from analysis.tree_analysis import compute_full_analysis
from training.utils import get_device

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
)
logger = logging.getLogger("segment")


class TreeSegmenter:
    """Loads a trained SegFormer checkpoint and produces every Module 4
    deliverable for a single cropped tree image."""

    def __init__(self, checkpoint_path: str, config_path: str = "configs/config.yaml"):
        self.cfg = self._load_config(config_path)
        self.checkpoint_path = checkpoint_path
        self.device = get_device(self.cfg.project.device)
        self.model = None
        self.sam2 = None
        self.load_model()

    @staticmethod
    def _load_config(config_path: str):
        return OmegaConf.load(config_path)

    # -----------------------------------------------------------------
    def load_model(self):
        """Loads the SegFormer checkpoint into self.model, and lazily builds
        the optional SAM2 fallback if enabled in config.

        Checkpoint state_dict keys are remapped from the old HuggingFace
        transformers v4.x layout to the current v5.x layout via
        remap_checkpoint_keys() so that checkpoints produced by either
        version load cleanly without any manual intervention.
        """
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
        self.model = build_model(self.cfg).to(self.device)
        raw_sd = checkpoint["model_state_dict"]
        remapped_sd = remap_checkpoint_keys(raw_sd)
        missing, unexpected = self.model.load_state_dict(remapped_sd, strict=False)
        if missing or unexpected:
            logger.warning(
                f"load_state_dict: {len(missing)} missing, {len(unexpected)} unexpected keys. "
                "This may indicate a checkpoint/model version mismatch."
            )
        else:
            logger.info(f"load_state_dict: all {len(remapped_sd)} keys matched (0 missing, 0 unexpected).")
        self.model.eval()
        logger.info(
            f"TreeSegmenter ready (checkpoint={self.checkpoint_path}, "
            f"device={self.device}, epoch={checkpoint.get('epoch', '?')}, "
            f"best_metric={checkpoint.get('best_metric', '?'):.4f})"
        )

        if self.cfg.model.sam2.enabled_for_inference_fallback:
            try:
                self.sam2 = SAM2Wrapper(
                    self.cfg.model.sam2.checkpoint, self.cfg.model.sam2.model_config,
                    device=str(self.device),
                )
            except (ImportError, FileNotFoundError) as e:
                logger.warning(f"SAM2 fallback requested but unavailable: {e}")
                self.sam2 = None

        return self.model

    # -----------------------------------------------------------------
    def segment_tree(self, image_bgr: np.ndarray) -> dict:
        """Runs the SegFormer forward pass and returns per-pixel class
        predictions + a per-pixel confidence map, at the ORIGINAL image
        resolution (upsampled from model resolution)."""
        orig_h, orig_w = image_bgr.shape[:2]
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        transform = get_eval_transforms(self.cfg.data.image_size)
        tensor = transform(image=image_rgb)["image"].unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)  # [1, C, model_h, model_w]
            probs = torch.softmax(logits, dim=1)
            
            # Descope trunk: target is now binary (bg vs tree). Merged model
            # was trained to distribute tree mass across canopy (1) and trunk (2).
            # We must sum them to get the true foreground confidence.
            bg_prob = probs[:, 0]
            fg_prob = probs[:, 1] + probs[:, 2]
            
            confidence = torch.max(bg_prob, fg_prob)
            pred = (fg_prob > bg_prob).long()

        pred_np = pred.squeeze(0).cpu().numpy().astype(np.uint8)
        conf_np = confidence.squeeze(0).cpu().numpy().astype(np.float32)

        pred_full = cv2.resize(pred_np, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
        conf_full = cv2.resize(conf_np, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

        min_conf = self.cfg.inference.min_confidence
        pred_full = np.where(conf_full >= min_conf, pred_full, 0)  # low-confidence -> background

        return {"class_map": pred_full, "confidence_map": conf_full, "image_rgb": image_rgb}

    # -----------------------------------------------------------------
    def extract_canopy(self, class_map: np.ndarray) -> np.ndarray:
        """Isolates the canopy sub-mask (class 1)."""
        return (class_map == self.cfg.classes.canopy).astype(np.uint8)

    # -----------------------------------------------------------------
    def generate_polygon(self, tree_mask: np.ndarray) -> dict:
        """Runs the post-processing polygon pipeline and returns a
        GeoJSON-style dict ready to be written to tree_polygon.json."""
        pp_cfg = self.cfg.postprocessing
        result = postprocess_mask(
            tree_mask, pp_cfg.morph_kernel_size, pp_cfg.small_region_min_area_px,
            pp_cfg.boundary_smoothing_epsilon_ratio, pp_cfg.hole_fill,
        )
        polygon = result["polygon"]
        if polygon is None:
            return {"type": "Polygon", "coordinates": [], "area_px": 0.0, "perimeter_px": 0.0}, result

        coords = list(polygon.exterior.coords)
        polygon_dict = {
            "type": "Polygon",
            "coordinates": [coords],
            "area_px": polygon.area,
            "perimeter_px": polygon.length,
        }
        return polygon_dict, result

    # -----------------------------------------------------------------
    def _compute_quality_scores(self, class_map: np.ndarray, confidence_map: np.ndarray,
                                 tree_mask_raw: np.ndarray, tree_mask_clean: np.ndarray,
                                 cca_result: dict) -> dict:
        quality_cfg = self.cfg.quality
        total_px = tree_mask_clean.size
        coverage = float((tree_mask_clean > 0).sum()) / total_px if total_px > 0 else 0.0

        fg_confidence = confidence_map[tree_mask_clean > 0]
        confidence_score = float(fg_confidence.mean()) if fg_confidence.size > 0 else 0.0

        largest_component_area = max((c["area"] for c in cca_result["components"]), default=0)
        raw_fg_area = int((tree_mask_raw > 0).sum())
        connectivity_score = largest_component_area / raw_fg_area if raw_fg_area > 0 else 0.0

        coverage_ok = quality_cfg.min_coverage_ratio <= coverage <= quality_cfg.max_coverage_ratio
        coverage_score = 1.0 if coverage_ok else 0.0

        quality_score = float(np.mean([confidence_score, connectivity_score, coverage_score]))

        reject_reason = None
        if not coverage_ok:
            reject_reason = "coverage_too_low" if coverage < quality_cfg.min_coverage_ratio else "coverage_too_high"
        elif quality_cfg.reject_on_disconnected and connectivity_score < quality_cfg.disconnection_min_largest_component_ratio:
            reject_reason = "mask_too_fragmented"
        elif quality_score < quality_cfg.min_quality_score:
            reject_reason = "low_quality_score"

        return {
            "coverage": round(coverage, 4),
            "quality": round(quality_score, 4),
            "confidence_score": round(confidence_score, 4),
            "connectivity_score": round(connectivity_score, 4),
            "coverage_score": round(coverage_score, 4),
            "reject_reason": reject_reason,
        }

    # -----------------------------------------------------------------
    def generate_metadata(self, image_rgb: np.ndarray, tree_mask: np.ndarray, canopy_mask: np.ndarray,
                           tree_mask_before_hole_fill: np.ndarray,
                           cca_result: dict, quality_scores: dict) -> dict:
        """Assembles the full mask_metadata.json content — everything Module 5
        needs, precomputed, so it requires no further preprocessing."""
        analysis = compute_full_analysis(
            image_rgb, tree_mask, canopy_mask, tree_mask_before_hole_fill
        )
        return {
            "quality_scores": quality_scores,
            "tree_analysis": analysis,
            "connected_components": {
                "num_components": cca_result["num_components"],
                "components": cca_result["components"],
            },
            "model": {
                "architecture": "SegFormer-B5",
                "checkpoint": str(self.checkpoint_path),
                "num_classes": self.cfg.model.segformer.num_classes,
            },
            "image_shape": {"height": image_rgb.shape[0], "width": image_rgb.shape[1]},
        }

    # -----------------------------------------------------------------
    def predict_api(self, image_path: str, out_dir: str = None, out_stem: str = None) -> dict:
        """Single-call entrypoint: segment -> postprocess -> analyze ->
        quality-gate -> write all Module 4 deliverables -> return the JSON
        contract Module 5 (or an API layer) expects.
        """
        out_dir = out_dir or self.cfg.paths.results_dir
        out_stem = out_stem or Path(image_path).stem
        result_dir = Path(out_dir) / out_stem
        result_dir.mkdir(parents=True, exist_ok=True)

        image_bgr = cv2.imread(image_path)
        if image_bgr is None:
            return {"error": "IMAGE_READ_FAILED", "ready_for_module5": False}

        seg = self.segment_tree(image_bgr)
        class_map, confidence_map, image_rgb = seg["class_map"], seg["confidence_map"], seg["image_rgb"]

        tree_mask_raw = (class_map > 0).astype(np.uint8)
        canopy_mask_raw = self.extract_canopy(class_map)
        trunk_mask_raw = np.zeros_like(tree_mask_raw)  # descoped: blank trunk mask

        # Optional SAM2 fallback if quality looks weak up front (pre-postprocess check)
        if self.sam2 is not None and tree_mask_raw.sum() > 0:
            provisional_conf = float(confidence_map[tree_mask_raw > 0].mean())
            if provisional_conf < self.cfg.model.sam2.fallback_quality_threshold:
                ys, xs = np.where(tree_mask_raw > 0)
                box = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
                logger.info(f"Low SegFormer confidence ({provisional_conf:.2f}) — invoking SAM2 fallback")
                sam2_mask = self.sam2.segment_with_box(image_rgb, box)
                tree_mask_raw = sam2_mask
                canopy_mask_raw = sam2_mask  # SAM2 fallback gives whole-tree only, no part split
                trunk_mask_raw = np.zeros_like(sam2_mask)

        pp_cfg = self.cfg.postprocessing
        pp_result = postprocess_mask(
            tree_mask_raw, pp_cfg.morph_kernel_size, pp_cfg.small_region_min_area_px,
            pp_cfg.boundary_smoothing_epsilon_ratio, pp_cfg.hole_fill,
        )
        tree_mask_clean = pp_result["cleaned_mask"]
        cca_result = pp_result["cca_result"]

        # Constrain canopy/trunk sub-masks to the cleaned tree silhouette
        canopy_mask_clean = (canopy_mask_raw & tree_mask_clean).astype(np.uint8)
        trunk_mask_clean = (trunk_mask_raw & tree_mask_clean).astype(np.uint8)

        quality_scores = self._compute_quality_scores(
            class_map, confidence_map, tree_mask_raw, tree_mask_clean, cca_result
        )

        if quality_scores["reject_reason"] is not None:
            logger.warning(f"MASK_QUALITY_REJECTED ({quality_scores['reject_reason']}) for {image_path}")
            return {
                "error": "MASK_QUALITY_REJECTED",
                "reason": quality_scores["reject_reason"],
                "coverage": quality_scores["coverage"],
                "quality": quality_scores["quality"],
                "ready_for_module5": False,
            }

        # --- Write all deliverables ---
        tree_mask_path = result_dir / "tree_mask.png"
        canopy_mask_path = result_dir / "canopy_mask.png"
        trunk_mask_path = result_dir / "trunk_mask.png"
        rgba_path = result_dir / "tree_rgba.png"
        polygon_path = result_dir / "tree_polygon.json"
        metadata_path = result_dir / "mask_metadata.json"

        cv2.imwrite(str(tree_mask_path), tree_mask_clean * 255)
        cv2.imwrite(str(canopy_mask_path), canopy_mask_clean * 255)
        cv2.imwrite(str(trunk_mask_path), trunk_mask_clean * 255)

        rgba = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2BGRA)
        rgba[:, :, 3] = tree_mask_clean * 255
        cv2.imwrite(str(rgba_path), rgba)

        polygon_dict, _ = self.generate_polygon(tree_mask_clean)
        with open(polygon_path, "w") as f:
            json.dump(polygon_dict, f, indent=2)

        metadata = self.generate_metadata(
            image_rgb, tree_mask_clean, canopy_mask_clean,
            tree_mask_raw, cca_result, quality_scores,
        )
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        return {
            "tree_mask": str(tree_mask_path),
            "tree_rgba": str(rgba_path),
            "canopy_mask": str(canopy_mask_path),
            "trunk_mask": str(trunk_mask_path),
            "polygon": str(polygon_path),
            "coverage": quality_scores["coverage"],
            "quality": quality_scores["quality"],
            "ready_for_module5": True,
            "metadata": str(metadata_path),
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default=None)
    args = parser.parse_args()

    segmenter = TreeSegmenter(args.checkpoint, config_path=args.config)
    result = segmenter.predict_api(args.image, out_dir=args.out_dir)

    print(json.dumps(result, indent=2))
    if "error" in result:
        logger.warning(f"Result: {result['error']}")
    else:
        logger.info(f"Segmentation ready for Module 5: coverage={result['coverage']:.2%} "
                    f"quality={result['quality']:.2f}")


if __name__ == "__main__":
    main()
