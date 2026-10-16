"""
tree_analysis.py
------------------
Computes scientific/geometric metrics from the cleaned segmentation output,
feeding Module 5 (Structural Feature Extraction) and Module 6 (Biomass).

Two metrics here are explicitly heuristic estimates, not ground truth, and
are documented as such rather than presented with false precision:

- occlusion_percentage: approximated from the area added back by hole-filling
  during post-processing (postprocessing/postprocess.py) relative to the
  final mask area — i.e., how much of the tree's silhouette was inferred
  rather than directly segmented as foreground (background/sky visible
  through canopy gaps, occluding poles/wires, etc.).
- shadow_percentage: approximated via an HSV brightness/saturation heuristic
  over pixels inside the tree mask (low value + moderate saturation is a
  standard, well-documented shadow cue), not a learned shadow detector.

Both are flagged with an `"is_estimate": true` field in the returned
metadata so Module 6 can weight them accordingly.
"""

import logging

import cv2
import numpy as np

logger = logging.getLogger("tree_analysis")


def coverage_ratios(tree_mask: np.ndarray, canopy_mask: np.ndarray) -> dict:
    total_px = tree_mask.size
    tree_px = int((tree_mask > 0).sum())
    canopy_px = int((canopy_mask > 0).sum())

    return {
        "tree_coverage_ratio": tree_px / total_px if total_px > 0 else 0.0,
        "canopy_coverage_ratio": canopy_px / total_px if total_px > 0 else 0.0,
        "canopy_share_of_tree": canopy_px / tree_px if tree_px > 0 else 0.0,
        "background_percentage": 1.0 - (tree_px / total_px if total_px > 0 else 0.0),
        "visible_crown_area_px": canopy_px,
        "tree_area_px": tree_px,
    }


def bounding_geometry(tree_mask: np.ndarray) -> dict:
    ys, xs = np.where(tree_mask > 0)
    if len(xs) == 0:
        return {
            "bounding_box": None, "bounding_area_px": 0, "aspect_ratio": 0.0,
            "tree_occupancy": 0.0,
        }

    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())
    box_w, box_h = (x2 - x1 + 1), (y2 - y1 + 1)
    bounding_area = box_w * box_h
    tree_area = int((tree_mask > 0).sum())

    return {
        "bounding_box": [x1, y1, x2, y2],
        "bounding_area_px": bounding_area,
        "aspect_ratio": box_w / box_h if box_h > 0 else 0.0,
        "tree_occupancy": tree_area / bounding_area if bounding_area > 0 else 0.0,  # fill-factor of its own bbox
    }


def pixel_statistics(image_rgb: np.ndarray, tree_mask: np.ndarray) -> dict:
    """Basic pixel-value statistics over the tree region, useful downstream
    for cross-checking health-model consistency and canopy density proxies."""
    fg = image_rgb[tree_mask > 0]
    if fg.size == 0:
        return {"mean_rgb": [0, 0, 0], "std_rgb": [0, 0, 0], "num_foreground_pixels": 0}

    return {
        "mean_rgb": fg.mean(axis=0).round(2).tolist(),
        "std_rgb": fg.std(axis=0).round(2).tolist(),
        "num_foreground_pixels": int(fg.shape[0]),
    }


def tree_orientation(tree_mask: np.ndarray) -> dict:
    """PCA on foreground pixel coordinates gives the tree's principal axis
    (orientation) and elongation (eigenvalue ratio) — a crown-shape proxy
    reusable by Module 5 for crown-diameter/height-axis estimation."""
    ys, xs = np.where(tree_mask > 0)
    if len(xs) < 2:
        return {"orientation_degrees": 0.0, "elongation": 0.0}

    coords = np.stack([xs, ys], axis=1).astype(np.float64)
    coords -= coords.mean(axis=0)
    cov = np.cov(coords, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]

    principal_axis = eigvecs[:, 0]
    angle_rad = np.arctan2(principal_axis[1], principal_axis[0])
    angle_deg = float(np.degrees(angle_rad)) % 180.0  # orientation is axis-symmetric

    elongation = float(eigvals[0] / eigvals[1]) if eigvals[1] > 1e-6 else float("inf")

    return {"orientation_degrees": round(angle_deg, 2), "elongation": round(elongation, 3)}


def estimate_occlusion_percentage(mask_before_hole_fill: np.ndarray, mask_after_hole_fill: np.ndarray) -> dict:
    filled_px = int(((mask_after_hole_fill > 0) & (mask_before_hole_fill == 0)).sum())
    final_area = int((mask_after_hole_fill > 0).sum())
    pct = filled_px / final_area if final_area > 0 else 0.0
    return {"occlusion_percentage": round(pct, 4), "is_estimate": True}


def estimate_shadow_percentage(image_rgb: np.ndarray, tree_mask: np.ndarray,
                                value_threshold: int = 70, sat_min: int = 20) -> dict:
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    fg_indices = tree_mask > 0
    if fg_indices.sum() == 0:
        return {"shadow_percentage": 0.0, "is_estimate": True}

    v_channel = hsv[:, :, 2]
    s_channel = hsv[:, :, 1]
    shadow_like = (v_channel < value_threshold) & (s_channel > sat_min) & fg_indices
    pct = float(shadow_like.sum()) / float(fg_indices.sum())
    return {"shadow_percentage": round(pct, 4), "is_estimate": True}


def compute_full_analysis(image_rgb: np.ndarray, tree_mask: np.ndarray, canopy_mask: np.ndarray,
                           mask_before_hole_fill: np.ndarray = None) -> dict:
    """Orchestrates the full tree-analysis metric suite into one dict, ready
    to be embedded directly in mask_metadata.json."""
    result = {}
    result.update(coverage_ratios(tree_mask, canopy_mask))
    result.update(bounding_geometry(tree_mask))
    result["pixel_statistics"] = pixel_statistics(image_rgb, tree_mask)
    result["orientation"] = tree_orientation(tree_mask)

    if mask_before_hole_fill is not None:
        result["occlusion"] = estimate_occlusion_percentage(mask_before_hole_fill, tree_mask)
    else:
        result["occlusion"] = {"occlusion_percentage": None, "is_estimate": True}

    result["shadow"] = estimate_shadow_percentage(image_rgb, tree_mask)

    return result
