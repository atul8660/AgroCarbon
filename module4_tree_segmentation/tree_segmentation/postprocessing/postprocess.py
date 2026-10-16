"""
postprocess.py
----------------
Post-processing suite applied to raw model output masks before they become
Module 4's final deliverables. Order matters — each step is documented with
why it runs where it does in the pipeline.

Pipeline (see `postprocess_mask()` for the orchestrated order):
    1. Morphological closing  -- fills small gaps/breaks in the mask boundary
    2. Morphological opening   -- removes small spurious foreground specks
    3. Small region removal    -- drops connected components below area threshold
    4. Hole filling             -- fills fully-enclosed background holes (e.g.
                                    sky visible through gaps in a sparse canopy —
                                    the tree's *silhouette* should be closed for
                                    downstream area/volume estimation, even
                                    though the raw canopy is not literally solid)
    5. Connected component analysis + largest component selection
                                 -- the tree crop should contain one primary tree;
                                    keeps the dominant blob, drops disconnected debris
    6. Boundary smoothing        -- cv2.approxPolyDP-based contour smoothing
    7. Contour extraction + polygon approximation -- final Shapely-validated polygon
"""

import logging
from typing import Optional

import cv2
import numpy as np
from shapely.geometry import Polygon
from shapely.validation import make_valid

logger = logging.getLogger("postprocess")


def morphological_closing(mask: np.ndarray, kernel_size: int) -> np.ndarray:
    """Closes small gaps/breaks along the mask boundary (dilate then erode)."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)


def morphological_opening(mask: np.ndarray, kernel_size: int) -> np.ndarray:
    """Removes small spurious foreground specks (erode then dilate)."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)


def remove_small_regions(mask: np.ndarray, min_area_px: int) -> np.ndarray:
    """Drops connected components smaller than min_area_px (noise removal)."""
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    cleaned = np.zeros_like(mask)
    for label_id in range(1, num_labels):  # skip background label 0
        if stats[label_id, cv2.CC_STAT_AREA] >= min_area_px:
            cleaned[labels == label_id] = 1
    return cleaned.astype(mask.dtype)


def fill_holes(mask: np.ndarray) -> np.ndarray:
    """Fills fully-enclosed background regions inside the foreground mask
    (e.g. sky gaps in a sparse canopy) so the tree silhouette used for
    area/volume proxies in Module 5 reflects the tree's outer envelope."""
    h, w = mask.shape
    flood_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
    filled = mask.copy()
    cv2.floodFill(filled, flood_mask, (0, 0), 1)
    filled_inv = cv2.bitwise_not(filled * 255) // 255
    return (mask | filled_inv).astype(mask.dtype)


def connected_component_analysis(mask: np.ndarray) -> dict:
    """Returns full CCA stats: number of components, per-component area/bbox,
    used both for largest-component selection and for the disconnection
    quality check in inference/segment.py."""
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    components = []
    for label_id in range(1, num_labels):
        components.append({
            "label_id": label_id,
            "area": int(stats[label_id, cv2.CC_STAT_AREA]),
            "bbox": [
                int(stats[label_id, cv2.CC_STAT_LEFT]),
                int(stats[label_id, cv2.CC_STAT_TOP]),
                int(stats[label_id, cv2.CC_STAT_LEFT] + stats[label_id, cv2.CC_STAT_WIDTH]),
                int(stats[label_id, cv2.CC_STAT_TOP] + stats[label_id, cv2.CC_STAT_HEIGHT]),
            ],
            "centroid": [float(centroids[label_id][0]), float(centroids[label_id][1])],
        })
    return {"num_components": len(components), "components": components, "labels": labels}


def select_largest_component(mask: np.ndarray, cca_result: Optional[dict] = None) -> np.ndarray:
    """Keeps only the largest connected component — the tree crop should
    contain one primary tree; disconnected small blobs are treated as debris
    (neighboring branches, artifacts) rather than part of the subject tree."""
    if cca_result is None:
        cca_result = connected_component_analysis(mask)
    if cca_result["num_components"] == 0:
        return mask
    largest = max(cca_result["components"], key=lambda c: c["area"])
    return (cca_result["labels"] == largest["label_id"]).astype(mask.dtype)


def extract_contours(mask: np.ndarray) -> list:
    """Extracts external contours from a binary mask."""
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def smooth_boundary(contour: np.ndarray, epsilon_ratio: float) -> np.ndarray:
    """cv2.approxPolyDP-based boundary smoothing/simplification, epsilon
    scaled to the contour's own perimeter so it behaves consistently across
    different crop resolutions."""
    perimeter = cv2.arcLength(contour, True)
    epsilon = epsilon_ratio * perimeter
    return cv2.approxPolyDP(contour, epsilon, True)


def polygon_from_mask(mask: np.ndarray, epsilon_ratio: float) -> Optional[Polygon]:
    """Extracts the largest external contour, smooths it, and returns a
    Shapely Polygon (validated/repaired via make_valid), or None if no
    foreground pixels are present."""
    contours = extract_contours(mask)
    if not contours:
        return None
    largest_contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest_contour) < 1:
        return None

    smoothed = smooth_boundary(largest_contour, epsilon_ratio)
    coords = smoothed.reshape(-1, 2).tolist()
    if len(coords) < 3:
        return None

    poly = Polygon(coords)
    if not poly.is_valid:
        poly = make_valid(poly)
        if poly.geom_type != "Polygon":
            # make_valid can return a MultiPolygon/GeometryCollection for
            # self-intersecting input; keep the largest constituent polygon.
            polys = [g for g in getattr(poly, "geoms", []) if g.geom_type == "Polygon"]
            if not polys:
                return None
            poly = max(polys, key=lambda p: p.area)
    return poly


def postprocess_mask(raw_mask: np.ndarray, morph_kernel_size: int, small_region_min_area_px: int,
                      boundary_smoothing_epsilon_ratio: float, hole_fill: bool = True) -> dict:
    """Runs the full documented post-processing pipeline on a raw binary
    mask (0/1) and returns the cleaned mask plus derived artifacts."""
    mask = (raw_mask > 0).astype(np.uint8)

    mask = morphological_closing(mask, morph_kernel_size)
    mask = morphological_opening(mask, morph_kernel_size)
    mask = remove_small_regions(mask, small_region_min_area_px)

    if hole_fill:
        mask = fill_holes(mask)

    cca_result = connected_component_analysis(mask)
    mask = select_largest_component(mask, cca_result)

    polygon = polygon_from_mask(mask, boundary_smoothing_epsilon_ratio)

    return {
        "cleaned_mask": mask,
        "cca_result": {"num_components": cca_result["num_components"],
                        "components": cca_result["components"]},
        "polygon": polygon,
    }
