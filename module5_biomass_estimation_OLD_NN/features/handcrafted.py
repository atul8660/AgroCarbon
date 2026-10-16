import cv2
import numpy as np

class GeometricFeatureExtractor:
    """
    Extracts handcrafted morphological and geometric features from 
    Module 4's segmentation masks (tree, canopy, trunk).
    """
    def __init__(self):
        self.num_features = 22
        
    def extract(self, tree_mask: np.ndarray, canopy_mask: np.ndarray, trunk_mask: np.ndarray, polygon: list) -> np.ndarray:
        """
        Calculates handcrafted constraints.
        Returns a 1D numpy array of length `self.num_features`.
        """
        features = []
        
        # 1-3. Basic Area calculations (pixels)
        tree_area = np.sum(tree_mask > 0)
        canopy_area = np.sum(canopy_mask > 0)
        trunk_area = np.sum(trunk_mask > 0)
        
        features.extend([tree_area, canopy_area, trunk_area])
        
        # Contours for Geometry
        tree_mask_u8 = (tree_mask > 0).astype(np.uint8) * 255
        contours, _ = cv2.findContours(tree_mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) > 0:
            main_contour = max(contours, key=cv2.contourArea)
            
            # 4. Perimeter
            perimeter = cv2.arcLength(main_contour, True)
            
            # 5-6. Bounding Box Width and Height
            x, y, w, h = cv2.boundingRect(main_contour)
            
            # 7. Aspect Ratio
            aspect_ratio = float(w) / h if h > 0 else 0
            
            # 8. Convex Hull Area
            hull = cv2.convexHull(main_contour)
            hull_area = cv2.contourArea(hull)
            
            # 9. Extent (Object Area / Bounding Box Area)
            rect_area = w * h
            extent = float(tree_area) / rect_area if rect_area > 0 else 0
            
            # 10. Solidity (Object Area / Convex Hull Area)
            solidity = float(tree_area) / hull_area if hull_area > 0 else 0
            
            # 11. Circularity
            circularity = (4 * np.pi * tree_area) / (perimeter ** 2) if perimeter > 0 else 0
            
            # 12. Equivalent Diameter
            equiv_diameter = np.sqrt(4 * tree_area / np.pi)
            
            # 13. Orientation (Angle of the fitted ellipse)
            if len(main_contour) >= 5:
                # Requires at least 5 points to fit an ellipse
                (_, _), (_, _), angle = cv2.fitEllipse(main_contour)
            else:
                angle = 0
            
            features.extend([
                perimeter, w, h, aspect_ratio, hull_area, 
                extent, solidity, circularity, equiv_diameter, angle
            ])
            
        else:
            # Fallback if no contours
            features.extend([0]*10)
            
        # 14. Occupancy Ratio
        total_pixels = tree_mask.shape[0] * tree_mask.shape[1]
        occupancy = tree_area / total_pixels if total_pixels > 0 else 0
        
        # 15. Canopy Ratio
        canopy_ratio = canopy_area / tree_area if tree_area > 0 else 0
        
        # 16. Trunk Ratio
        trunk_ratio = trunk_area / tree_area if tree_area > 0 else 0
        
        features.extend([occupancy, canopy_ratio, trunk_ratio])
        
        # 17. Polygon Complexity (Number of vertices)
        poly_complexity = len(polygon) if polygon else 0
        features.append(poly_complexity)
        
        # Padding for GLCM/Texture/Color Stats (placeholders to reach 22 as an example)
        # In a full run, we'd compute Haralick textures or LBP over the `cropped_tree.jpg`.
        # Here we pad the remaining 5 features with 0 for pure geometry demo.
        remaining = self.num_features - len(features)
        features.extend([0.0] * remaining)
        
        assert len(features) == self.num_features, f"Expected {self.num_features} features but got {len(features)}"
        
        return np.array(features, dtype=np.float32)
