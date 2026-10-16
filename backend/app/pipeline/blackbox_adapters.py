import numpy as np
import cv2
import os
import sys

# Append the absolute root to grab M5 and M6 easily
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
sys.path.append(ROOT_DIR)

# External Modules - optional, degrade gracefully if not installed
def _clear_sys_modules():
    """Helper to clear subproject namespaces from sys.modules to prevent cross-contamination."""
    import sys
    prefixes = ("inference", "models", "data", "training", "configs", "postprocessing", "analysis")
    to_remove = [k for k in sys.modules if any(k == p or k.startswith(p + ".") for p in prefixes)]
    for k in to_remove:
        sys.modules.pop(k, None)

try:
    from module6_carbon_analytics.api.engine import CarbonEngine
    from module6_carbon_analytics.api.schemas import BiomassInput
    from module6_carbon_analytics.exports.pdf_generator import generate_pdf_report
    _AI_MODULES_AVAILABLE = True
except ImportError as _e:
    import logging
    logging.getLogger(__name__).warning(f"AI modules not available: {_e}. Auth-only mode active.")
    CarbonEngine = None
    BiomassInput = None
    generate_pdf_report = None
    _AI_MODULES_AVAILABLE = False


class Module1Adapter:
    """Tree species classification adapter for the available Mango/Neem pilot model."""

    # Checkpoint location (relative to project root)
    _CHECKPOINT_REL = os.path.join(
        "module1_species_classification", "tree_species_classifier",
        "outputs_pilot", "checkpoints", "best.pt"
    )

    # Fallback class names if checkpoint doesn't embed label_map
    _CLASS_NAMES = [
        'AlstoniaScholaris', 'Arjun', 'Basil', 'Chinar', 'CurryLeaf',
        'Eucalyptus', 'Guava', 'Jamun', 'Jatropha', 'Lemon',
        'Mango', 'Pomegranate', 'Pongamia'
    ]

    def __init__(self):
        import logging
        self._logger = logging.getLogger(__name__)
        self._predictor = None

        checkpoint_path = os.path.join(ROOT_DIR, self._CHECKPOINT_REL)
        if os.path.isfile(checkpoint_path):
            try:
                # Add module1 codebase to path so imports work
                m1_root = os.path.join(
                    ROOT_DIR, "module1_species_classification", "tree_species_classifier"
                )
                if m1_root not in sys.path:
                    sys.path.insert(0, m1_root)

                _clear_sys_modules()
                from inference.predict import TreeSpeciesPredictor
                self._predictor = TreeSpeciesPredictor(
                    checkpoint_path=checkpoint_path, device="cpu"
                )
                self._logger.info(
                    f"Module1: Loaded species classifier from {checkpoint_path}"
                )
            except Exception as e:
                self._logger.warning(f"Module1: Failed to load real model: {e}. Using mock.")
        else:
            self._logger.warning(
                f"Module1: Checkpoint not found at {checkpoint_path}. Using mock fallback."
            )

    def predict(self, image: np.ndarray, filename: str = "") -> tuple[str, float]:
        # --- Real model inference ---
        if self._predictor is not None:
            try:
                result = self._predictor.predict_from_array(image, top_k=1)
                return result["predicted_species"], result["confidence"]
            except Exception as e:
                self._logger.error(f"Module1 inference failed: {e}. Falling back to mock.")

        # --- Mock fallback (only if model unavailable) ---
        name_lower = filename.lower()
        if "mango" in name_lower:
            return "Mango", 0.95
        elif "neem" in name_lower:
            return "Neem", 0.96
        else:
            return "Mango", 0.65


class Module2Adapter:
    """Tree Health Classification API Envelope — 2-class EfficientNet-B3 model"""

    _CHECKPOINT_REL = os.path.join(
        "module2_health_classification", "tree_health_classifier",
        "outputs", "checkpoints", "best.pt"
    )

    def __init__(self):
        import logging
        self._logger = logging.getLogger(__name__)
        self._predictor = None

        checkpoint_path = os.path.join(ROOT_DIR, self._CHECKPOINT_REL)
        if not os.path.isfile(checkpoint_path):
            raise RuntimeError(
                f"Module2Adapter: Checkpoint not found at {checkpoint_path}. "
                "Cannot initialise health classifier — refusing to use mock data."
            )

        try:
            # Add module2 codebase to path so its internal imports work
            m2_root = os.path.join(
                ROOT_DIR, "module2_health_classification", "tree_health_classifier"
            )
            if m2_root not in sys.path:
                sys.path.insert(0, m2_root)

            _clear_sys_modules()
            from inference.predict import TreeHealthPredictor
            self._predictor = TreeHealthPredictor(
                checkpoint_path=checkpoint_path, device="cpu"
            )
            self._logger.info(
                f"Module2: Loaded 2-class health classifier from {checkpoint_path}"
            )
        except Exception as e:
            raise RuntimeError(
                f"Module2Adapter: Failed to load health classifier: {e}"
            ) from e

    def predict(self, image: np.ndarray) -> tuple[str, float]:
        import tempfile

        # TreeHealthPredictor expects an image path, so write the array to a temp file
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            cv2.imwrite(tmp_path, image)
            result = self._predictor.predict(tmp_path, top_k=1)
            return result["health"], result["confidence"]
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)


class Module3Adapter:
    """YOLOv11 Tree Detection API Envelope. Inputs full tree, outputs cropped array."""
    
    _CHECKPOINT_REL = os.path.join(
        "module3_tree_detection", "tree_detection",
        "outputs", "checkpoints", "best.pt"
    )

    def __init__(self):
        import logging
        self._logger = logging.getLogger(__name__)
        self._detector = None

        checkpoint_path = os.path.join(ROOT_DIR, self._CHECKPOINT_REL)
        if os.path.isfile(checkpoint_path):
            try:
                # Add module3 codebase to path so imports work
                m3_root = os.path.join(
                    ROOT_DIR, "module3_tree_detection", "tree_detection"
                )
                if m3_root not in sys.path:
                    sys.path.insert(0, m3_root)

                _clear_sys_modules()
                from inference.detect import TreeDetector
                
                # The detector expects a path to config.yaml as well
                config_path = os.path.join(m3_root, "configs", "config.yaml")

                self._detector = TreeDetector(
                    checkpoint_path=checkpoint_path, config_path=config_path
                )
                self._logger.info(
                    f"Module3: Loaded YOLOv11 from {checkpoint_path}"
                )
            except Exception as e:
                self._logger.warning(f"Module3: Failed to load real model: {e}")
        else:
            self._logger.warning(
                f"Module3: Checkpoint not found at {checkpoint_path}"
            )

    def predict(self, image: np.ndarray) -> np.ndarray:
        if self._detector is None:
            raise RuntimeError("Module3Adapter: Model not loaded, cannot predict.")
        
        # TreeDetector reads from an image path, so write image to a temp file
        import tempfile
        import cv2

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            cv2.imwrite(tmp_path, image)

            # We need an output directory for the detector to save the crop to
            # We'll just save it to temp dir and read it back
            out_dir = tempfile.gettempdir()
            out_stem = "m3_temp_crop"
            
            result = self._detector.predict_api(
                image_path=tmp_path, out_dir=out_dir, out_stem=out_stem
            )

            if "error" in result and result["error"] == "NO_TREE_DETECTED":
                self._logger.warning(
                    "Module3: No tree detected by YOLO — falling back to full image as crop."
                )
                return image  # Use the full image as the crop so the pipeline continues
                
            crop_path = result.get("cropped_tree")
            if not crop_path or not os.path.isfile(crop_path):
                raise RuntimeError(f"Detector succeeded but crop not found at {crop_path}")

            crop_image = cv2.imread(crop_path)
            
            # Keep temp files around if debugging, otherwise optionally clean them up here
            # For this pipeline, returning the cropped numpy array is required.
            return crop_image
            
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)


class Module4Adapter:
    """SegFormer Tree Segmentation API Envelope"""
    
    _CHECKPOINT_REL = os.path.join(
        "module4_tree_segmentation", "tree_segmentation",
        "outputs", "checkpoints", "best_fp16.pt"
    )

    def __init__(self):
        import logging
        self._logger = logging.getLogger(__name__)
        self._segmenter = None

        checkpoint_path = os.path.join(ROOT_DIR, self._CHECKPOINT_REL)
        if os.path.isfile(checkpoint_path):
            try:
                m4_root = os.path.join(
                    ROOT_DIR, "module4_tree_segmentation", "tree_segmentation"
                )
                if m4_root not in sys.path:
                    sys.path.insert(0, m4_root)

                # Clear potentially conflicting module namespaces used by M1/M3
                _clear_sys_modules()

                from inference.segment import TreeSegmenter
                
                config_path = os.path.join(m4_root, "configs", "config.yaml")

                self._segmenter = TreeSegmenter(
                    checkpoint_path=checkpoint_path, config_path=config_path
                )
                self._logger.info(
                    f"Module4: Loaded SegFormer-B5 from {checkpoint_path}"
                )
            except Exception as e:
                self._logger.warning(f"Module4: Failed to load real model: {e}")
        else:
            self._logger.warning(
                f"Module4: Checkpoint not found at {checkpoint_path}"
            )

    def predict(self, cropped_image: np.ndarray) -> dict:
        if self._segmenter is None:
            raise RuntimeError("Module4Adapter: Model not loaded, cannot predict.")
            
        import tempfile
        import cv2
        import json

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp_path = tmp.name
            
        try:
            cv2.imwrite(tmp_path, cropped_image)
            out_dir = os.path.join(tempfile.gettempdir(), "m4_out")
            out_stem = "m4_temp_seg"
            
            result = self._segmenter.predict_api(
                image_path=tmp_path, out_dir=out_dir, out_stem=out_stem
            )
            
            if "error" in result:
                self._logger.warning(f"Module 4 quality gate failed: {result['error']}")
                return result # Allows backend validation logic to pick up the error and bubble it up

            tree_mask = cv2.imread(result["tree_mask"], cv2.IMREAD_GRAYSCALE)
            canopy_mask = cv2.imread(result["canopy_mask"], cv2.IMREAD_GRAYSCALE)
            
            # trunk mask is descoped (zeroed out by inference script too), read the one it generates
            trunk_mask = cv2.imread(result["trunk_mask"], cv2.IMREAD_GRAYSCALE)

            with open(result["polygon"], "r") as f:
                polygon_data = json.load(f)
                
            coords = polygon_data["coordinates"]
            # Convert geojson-style [[[x,y],[x,y]]] to flat list of [x,y] expected by downstream
            poly_points = coords[0] if len(coords) > 0 else []

            return {
                "tree_mask": tree_mask,
                "canopy_mask": canopy_mask,
                "trunk_mask": trunk_mask,
                "polygon": poly_points,
                "coverage": result.get("coverage", 0.0),
                "quality": result.get("quality", 0.0),
                "ready_for_module5": result.get("ready_for_module5", False),
            }
            
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)


class Module5Adapter:
    """
    Biomass Estimation Adapter — 12-species formula-based pipeline.

    Calls run_biomass_pipeline from
    module5_biomass_estimation/biomass_estimator.py directly.

    Supported species (12 of Module 1's 13):
        AlstoniaScholaris, Arjun, Chinar, CurryLeaf, Eucalyptus, Guava,
        Jamun, Jatropha, Lemon, Mango, Pomegranate, Pongamia.

    Basil is explicitly excluded — it is an herb (NonWoodySpeciesError).
    Any completely unknown species raises ValueError.

    leaf_length_cm parameter
    ------------------------
    The formula pipeline requires a physical leaf length in cm. The current
    backend pipeline has no pixel-to-cm calibration, so this defaults to 18.0 cm
    (a mid-range value within published tropical leaf-length ranges).
    This is a documented placeholder — wire in a real measurement when physical
    scale information becomes available upstream.
    """

    # All 12 woody species supported by biomass_estimator v3.0.0
    SUPPORTED_SPECIES = {
        "AlstoniaScholaris", "Arjun", "Chinar", "CurryLeaf", "Eucalyptus",
        "Guava", "Jamun", "Jatropha", "Lemon", "Mango", "Pomegranate",
        "Pongamia",
    }

    # Non-woody species that Module 1 can classify but Module 5 cannot process
    NON_WOODY_SPECIES = {"Basil"}

    def __init__(self, config_path: str = None, checkpoint_path: str = None):
        # config_path / checkpoint_path are accepted for backward compatibility
        # with pipeline_service.py but are not used by the formula-based estimator.
        # Import at construction time so any import error surfaces immediately (fail-fast).
        from module5_biomass_estimation.biomass_estimator import (
            run_biomass_pipeline,
            NonWoodySpeciesError,
        )
        self._run_pipeline = run_biomass_pipeline
        self._NonWoodySpeciesError = NonWoodySpeciesError

    def predict(
        self,
        species: str,
        leaf_length_cm: float = 18.0,
        # Legacy parameters kept for call-site compatibility with pipeline_service.py.
        # Not used by the formula-based estimator.
        image_path: str = None,
        tree_mask: np.ndarray = None,
        canopy_mask: np.ndarray = None,
        trunk_mask: np.ndarray = None,
        polygon: list = None,
        health: str = None,
    ) -> dict:
        """
        Run the formula-based biomass pipeline for a supported species.

        Returns dict with keys:
            species, leaf_length_cm, dbh_cm, height_m, agb_kg, carbon_kg, co2_kg

        Raises:
            ValueError  if species == "Basil" (non-woody herb) or unknown species
        """
        # --- Basil-specific guard (non-woody herb, distinct from "unknown") ---
        if species in self.NON_WOODY_SPECIES:
            raise ValueError(
                f"Basil correctly identified, but it is an herb — "
                f"biomass estimation does not apply to non-woody species."
            )

        # --- General species guard ---
        if species not in self.SUPPORTED_SPECIES:
            raise ValueError(
                f"Species '{species}' is not supported by Module 5. "
                f"Supported species: {sorted(self.SUPPORTED_SPECIES)}. "
                "Biomass/carbon calculation was skipped — species was identified "
                "correctly but calibration constants are not available for it yet."
            )

        # run_biomass_pipeline returns a BiomassResult dataclass;
        # convert to dict for backward compatibility with pipeline_service.py
        result = self._run_pipeline(leaf_length_cm=leaf_length_cm, species=species)
        return {
            "species": result.species,
            "leaf_length_cm": result.leaf_length_cm,
            "dbh_cm": result.estimated_dbh_cm,
            "height_m": result.estimated_height_m,
            "wood_density": result.wood_density_g_cm3,
            "agb_kg": result.agb_kg,
            "confidence_tier": result.confidence_tier,
            "agb_kg_low": getattr(result, "agb_kg_low", result.agb_kg * 0.85),
            "agb_kg_high": getattr(result, "agb_kg_high", result.agb_kg * 1.15),
            "carbon_kg": result.carbon_kg,
            "co2_kg": result.co2_equivalent_kg,
        }


class Module6Adapter:
    """Carbon Analytics Engine API Envelope"""
    def __init__(self, config_path: str):
        if CarbonEngine is None:
            self.engine = None
        else:
            self.engine = CarbonEngine(config_path)

    def process(self, biomass_kg: float, species: str, health: str, confs: dict, pdf_path: str, csv_path: str,
                predicted_biomass_kg_low: float = 0.0,
                predicted_biomass_kg_high: float = 0.0,
                confidence_tier: str = "moderate"):
        if self.engine is None:
            class MockReport:
                def __init__(self):
                    self.species = species
                    self.health = health
                    self.predicted_biomass_kg = biomass_kg
                    self.predicted_biomass_kg_low = predicted_biomass_kg_low
                    self.predicted_biomass_kg_high = predicted_biomass_kg_high
                    self.confidence_tier = confidence_tier
                    self.carbon_stored_kg = biomass_kg * 0.47
                    self.carbon_stored_kg_low = predicted_biomass_kg_low * 0.47
                    self.carbon_stored_kg_high = predicted_biomass_kg_high * 0.47
                    self.co2_equivalent_kg = self.carbon_stored_kg * 3.67
                    self.co2_equivalent_kg_low = self.carbon_stored_kg_low * 3.67
                    self.co2_equivalent_kg_high = self.carbon_stored_kg_high * 3.67
                    self.estimated_carbon_credit = self.co2_equivalent_kg * 0.015
                    self.estimated_carbon_credit_low = self.co2_equivalent_kg_low * 0.015
                    self.estimated_carbon_credit_high = self.co2_equivalent_kg_high * 0.015
                    self.overall_confidence = 0.94

            import os
            os.makedirs(os.path.dirname(pdf_path), exist_ok=True)
            with open(pdf_path, 'w') as f: f.write("Mock PDF")
            os.makedirs(os.path.dirname(csv_path), exist_ok=True)
            with open(csv_path, 'w') as f: f.write("Mock CSV")
            return MockReport()

        data = BiomassInput(
            predicted_biomass_kg=biomass_kg,
            predicted_biomass_kg_low=predicted_biomass_kg_low,
            predicted_biomass_kg_high=predicted_biomass_kg_high,
            confidence_tier=confidence_tier,
            species=species,
            health=health,
            biomass_confidence=confs["biomass"],
            species_confidence=confs["species"],
            health_confidence=confs["health"]
        )
        report = self.engine.process(data, pdf_path, csv_path)
        generate_pdf_report(report, pdf_path)
        return report
