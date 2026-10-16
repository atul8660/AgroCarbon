from omegaconf import OmegaConf

try:
    # When called from the backend (ROOT_DIR on sys.path)
    from module6_carbon_analytics.api.schemas import BiomassInput, CarbonReportOutput
    from module6_carbon_analytics.utils.math_utils import (
        calculate_carbon_stored,
        calculate_co2_equivalent,
        estimate_carbon_credit_value,
        calculate_overall_confidence
    )
except ImportError:
    # When run standalone from within module6_carbon_analytics/
    from api.schemas import BiomassInput, CarbonReportOutput
    from utils.math_utils import (
        calculate_carbon_stored,
        calculate_co2_equivalent,
        estimate_carbon_credit_value,
        calculate_overall_confidence
    )


class CarbonEngine:
    """
    The core controller for converting AGB into environmental metrics.
    """
    def __init__(self, config_path: str = "configs/config.yaml"):
        self.cfg = OmegaConf.load(config_path)

    def process(self, data: BiomassInput, pdf_path: str, csv_path: str) -> CarbonReportOutput:
        """
        Executes the logical flow for Carbon conversion and returns the unified Report.
        """
        carbon = calculate_carbon_stored(
            data.predicted_biomass_kg, 
            self.cfg.constants.carbon_fraction
        )
        
        co2 = calculate_co2_equivalent(
            carbon, 
            self.cfg.constants.atomic_ratio_co2_to_c
        )
        
        credit = estimate_carbon_credit_value(
            co2, 
            self.cfg.market.estimated_price_per_ton_co2
        )
        
        # Calculate boundaries for carbon and co2 based on passed in biomass boundaries
        carbon_low = calculate_carbon_stored(data.predicted_biomass_kg_low, self.cfg.constants.carbon_fraction)
        carbon_high = calculate_carbon_stored(data.predicted_biomass_kg_high, self.cfg.constants.carbon_fraction)
        co2_low = calculate_co2_equivalent(carbon_low, self.cfg.constants.atomic_ratio_co2_to_c)
        co2_high = calculate_co2_equivalent(carbon_high, self.cfg.constants.atomic_ratio_co2_to_c)
        credit_low = estimate_carbon_credit_value(co2_low, self.cfg.market.estimated_price_per_ton_co2)
        credit_high = estimate_carbon_credit_value(co2_high, self.cfg.market.estimated_price_per_ton_co2)

        overall_conf = calculate_overall_confidence(
            data.biomass_confidence, 
            data.species_confidence, 
            data.health_confidence
        )
        
        return CarbonReportOutput(
            species=data.species,
            health=data.health,
            predicted_biomass_kg=round(data.predicted_biomass_kg, 2),
            carbon_stored_kg=carbon,
            co2_equivalent_kg=co2,
            estimated_carbon_credit=credit,
            
            confidence_tier=data.confidence_tier,
            estimated_dbh_cm=data.estimated_dbh_cm,
            estimated_height_m=data.estimated_height_m,
            wood_density=data.wood_density,
            
            predicted_biomass_kg_low=round(data.predicted_biomass_kg_low, 2),
            predicted_biomass_kg_high=round(data.predicted_biomass_kg_high, 2),
            carbon_stored_kg_low=carbon_low,
            carbon_stored_kg_high=carbon_high,
            co2_equivalent_kg_low=co2_low,
            co2_equivalent_kg_high=co2_high,
            estimated_carbon_credit_low=credit_low,
            estimated_carbon_credit_high=credit_high,
            
            overall_confidence=overall_conf,
            prediction_time=data.timestamp.isoformat(),
            report_pdf=pdf_path,
            report_csv=csv_path
        )
