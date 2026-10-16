from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime

class BiomassInput(BaseModel):
    predicted_biomass_kg: float = Field(..., description="AGB from Module 5&6")
    species: str = Field(..., description="Species from Module 1")
    health: str = Field(..., description="Health status from Module 2")
    biomass_confidence: float = Field(..., ge=0.0, le=1.0)
    species_confidence: float = Field(..., ge=0.0, le=1.0)
    health_confidence: float = Field(..., ge=0.0, le=1.0)
    gps_coordinates: Optional[str] = Field("Unknown", description="Optional GPS Location")
    timestamp: datetime = Field(default_factory=datetime.now)
    
    confidence_tier: str = Field("moderate", description="High or moderate confidence")
    estimated_dbh_cm: float = Field(0.0)
    estimated_height_m: float = Field(0.0)
    wood_density: float = Field(0.0)
    
    predicted_biomass_kg_low: float = Field(0.0)
    predicted_biomass_kg_high: float = Field(0.0)

class CarbonReportOutput(BaseModel):
    species: str
    health: str
    predicted_biomass_kg: float
    carbon_stored_kg: float
    co2_equivalent_kg: float
    estimated_carbon_credit: float
    
    confidence_tier: str
    estimated_dbh_cm: float = 0.0
    estimated_height_m: float = 0.0
    wood_density: float = 0.0
    
    predicted_biomass_kg_low: float
    predicted_biomass_kg_high: float
    carbon_stored_kg_low: float
    carbon_stored_kg_high: float
    co2_equivalent_kg_low: float
    co2_equivalent_kg_high: float
    estimated_carbon_credit_low: float
    estimated_carbon_credit_high: float
    
    overall_confidence: float
    prediction_time: str
    report_pdf: str
    report_csv: str
    disclaimer: str = "Values are educational estimates. Official carbon credits require third-party verification."
