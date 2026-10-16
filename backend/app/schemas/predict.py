from pydantic import BaseModel, Field

class CarbonAnalysisResponse(BaseModel):
    species: str = Field(..., description="Detected Tree Species")
    species_confidence: float = Field(..., description="Module 1 Confidence")
    health: str = Field(..., description="Tree Health Status")
    health_confidence: float = Field(..., description="Module 2 Confidence")
    
    predicted_biomass_kg: float = Field(..., description="Module 5 AGB")
    carbon_stored_kg: float = Field(..., description="Module 6 Carbon")
    co2_equivalent_kg: float = Field(..., description="Estimated CO2 equivalent stored")
    estimated_carbon_credit: float = Field(..., description="Projected educational financial carbon credit equivalent")
    # Boundary Ranges
    confidence_tier: str = Field(default="moderate", description="high or moderate")
    
    # Calculation Details
    estimated_dbh_cm: float = Field(default=0.0)
    estimated_height_m: float = Field(default=0.0)
    wood_density: float = Field(default=0.0)
    
    predicted_biomass_kg_low: float = Field(default=0.0)
    predicted_biomass_kg_high: float = Field(default=0.0)
    carbon_stored_kg_low: float = Field(default=0.0)
    carbon_stored_kg_high: float = Field(default=0.0)
    co2_equivalent_kg_low: float = Field(default=0.0)
    co2_equivalent_kg_high: float = Field(default=0.0)
    estimated_carbon_credit_low: float = Field(default=0.0)
    estimated_carbon_credit_high: float = Field(default=0.0)
    
    overall_confidence: float = Field(..., description="Pipeline overall confidence")
    
    # Internal Disk Legacy Properties (Do Not Remove, used by PipelineService)
    report_pdf: str | None = Field(default=None, description="Internal path")
    report_csv: str | None = Field(default=None, description="Internal path")
    
    # S3 Asset Pointers
    full_tree_image_url: str | None = Field(default=None, description="AWS S3 public URL for uploaded full tree")
    closeup_image_url: str | None = Field(default=None, description="AWS S3 public URL for uploaded closeup")
    pdf_report_url: str | None = Field(default=None, description="AWS S3 public URL to the PDF report")
    csv_report_url: str | None = Field(default=None, description="AWS S3 public URL to the CSV log")

    # DB-populated after persistence — returned to client for reference
    prediction_id: int | None = Field(default=None, description="Database row ID of the saved prediction")
    prediction_timestamp: str | None = Field(default=None, description="ISO-8601 timestamp of when the prediction was saved")

from datetime import datetime
from pydantic import ConfigDict

class PredictionHistoryResponse(BaseModel):
    id: int
    user_id: int
    species: str
    health: str
    biomass_kg: float
    carbon_kg: float
    co2_kg: float
    estimated_carbon_credit: float
    
    confidence_tier: str = "moderate"
    biomass_kg_low: float = 0.0
    biomass_kg_high: float = 0.0
    carbon_kg_low: float = 0.0
    carbon_kg_high: float = 0.0
    co2_kg_low: float = 0.0
    co2_kg_high: float = 0.0
    estimated_carbon_credit_low: float = 0.0
    estimated_carbon_credit_high: float = 0.0
    
    prediction_date: datetime
    
    full_tree_url: str | None = None
    closeup_url: str | None = None
    pdf_report_url: str | None = None
    csv_report_url: str | None = None
    
    model_config = ConfigDict(from_attributes=True)

