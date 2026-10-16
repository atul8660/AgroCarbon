import os
import time
from fastapi import FastAPI
from api.schemas import BiomassInput, CarbonReportOutput
from api.engine import CarbonEngine
from exports.pdf_generator import generate_pdf_report
from exports.csv_generator import generate_csv_report
from omegaconf import OmegaConf

cfg = OmegaConf.load("configs/config.yaml")

app = FastAPI(
    title=cfg.api.title,
    description="Transforms Biomass predictions into Environmental and Carbon intelligence."
)

engine = CarbonEngine("configs/config.yaml")

@app.post("/api/v1/predict_carbon", response_model=CarbonReportOutput)
def calculate_carbon_metrics(data: BiomassInput):
    """
    Main endpoint that ties all modules together.
    Ingests Biomass/Species/Health and outputs final carbon metrics and reports.
    """
    timestamp_fs = int(time.time())
    
    pdf_path = os.path.join(cfg.paths.pdf_exports_dir, f"report_{timestamp_fs}.pdf")
    csv_path = os.path.join(cfg.paths.csv_exports_dir, "master_carbon_log.csv")
    
    # 1. Process Core Logic
    report_output = engine.process(data, pdf_path, csv_path)
    
    # 2. Generate PDF and CSV Side Effects
    generate_pdf_report(report_output, pdf_path)
    generate_csv_report(report_output, csv_path)
    
    return report_output
