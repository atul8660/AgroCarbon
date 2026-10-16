import os
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

try:
    from module6_carbon_analytics.api.schemas import CarbonReportOutput
except ImportError:
    from api.schemas import CarbonReportOutput


def generate_pdf_report(report: CarbonReportOutput, output_path: str):
    """
    Generates a professional PDF containing all tree statistics and disclaimers.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    c = canvas.Canvas(output_path, pagesize=letter)
    
    # Title
    c.setFont("Helvetica-Bold", 16)
    c.drawString(100, 750, "AgroCarbonAI: Mobile Based Carbon Estimation System")
    
    # Subtitle
    c.setFont("Helvetica", 12)
    c.drawString(100, 730, "Module 6: Environmental Impact Report")
    
    # Body
    c.setFont("Helvetica", 11)
    y = 690
    lines = [
        f"Prediction Time: {report.prediction_time}",
        f"Tree Species: {report.species}",
        f"Health Status: {report.health}",
        "",
        f"1. Predicted Above Ground Biomass (AGB): {report.predicted_biomass_kg} kg",
        f"2. Carbon Stored (IPCC 0.47 factor): {report.carbon_stored_kg} kg",
        f"3. CO2 Equivalent Absorbed: {report.co2_equivalent_kg} kg",
        "",
        f"Educational Estimated Carbon Credit Value: \u20b9{report.estimated_carbon_credit} INR",
        "",
        f"Overall System Confidence: {report.overall_confidence * 100:.2f}%",
    ]
    
    for line in lines:
        c.drawString(100, y, line)
        y -= 20
        
    # Disclaimer block
    c.setFont("Helvetica-Oblique", 9)
    y -= 30
    c.drawString(100, y, "DISCLAIMER: " + report.disclaimer)
    c.drawString(100, y - 12, "This system provides automated estimations for research purposes and does not")
    c.drawString(100, y - 24, "represent third-party verified, tradable carbon assets.")
    
    c.save()
    return output_path
