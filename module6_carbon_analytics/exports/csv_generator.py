import os
import pandas as pd
from api.schemas import CarbonReportOutput

def generate_csv_report(report: CarbonReportOutput, output_path: str):
    """
    Converts the output into a single row DataFrame and appends/creates a CSV.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    data = report.model_dump()
    df = pd.DataFrame([data])
    
    # If the CSV exists, append without header. If not, write with header.
    if os.path.exists(output_path):
        df.to_csv(output_path, mode='a', header=False, index=False)
    else:
        df.to_csv(output_path, mode='w', header=True, index=False)
        
    return output_path
