# Module 6: Carbon Analytics & Environmental Reporting Engine

The final node in the AI-Powered Tree Carbon Assessment System.

Module 6 consumes the Above Ground Biomass (AGB) calculations from Module 5, alongside categorical metadata from Modules 1-4, to generate scientifically robust estimates of:
- Carbon Stored (kg)
- CO2 Equivalent Absorbed (kg)
- Educational Carbon Credit Estimates

## Disclaimer
The carbon credit values produced by this engine are **Educational Estimates Only**. Official certified carbon credits require third-party independent field verification, permanence guarantees, and baseline additionality assessments in accordance with standards like Verra or Gold Standard. 

## Scientific Basis
We adhere to the IPCC (Intergovernmental Panel on Climate Change) standards for Carbon Fractions.
- **Carbon Stored**: AGB * Default Carbon Fraction (0.47)
- **CO2 Equivalent**: Carbon * (44 / 12) (Stoichiometric Conversion)

## Structure
- `/api`: FastAPI servers and Pydantic validation schemas.
- `/exports`: ReportLab PDF and Pandas CSV generators.
- `/utils`: Strict math utilities for conversion execution.

## Usage
`pip install -r requirements.txt`

`uvicorn api.main:app --reload`
Navigate to `http://localhost:8000/docs` to test the API endpoint.
