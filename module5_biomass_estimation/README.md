# Module 5 — Biomass Estimation (Formula-Based)

> **Version 2.0** — Formula-based replacement of the previous neural-network approach.
> This module implements the peer-reviewed **Chave et al. (2014)** allometric equation
> for above-ground biomass (AGB) estimation in tropical trees.

---

## Why Formula-Based, Not ML-Based

The previous version of Module 5 used a trained convolutional neural network
(ConvNeXt backbone + fusion head + regression head) to predict biomass directly
from an image.  That approach was removed for two reasons:

1. **Lack of a labelled training corpus.**  Accurate end-to-end image→biomass
   regression requires thousands of paired (image, field-measured AGB) samples.
   No such dataset was available for the target species.

2. **Scientific reproducibility.**  The allometric equation published by
   Chave et al. (2014) is:
   - Derived from 58 tropical forest sites worldwide (> 4,000 trees measured)
   - Peer-reviewed and widely adopted in IPCC carbon accounting frameworks
   - Deterministic and fully auditable — every output can be back-calculated from
     the input parameters

### Citation

> Chave, J., Réjou-Méchain, M., Búrquez, A., Chidumayo, E., Colgan, M. S.,
> Delitti, W. B. C., … Vieilledent, G. (2014).
> **Improved allometric models to estimate the above-ground biomass of tropical trees.**
> *Global Change Biology*, **20**(10), 3177–3190.
> https://doi.org/10.1111/gcb.12629

---

## The 4-Step Pipeline

```
Leaf length (cm)
    │
    │  Step 1 — Power-law scaling (species-specific)
    │  log₁₀(DBH) = a + b · log₁₀(leaf_length)
    ▼
DBH — Diameter at Breast Height (cm)
    │
    │  Step 2 — Weibull allometric model (species-specific)
    │  H = a · (1 − exp(−b · DBH^c))
    ▼
Height (m)
    │
    │  Step 3 — Chave et al. (2014) Eq. 4
    │  AGB = 0.0673 · (ρ · D² · H)^0.976
    ▼
Above-Ground Biomass — AGB (kg)
    │
    │  Step 4 — IPCC conversion factors
    │  carbon_kg = AGB × 0.47
    │  co2_kg    = carbon_kg × 3.6667   (44⁄12)
    ▼
carbon_kg  |  co2_kg
```

### Step-by-step description

| Step | Function | Input | Output |
|------|----------|-------|--------|
| 1 | `estimate_dbh_from_leaf()` | leaf length (cm) | DBH (cm) |
| 2 | `estimate_height_from_dbh()` | DBH (cm) | height (m) |
| 3 | `estimate_agb_chave2014()` | DBH, height, wood density | AGB (kg) |
| 4 | `agb_to_carbon_and_co2()` | AGB (kg) | carbon (kg), CO₂-eq (kg) |

The convenience wrapper **`run_biomass_pipeline(leaf_length_cm, species)`**
executes all four steps and returns a single result dictionary.

---

## Supported Species

| Species | Wood density (g cm⁻³) | Source |
|---------|----------------------|--------|
| Mango (*Mangifera indica*) | 0.52 | Global Wood Density Database |
| Neem (*Azadirachta indica*) | 0.56 | Global Wood Density Database |

Additional species can be added to the `SPECIES_PARAMS` dictionary in
`biomass_estimator.py`.

---

## ⚠️ Documented Limitation — Placeholder Leaf-to-DBH Constants

The **leaf-to-DBH power-law scaling constants** (`leaf_to_dbh_a`, `leaf_to_dbh_b`)
for both Mango and Neem are **current placeholders**, not field-validated values.
They are derived from plausible literature ranges for tropical species and are
designed to produce biologically reasonable DBH estimates for typical leaf lengths.

> **This is a documented limitation, not a bug.**

### Why they are placeholders

Accurately mapping leaf size to DBH requires paired field measurement data:
concurrent observations of mean leaf length and measured trunk diameter for a
statistically significant sample of trees.  This data was not available at the
time of implementation.

### How to fix this

1. Collect field measurements: for each tree, record `(mean_leaf_length_cm, DBH_cm)`.
2. Fit a log-linear regression:
   `log₁₀(DBH) = a + b · log₁₀(leaf_length)` via least squares.
3. Replace `leaf_to_dbh_a` and `leaf_to_dbh_b` in `SPECIES_PARAMS` with the
   regression coefficients.  No other code needs to change.

The Chave et al. (2014) AGB formula (Step 3) and the IPCC carbon factors
(Step 4) are **not placeholders** — they are established scientific constants.

---

## Running the Tests

```powershell
# From inside this folder
python -m pytest test_biomass_estimator.py -v
```

The test suite verifies:
- Positive outputs for Mango and Neem via `run_biomass_pipeline()`
- `ValueError` raised for non-positive leaf lengths
- Internal formula consistency (`carbon = AGB × 0.47`, `CO₂ = carbon × 3.6667`)
- Edge cases (very small / very large leaf inputs)
- Hand-computed reference value for the Chave 2014 Eq. 4 formula

---

## Module Structure

```
module5_biomass_estimation/
├── biomass_estimator.py        ← core pipeline (Chave 2014 + helpers)
├── README.md                   ← this file
└── test_biomass_estimator.py   ← pytest test suite
```

---

## Relationship to Other Modules

| Upstream | Provides | Downstream |
|----------|---------|------------|
| Module 1 (Species Classification) | `species` string | this module |
| Module 4 (Segmentation) | leaf region for measurement | this module |
| **This module** | `agb_kg`, `carbon_kg`, `co2_kg` | **Module 6 (Carbon Analytics)** |
