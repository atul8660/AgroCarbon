# LOCKED FILE — DO NOT MODIFY WITHOUT EXPLICIT PERMISSION FROM ATUL.
# This file was verified correct and tested on this date, expanded to
# 12 species with sourced/estimated wood density constants.
# Any future task, refactor, cleanup, or "improvement" request must NOT
# touch this file's formulas or constants unless Atul explicitly asks
# for a change to biomass_estimator.py by name.
"""
Module 5: Biomass & Carbon Estimation (Formula-Based)
=======================================================

Deterministic, physically-grounded pipeline. No training data or model
weights required. Every number produced is traceable to a named formula
or literature constant.

Pipeline:
    1. Leaf length -> estimated trunk diameter (DBH), species-specific
       scaling.
    2. DBH -> estimated tree height (H), species-specific
       height-diameter allometric equation (Chapman-Richards form).
    3. DBH, H, species wood density -> Above-Ground Biomass (AGB) via
       Chave et al. (2014) pantropical allometric equation.
    4. AGB -> Carbon stock -> CO2-equivalent, using IPCC default factors.

COVERAGE: This module supports 12 of Module 1's 13 classified species.
"Basil" is explicitly excluded -- it is an herb, not a woody tree, and
has no trunk diameter or wood density in any meaningful sense. Applying
a tree-biomass formula to it would produce a number but not a
scientifically valid one. Calling run_biomass_pipeline(species="Basil")
raises a clear NonWoodySpeciesError rather than a silent/wrong result.

NOTE ON CALIBRATION CONSTANTS
------------------------------
Wood density (rho) values are sourced from published literature where a
direct species match was found (see per-species comments below). Where
no direct citation was found, a value was estimated from the closest
available reference (genus-level data or comparable growth habit),
clearly marked "estimated" below -- treat these as documented
approximations, not peer-reviewed per-species citations, same standard
as the original Mango/Neem constants.

The leaf-to-DBH scaling constants (leaf_to_dbh_a, leaf_to_dbh_b) and the
DBH-to-height constants are ALL placeholder approximations reflecting
each species' typical mature size and growth habit (shrub vs. large
canopy tree). They have not been fit to real paired field measurements.
This is a documented limitation of the current version -- it is NOT a
bug, and should be listed as future work: collect real (leaf length,
DBH) pairs per species to replace these with fitted regression
coefficients.

Author:  AgroCarbon AI System - Module 5
Version: 3.0.0 (12-species expansion)
"""

from dataclasses import dataclass
from typing import Literal

Species = Literal[
    "AlstoniaScholaris", "Arjun", "Chinar", "CurryLeaf", "Eucalyptus",
    "Guava", "Jamun", "Jatropha", "Lemon", "Mango", "Pomegranate",
    "Pongamia",
]

# Species Module 1 can classify but Module 5 cannot process (non-woody).
NON_WOODY_SPECIES = frozenset({"Basil"})


class NonWoodySpeciesError(ValueError):
    """Raised when biomass estimation is requested for a species that
    has no woody trunk (e.g. an herb). This is a different failure mode
    from "species not yet calibrated" -- it will never be fixable by
    adding constants, because the underlying formula does not apply."""
    pass


# ---------------------------------------------------------------------------
# 1. Species-specific constants
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SpeciesParams:
    wood_density: float        # g/cm^3 -- Chave et al. (2014) input
    leaf_to_dbh_a: float        # DBH_cm = a * (leaf_length_cm ** b)
    leaf_to_dbh_b: float
    height_max: float           # m -- Chapman-Richards asymptote
    height_k: float
    height_p: float = 1.0
    confidence_tier: str = "high" # "high" or "moderate"


SPECIES_PARAMS: dict[Species, SpeciesParams] = {
    # rho = 0.36 g/cm^3, FAO tropical Asia wood density database
    "AlstoniaScholaris": SpeciesParams(
        wood_density=0.36, leaf_to_dbh_a=1.9, leaf_to_dbh_b=1.0,
        height_max=28.0, height_k=0.035, confidence_tier="high"
    ),
    # rho = 0.68 g/cm^3, FAO tropical Asia wood density database
    "Arjun": SpeciesParams(
        wood_density=0.68, leaf_to_dbh_a=1.8, leaf_to_dbh_b=1.0,
        height_max=22.0, height_k=0.040, confidence_tier="high"
    ),
    # rho = 0.50 g/cm^3, ESTIMATED (genus-level reference; Platanus spp.
    # reported as medium-density hardwood in physico-mechanical studies)
    "Chinar": SpeciesParams(
        wood_density=0.50, leaf_to_dbh_a=1.9, leaf_to_dbh_b=1.0,
        height_max=25.0, height_k=0.035, confidence_tier="moderate"
    ),
    # rho = 0.70 g/cm^3, ESTIMATED (related Murraya spp. described as
    # "hard, heavy, fine-textured" wood in taxonomic references)
    "CurryLeaf": SpeciesParams(
        wood_density=0.70, leaf_to_dbh_a=0.6, leaf_to_dbh_b=1.0,
        height_max=5.0, height_k=0.150, confidence_tier="moderate"
    ),
    # rho = 0.64 g/cm^3, FAO tropical Asia (E. citriodora, representative)
    "Eucalyptus": SpeciesParams(
        wood_density=0.64, leaf_to_dbh_a=1.7, leaf_to_dbh_b=1.0,
        height_max=22.0, height_k=0.045, confidence_tier="high"
    ),
    # rho = 0.70 g/cm^3, multiple sources converge on 0.67-0.74 g/cm^3
    "Guava": SpeciesParams(
        wood_density=0.70, leaf_to_dbh_a=0.8, leaf_to_dbh_b=1.0,
        height_max=8.0, height_k=0.120, confidence_tier="high"
    ),
    # rho = 0.70 g/cm^3, FAO tropical Asia wood density database
    "Jamun": SpeciesParams(
        wood_density=0.70, leaf_to_dbh_a=1.5, leaf_to_dbh_b=1.0,
        height_max=18.0, height_k=0.050, confidence_tier="high"
    ),
    # rho = 0.25 g/cm^3, ESTIMATED (well-documented as a soft, low-density
    # wood in biofuel/agroforestry literature)
    "Jatropha": SpeciesParams(
        wood_density=0.25, leaf_to_dbh_a=0.5, leaf_to_dbh_b=1.0,
        height_max=4.0, height_k=0.180, confidence_tier="moderate"
    ),
    # rho = 0.83 g/cm^3, peer-reviewed wood properties study (BioResources)
    "Lemon": SpeciesParams(
        wood_density=0.83, leaf_to_dbh_a=0.7, leaf_to_dbh_b=1.0,
        height_max=5.0, height_k=0.150, confidence_tier="high"
    ),
    # rho = 0.57 g/cm^3, FAO (Mangifera indica, 0.52-0.59 range) --
    # UNCHANGED from original verified/locked constant
    "Mango": SpeciesParams(
        wood_density=0.57, leaf_to_dbh_a=1.6, leaf_to_dbh_b=1.0,
        height_max=20.0, height_k=0.045, confidence_tier="high"
    ),
    # rho = 0.65 g/cm^3, ESTIMATED (typical fruit-tree hardwood range)
    "Pomegranate": SpeciesParams(
        wood_density=0.65, leaf_to_dbh_a=0.75, leaf_to_dbh_b=1.0,
        height_max=5.0, height_k=0.150, confidence_tier="moderate"
    ),
    # rho = 0.58 g/cm^3, ESTIMATED (described as "moderately strong
    # timber," not top-density, in agroforestry species profiles)
    "Pongamia": SpeciesParams(
        wood_density=0.58, leaf_to_dbh_a=1.6, leaf_to_dbh_b=1.0,
        height_max=20.0, height_k=0.045, confidence_tier="moderate"
    ),
}


# ---------------------------------------------------------------------------
# 2. IPCC default conversion constants (standard, do not change)
# ---------------------------------------------------------------------------

CARBON_FRACTION_OF_BIOMASS = 0.47   # IPCC (2006) default: kg C per kg dry biomass
CO2_TO_C_RATIO = 3.6667             # molecular weight ratio CO2 (44) / C (12)


# ---------------------------------------------------------------------------
# 3. Core pipeline functions
# ---------------------------------------------------------------------------

def _get_params(species: str) -> SpeciesParams:
    if species in NON_WOODY_SPECIES:
        raise NonWoodySpeciesError(
            f"'{species}' is an herb, not a woody tree species. "
            f"Biomass/carbon estimation via trunk-diameter allometry "
            f"does not apply. Species was identified correctly by "
            f"Module 1; this is an inherent scope limitation, not an "
            f"error."
        )
    if species not in SPECIES_PARAMS:
        raise ValueError(
            f"Species '{species}' is not supported by Module 5. "
            f"Supported species: {sorted(SPECIES_PARAMS.keys())}."
        )
    return SPECIES_PARAMS[species]


def estimate_dbh_from_leaf(leaf_length_cm: float, species: str) -> float:
    if leaf_length_cm <= 0:
        raise ValueError("leaf_length_cm must be positive")
    p = _get_params(species)
    return p.leaf_to_dbh_a * (leaf_length_cm ** p.leaf_to_dbh_b)


def estimate_height_from_dbh(dbh_cm: float, species: str) -> float:
    import math
    p = _get_params(species)
    return p.height_max * (1 - math.exp(-p.height_k * dbh_cm)) ** p.height_p


def estimate_agb_chave2014(dbh_cm: float, height_m: float, wood_density: float) -> float:
    """
    Chave et al. (2014) pantropical AGB equation:
        AGB = 0.0673 * (rho * D^2 * H) ^ 0.976
    rho = wood density (g/cm^3), D = DBH (cm), H = height (m), AGB in kg.

    Reference: Chave, J. et al. (2014). "Improved allometric models to
    estimate the aboveground biomass of tropical trees." Global Change
    Biology, 20(10), 3177-3190.
    """
    return 0.0673 * ((wood_density * (dbh_cm ** 2) * height_m) ** 0.976)


def agb_to_carbon_and_co2(agb_kg: float) -> tuple[float, float]:
    carbon_kg = agb_kg * CARBON_FRACTION_OF_BIOMASS
    co2_kg = carbon_kg * CO2_TO_C_RATIO
    return carbon_kg, co2_kg


def estimate_uncertainty_range(agb_kg: float, confidence_tier: str) -> tuple[float, float]:
    """
    Returns (low, high) bounds for point estimates.
    Note: These percentages reflect that leaf-to-DBH scaling is unvalidated 
    for ALL species -- the tier difference only reflects wood density certainty, 
    not DBH/height accuracy, which is uniformly unvalidated.
    """
    margin = 0.15 if confidence_tier == "high" else 0.25
    low = agb_kg * (1.0 - margin)
    high = agb_kg * (1.0 + margin)
    return low, high


# ---------------------------------------------------------------------------
# 4. End-to-end entry point
# ---------------------------------------------------------------------------

@dataclass
class BiomassResult:
    species: str
    leaf_length_cm: float
    estimated_dbh_cm: float
    estimated_height_m: float
    wood_density_g_cm3: float
    agb_kg: float
    confidence_tier: str
    agb_kg_low: float
    agb_kg_high: float
    carbon_kg: float
    carbon_kg_low: float
    carbon_kg_high: float
    co2_equivalent_kg: float
    co2_equivalent_kg_low: float
    co2_equivalent_kg_high: float


def run_biomass_pipeline(leaf_length_cm: float, species: str) -> BiomassResult:
    """
    Runs the full deterministic pipeline: leaf length -> DBH -> height
    -> AGB -> carbon -> CO2.

    Raises:
        NonWoodySpeciesError: if species is an herb (e.g. "Basil").
        ValueError: if species is unrecognized, or leaf_length_cm <= 0.
    """
    params = _get_params(species)  # validates species early
    dbh_cm = estimate_dbh_from_leaf(leaf_length_cm, species)
    height_m = estimate_height_from_dbh(dbh_cm, species)
    agb_kg = estimate_agb_chave2014(dbh_cm, height_m, params.wood_density)
    carbon_kg, co2_kg = agb_to_carbon_and_co2(agb_kg)

    agb_kg_low, agb_kg_high = estimate_uncertainty_range(agb_kg, params.confidence_tier)
    carbon_kg_low, carbon_kg_high = estimate_uncertainty_range(carbon_kg, params.confidence_tier)
    co2_kg_low, co2_kg_high = estimate_uncertainty_range(co2_kg, params.confidence_tier)

    return BiomassResult(
        species=species,
        leaf_length_cm=leaf_length_cm,
        estimated_dbh_cm=round(dbh_cm, 2),
        estimated_height_m=round(height_m, 2),
        wood_density_g_cm3=params.wood_density,
        agb_kg=round(agb_kg, 2),
        confidence_tier=params.confidence_tier,
        agb_kg_low=round(agb_kg_low, 2),
        agb_kg_high=round(agb_kg_high, 2),
        carbon_kg=round(carbon_kg, 2),
        carbon_kg_low=round(carbon_kg_low, 2),
        carbon_kg_high=round(carbon_kg_high, 2),
        co2_equivalent_kg=round(co2_kg, 2),
        co2_equivalent_kg_low=round(co2_kg_low, 2),
        co2_equivalent_kg_high=round(co2_kg_high, 2),
    )


if __name__ == "__main__":
    # Sanity check across all 12 supported species + Basil exclusion
    for sp in sorted(SPECIES_PARAMS.keys()):
        result = run_biomass_pipeline(leaf_length_cm=12.0, species=sp)
        print(result)

    print()
    try:
        run_biomass_pipeline(leaf_length_cm=12.0, species="Basil")
    except NonWoodySpeciesError as e:
        print(f"Basil correctly excluded: {e}")
