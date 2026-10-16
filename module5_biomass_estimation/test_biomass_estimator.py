"""
test_biomass_estimator.py
=========================
Unit tests for the formula-based biomass estimation pipeline (v3.0.0).
Covers all 12 supported woody species + Basil exclusion.

Reference: Chave et al. (2014) allometric equation (Eq. 4).

Run with:
    python -m pytest test_biomass_estimator.py -v
"""

import math
import pytest
from biomass_estimator import (
    run_biomass_pipeline,
    estimate_dbh_from_leaf,
    estimate_height_from_dbh,
    estimate_agb_chave2014,
    agb_to_carbon_and_co2,
    CARBON_FRACTION_OF_BIOMASS,
    CO2_TO_C_RATIO,
    SPECIES_PARAMS,
    NON_WOODY_SPECIES,
    NonWoodySpeciesError,
    BiomassResult,
)

# ---------------------------------------------------------------------------
# Fixtures / shared data
# ---------------------------------------------------------------------------
VALID_LEAF_CM = 20.0      # a realistic tropical leaf length
SUPPORTED_SPECIES = sorted(SPECIES_PARAMS.keys())  # all 12
assert len(SUPPORTED_SPECIES) == 12, f"Expected 12 species, got {len(SUPPORTED_SPECIES)}"


# ===========================================================================
# 1. run_biomass_pipeline() — positive-value tests for ALL 12 species
# ===========================================================================
class TestRunBiomassPipelinePositiveValues:
    """All pipeline outputs must be positive finite floats for valid inputs."""

    @pytest.mark.parametrize("species", SUPPORTED_SPECIES)
    def test_dbh_positive(self, species):
        result = run_biomass_pipeline(VALID_LEAF_CM, species)
        assert result.estimated_dbh_cm > 0, f"dbh_cm should be > 0 for {species}"

    @pytest.mark.parametrize("species", SUPPORTED_SPECIES)
    def test_height_positive(self, species):
        result = run_biomass_pipeline(VALID_LEAF_CM, species)
        assert result.estimated_height_m > 0, f"height_m should be > 0 for {species}"

    @pytest.mark.parametrize("species", SUPPORTED_SPECIES)
    def test_agb_positive(self, species):
        result = run_biomass_pipeline(VALID_LEAF_CM, species)
        assert result.agb_kg > 0, f"agb_kg should be > 0 for {species}"

    @pytest.mark.parametrize("species", SUPPORTED_SPECIES)
    def test_carbon_positive(self, species):
        result = run_biomass_pipeline(VALID_LEAF_CM, species)
        assert result.carbon_kg > 0, f"carbon_kg should be > 0 for {species}"

    @pytest.mark.parametrize("species", SUPPORTED_SPECIES)
    def test_co2_positive(self, species):
        result = run_biomass_pipeline(VALID_LEAF_CM, species)
        assert result.co2_equivalent_kg > 0, f"co2_kg should be > 0 for {species}"

    @pytest.mark.parametrize("species", SUPPORTED_SPECIES)
    def test_all_values_are_finite(self, species):
        result = run_biomass_pipeline(VALID_LEAF_CM, species)
        for attr in ("estimated_dbh_cm", "estimated_height_m", "agb_kg", "carbon_kg", "co2_equivalent_kg"):
            assert math.isfinite(getattr(result, attr)), f"{attr} is not finite for {species}"

    @pytest.mark.parametrize("species", SUPPORTED_SPECIES)
    def test_species_echoed(self, species):
        result = run_biomass_pipeline(VALID_LEAF_CM, species)
        assert result.species == species

    @pytest.mark.parametrize("species", SUPPORTED_SPECIES)
    def test_leaf_length_echoed(self, species):
        result = run_biomass_pipeline(VALID_LEAF_CM, species)
        assert result.leaf_length_cm == VALID_LEAF_CM

    @pytest.mark.parametrize("species", SUPPORTED_SPECIES)
    def test_returns_biomass_result(self, species):
        result = run_biomass_pipeline(VALID_LEAF_CM, species)
        assert isinstance(result, BiomassResult)


# ===========================================================================
# 2. estimate_dbh_from_leaf() — ValueError for non-positive leaf length
# ===========================================================================
class TestEstimateDbhFromLeafErrors:

    @pytest.mark.parametrize("bad_leaf", [0.0, -1.0, -100.0, -0.001])
    def test_raises_value_error_for_non_positive_leaf(self, bad_leaf):
        """Must raise ValueError for leaf_length_cm <= 0."""
        with pytest.raises(ValueError):
            estimate_dbh_from_leaf(bad_leaf, "Mango")

    def test_raises_value_error_zero_eucalyptus(self):
        with pytest.raises(ValueError):
            estimate_dbh_from_leaf(0.0, "Eucalyptus")

    def test_does_not_raise_for_positive_leaf(self):
        """Should succeed for any positive value."""
        result = estimate_dbh_from_leaf(15.0, "Mango")
        assert result > 0

    def test_raises_value_error_for_unknown_species(self):
        """Should raise ValueError for an unrecognised species."""
        with pytest.raises(ValueError):
            estimate_dbh_from_leaf(20.0, "UnknownTree")


# ===========================================================================
# 3. Basil exclusion — NonWoodySpeciesError (subclass of ValueError)
# ===========================================================================
class TestBasilExclusion:

    def test_basil_raises_non_woody_error(self):
        """Basil must raise NonWoodySpeciesError, not a generic ValueError."""
        with pytest.raises(NonWoodySpeciesError):
            run_biomass_pipeline(leaf_length_cm=12.0, species="Basil")

    def test_basil_error_is_value_error_subclass(self):
        """NonWoodySpeciesError should also be caught by except ValueError."""
        with pytest.raises(ValueError):
            run_biomass_pipeline(leaf_length_cm=12.0, species="Basil")

    def test_basil_error_message_mentions_herb(self):
        """The error message should explain why Basil is excluded."""
        with pytest.raises(NonWoodySpeciesError, match="herb"):
            run_biomass_pipeline(leaf_length_cm=12.0, species="Basil")

    def test_basil_in_non_woody_set(self):
        assert "Basil" in NON_WOODY_SPECIES

    def test_basil_not_in_species_params(self):
        assert "Basil" not in SPECIES_PARAMS

    def test_dbh_also_rejects_basil(self):
        with pytest.raises(NonWoodySpeciesError):
            estimate_dbh_from_leaf(10.0, "Basil")

    def test_height_also_rejects_basil(self):
        with pytest.raises(NonWoodySpeciesError):
            estimate_height_from_dbh(15.0, "Basil")


# ===========================================================================
# 4. Internal consistency: carbon = AGB × 0.47, CO₂ = carbon × 3.6667
# ===========================================================================
class TestInternalConsistency:

    @pytest.mark.parametrize("species", SUPPORTED_SPECIES)
    def test_carbon_equals_agb_times_carbon_fraction(self, species):
        result = run_biomass_pipeline(VALID_LEAF_CM, species)
        expected_carbon = result.agb_kg * CARBON_FRACTION_OF_BIOMASS
        assert math.isclose(result.carbon_kg, expected_carbon, rel_tol=1e-2), (
            f"carbon_kg mismatch for {species}: "
            f"got {result.carbon_kg}, expected {expected_carbon}"
        )

    @pytest.mark.parametrize("species", SUPPORTED_SPECIES)
    def test_co2_equals_carbon_times_conversion_factor(self, species):
        result = run_biomass_pipeline(VALID_LEAF_CM, species)
        expected_co2 = result.carbon_kg * CO2_TO_C_RATIO
        assert math.isclose(result.co2_equivalent_kg, expected_co2, rel_tol=1e-2), (
            f"co2_kg mismatch for {species}: "
            f"got {result.co2_equivalent_kg}, expected {expected_co2}"
        )

    @pytest.mark.parametrize("species", SUPPORTED_SPECIES)
    def test_co2_larger_than_carbon(self, species):
        result = run_biomass_pipeline(VALID_LEAF_CM, species)
        assert result.co2_equivalent_kg > result.carbon_kg

    @pytest.mark.parametrize("species", SUPPORTED_SPECIES)
    def test_carbon_smaller_than_agb(self, species):
        result = run_biomass_pipeline(VALID_LEAF_CM, species)
        assert result.carbon_kg < result.agb_kg

    def test_agb_to_carbon_and_co2_directly(self):
        agb = 500.0
        carbon, co2 = agb_to_carbon_and_co2(agb)
        assert math.isclose(carbon, agb * CARBON_FRACTION_OF_BIOMASS, rel_tol=1e-12)
        assert math.isclose(co2, carbon * CO2_TO_C_RATIO, rel_tol=1e-12)


# ===========================================================================
# 5. Additional robustness / edge-case tests
# ===========================================================================
class TestEdgeCases:

    @pytest.mark.parametrize("species", SUPPORTED_SPECIES)
    def test_very_small_leaf(self, species):
        result = run_biomass_pipeline(0.1, species)
        for attr in ("estimated_dbh_cm", "estimated_height_m", "agb_kg", "carbon_kg", "co2_equivalent_kg"):
            assert getattr(result, attr) >= 0.0

    def test_very_large_leaf(self):
        result = run_biomass_pipeline(200.0, "Mango")
        for attr in ("estimated_dbh_cm", "estimated_height_m", "agb_kg", "carbon_kg", "co2_equivalent_kg"):
            assert getattr(result, attr) > 0
            assert math.isfinite(getattr(result, attr))

    def test_chave2014_known_value(self):
        """
        Smoke-test Chave 2014 Eq.4 against a hand-computed reference.
        DBH=30 cm, H=15 m, ρ=0.52 g/cm³:
            AGB = 0.0673 × (0.52 × 900 × 15)^0.976
                = 0.0673 × (7020)^0.976
        """
        agb = estimate_agb_chave2014(dbh_cm=30.0, height_m=15.0, wood_density=0.52)
        expected = 0.0673 * (0.52 * (30.0 ** 2) * 15.0) ** 0.976
        assert math.isclose(agb, expected, rel_tol=1e-9)

    def test_species_count(self):
        """Exactly 12 species should be supported."""
        assert len(SPECIES_PARAMS) == 12

    def test_unknown_species_raises(self):
        with pytest.raises(ValueError, match="not supported"):
            run_biomass_pipeline(12.0, "BananaTree")
