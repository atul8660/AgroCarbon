def calculate_carbon_stored(biomass_kg: float, carbon_fraction: float) -> float:
    """
    Carbon Stored (kg) = Biomass (kg) * Carbon Fraction
    Standard IPCC carbon fraction is 0.47.
    """
    return round(biomass_kg * carbon_fraction, 2)

def calculate_co2_equivalent(carbon_kg: float, atomic_ratio: float) -> float:
    """
    CO2 Equivalent (kg) = Carbon * (44 / 12)
    This calculates how much atmospheric CO2 was sequestered to form the carbon.
    """
    return round(carbon_kg * atomic_ratio, 2)

def estimate_carbon_credit_value(co2_kg: float, price_per_ton: float) -> float:
    """
    Calculates estimated value in USD based on tonne price.
    Note: 1 Ton = 1000 kg.
    """
    tons_co2 = co2_kg / 1000.0
    return round(tons_co2 * price_per_ton, 2)

def calculate_overall_confidence(biomass_conf: float, species_conf: float, health_conf: float) -> float:
    """
    Computes a harmonic mean or weighted average to get an overall system confidence.
    """
    # Simple weighted average
    overall = (0.5 * biomass_conf) + (0.3 * species_conf) + (0.2 * health_conf)
    return round(overall, 3)
