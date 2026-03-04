"""Sample size handling: padding, thresholds, and confidence levels."""


# Padding constants from basketball analytics research
PADDING_CONSTANTS = {
    "efg": 74,       # ~74 FGA for FG-based stats to stabilize
    "rating": 50,    # ORtg/DRtg stabilization
}

MINIMUM_THRESHOLDS = {
    "efg": 20,       # Gray out cells with < 20 FGA
    "rating": 20,    # Mark ratings with < 20 possessions
}


def apply_padding(
    observed_value: float | None,
    sample_size: int,
    league_average: float,
    stat_type: str = "efg",
) -> float | None:
    """
    Apply Bayesian-style padding to blend observed value toward league average.

    Formula: padded = (observed * n + padding * league_avg) / (n + padding)

    As n -> infinity, padded -> observed.
    As n -> 0, padded -> league_avg.
    """
    if observed_value is None:
        return None

    padding = PADDING_CONSTANTS.get(stat_type, 74)
    padded = (observed_value * sample_size + padding * league_average) / (sample_size + padding)
    return round(padded, 4)


def is_below_threshold(sample_size: int, stat_type: str = "efg") -> bool:
    """Check if sample size is below the minimum threshold."""
    threshold = MINIMUM_THRESHOLDS.get(stat_type, 20)
    return sample_size < threshold


def confidence_level(sample_size: int, stat_type: str = "efg") -> str:
    """Return 'low' (<threshold), 'medium' (<padding), or 'high' (>=padding)."""
    padding = PADDING_CONSTANTS.get(stat_type, 74)
    threshold = MINIMUM_THRESHOLDS.get(stat_type, 20)
    if sample_size < threshold:
        return "low"
    elif sample_size < padding:
        return "medium"
    else:
        return "high"
