"""Possession estimation from box score / event data."""


def estimate_possessions(fga: int, oreb: int, tov: int, fta: int) -> float:
    """
    Estimate possessions using the standard formula.

    Possessions = FGA - OREB + TOV + 0.44 * FTA

    The 0.44 coefficient accounts for and-ones, technical FTs,
    and other non-possession-ending free throws.
    """
    return fga - oreb + tov + 0.44 * fta
