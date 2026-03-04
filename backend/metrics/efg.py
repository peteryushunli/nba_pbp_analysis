"""eFG% calculation utilities."""


def compute_efg(fgm: int, fg3m: int, fga: int) -> float | None:
    """
    Compute effective field goal percentage.

    eFG% = (FGM + 0.5 * FG3M) / FGA
    """
    if fga == 0:
        return None
    return round((fgm + 0.5 * fg3m) / fga, 4)
