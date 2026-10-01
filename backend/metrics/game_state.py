"""Shared clock and season conventions. Public season arguments use ending year."""
import numpy as np
import pandas as pd


def elapsed_seconds(period: pd.Series, remaining: pd.Series) -> pd.Series:
    """Seconds since tipoff; regulation quarters are 720s, overtime periods 300s."""
    return pd.Series(np.where(period <= 4, period * 720 - remaining,
                             2880 + (period - 4) * 300 - remaining), index=period.index)


def clock_seconds(clock: pd.Series) -> pd.Series:
    parts = clock.str.split(':', expand=True).astype(int)
    return parts[0] * 60 + parts[1]


def season_from_game_id(game_id: pd.Series) -> pd.Series:
    """Regular-season NBA IDs encode starting year, independent of filename/date."""
    ids = game_id.astype(str).str.zfill(10)
    if not ids.str.match(r'002\d{7}$').all():
        raise ValueError('Expected regular-season NBA game IDs')
    return ids.str[3:5].astype(int) + 2001
