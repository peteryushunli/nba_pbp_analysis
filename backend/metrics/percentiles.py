"""Percentile computation for on/off efficiency stats."""

import pandas as pd


def compute_on_off_percentiles(df: pd.DataFrame, position: str = "All") -> pd.DataFrame:
    """
    Compute net rating and percentiles for on/off differential stats.

    Adds columns: net_on, net_off, net_diff, off_diff_pctl, def_diff_pctl, net_diff_pctl.
    Percentiles computed within position group (G/F/C) or all players.
    100th percentile = best performance.

    Directionality:
    - off_diff: higher is better (you boost offense) → ascending
    - def_diff: lower is better (you improve defense) → descending
    - net_diff: higher is better (you boost net rating) → ascending
    """
    result = df.copy()

    # Compute net rating columns
    result["net_on"] = result["off_rating_on"] - result["def_rating_on"]
    result["net_off"] = result["off_rating_off"] - result["def_rating_off"]
    result["net_diff"] = result["net_on"] - result["net_off"]

    if position and position != "All":
        mask = result["position"] == position
    else:
        mask = pd.Series(True, index=result.index)

    subset = result.loc[mask].copy()

    pctl_cols = ["off_diff_pctl", "def_diff_pctl", "net_diff_pctl"]

    if len(subset) < 2:
        for col in pctl_cols:
            result.loc[mask, col] = 50.0
        return result

    # Offense diff: higher is better
    subset["off_diff_pctl"] = subset["off_diff"].rank(pct=True) * 100
    # Defense diff: lower is better (you improve defense)
    subset["def_diff_pctl"] = (1 - subset["def_diff"].rank(pct=True)) * 100
    # Net diff: higher is better
    subset["net_diff_pctl"] = subset["net_diff"].rank(pct=True) * 100

    result.loc[mask, pctl_cols] = subset[pctl_cols]

    for col in pctl_cols:
        result[col] = result[col].clip(0, 100)

    return result
