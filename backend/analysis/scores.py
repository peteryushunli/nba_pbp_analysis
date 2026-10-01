"""Chronological cumulative scores from legacy NBA event logs."""
import pandas as pd
from backend.metrics.game_state import clock_seconds


def final_scores(events: pd.DataFrame) -> pd.Series:
    # A correction can have an event ID greater than the game-end marker while
    # referring to an earlier quarter. Event IDs alone do not order game time.
    scored=events.dropna(subset=['SCORE']).copy()
    scored['clock_seconds']=clock_seconds(scored.PCTIMESTRING)
    return scored.sort_values(['GAME_ID','PERIOD','clock_seconds','EVENTNUM'],
        ascending=[True,True,False,True],kind='stable').groupby('GAME_ID').SCORE.last()
