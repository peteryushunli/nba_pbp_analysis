"""Transform raw PBP data into enriched, analysis-ready Parquet files."""

import pandas as pd
import numpy as np
from pathlib import Path

from backend.config import settings

# EVENTMSGTYPE codes from NBA PBP
EVENT_MADE_SHOT = 1
EVENT_MISSED_SHOT = 2
EVENT_FREE_THROW = 3
EVENT_REBOUND = 4
EVENT_TURNOVER = 5
EVENT_FOUL = 6
EVENT_VIOLATION = 7
EVENT_SUBSTITUTION = 8
EVENT_TIMEOUT = 9
EVENT_JUMPBALL = 10
EVENT_EJECTION = 11
EVENT_PERIOD_BEGIN = 12
EVENT_PERIOD_END = 13


def parse_game_clock(pctimestring: pd.Series, period: pd.Series) -> pd.Series:
    """
    Convert PCTIMESTRING ('MM:SS') + PERIOD to TIME_ELAPSED in seconds.

    Regulation: periods 1-4 are 12 min each (720s).
    Overtime: periods 5+ are 5 min each (300s).
    """
    parts = pctimestring.str.split(":", expand=True).astype(int)
    minutes_remaining = parts[0]
    seconds_remaining = parts[1]

    reg_mask = period <= 4
    time_elapsed = pd.Series(0, index=period.index, dtype=int)

    # Regulation: (period-1)*720 + (720 - remaining)
    time_elapsed[reg_mask] = (
        (period[reg_mask] - 1) * 720
        + 720 - minutes_remaining[reg_mask] * 60 - seconds_remaining[reg_mask]
    )

    # Overtime: 48*60 + (period-5)*300 + (300 - remaining)
    ot_mask = ~reg_mask
    time_elapsed[ot_mask] = (
        2880
        + (period[ot_mask] - 5) * 300
        + 300 - minutes_remaining[ot_mask] * 60 - seconds_remaining[ot_mask]
    )

    return time_elapsed


def parse_score_margin(score_margin: pd.Series) -> pd.Series:
    """
    Convert SCOREMARGIN column to ABS_SCORE_DIFF.

    SCOREMARGIN values: 'TIE', signed int strings ('+5', '-3'), or NaN.
    Forward-fills NaN values, starts game at 0.
    """
    def _parse(val):
        if pd.isna(val) or val == "" or val is None:
            return np.nan
        if str(val).upper() == "TIE":
            return 0
        try:
            return abs(int(val))
        except (ValueError, TypeError):
            return np.nan

    result = score_margin.apply(_parse)
    result = result.ffill().fillna(0).astype(int)
    return result


def classify_events(df: pd.DataFrame) -> pd.DataFrame:
    """
    From EVENTMSGTYPE and description columns, extract structured event flags.

    Adds boolean flag columns to the DataFrame (vectorized where possible).

    Key attribution rules:
    - is_ast: PLAYER2 on made shots (EVENTMSGTYPE=1)
    - is_stl: identified from description text on turnover events (EVENTMSGTYPE=5)
    - is_blk: identified from description text on missed shots (EVENTMSGTYPE=2)
    """
    etype = df["EVENTMSGTYPE"]

    # Combine description columns for text matching
    home_desc = df.get("HOMEDESCRIPTION", pd.Series("", index=df.index)).fillna("")
    visitor_desc = df.get("VISITORDESCRIPTION", pd.Series("", index=df.index)).fillna("")
    neutral_desc = df.get("NEUTRALDESCRIPTION", pd.Series("", index=df.index)).fillna("")
    desc_upper = (home_desc + " " + visitor_desc + " " + neutral_desc).str.upper()

    is_3pt = desc_upper.str.contains("3PT", na=False)

    # Field goals
    df["is_fga"] = ((etype == EVENT_MADE_SHOT) | (etype == EVENT_MISSED_SHOT)).astype(int)
    df["is_fgm"] = (etype == EVENT_MADE_SHOT).astype(int)
    df["is_3pa"] = (df["is_fga"] == 1) & is_3pt
    df["is_3pa"] = df["is_3pa"].astype(int)
    df["is_3pm"] = (df["is_fgm"] == 1) & is_3pt
    df["is_3pm"] = df["is_3pm"].astype(int)

    # Free throws
    df["is_fta"] = (etype == EVENT_FREE_THROW).astype(int)
    ft_miss = desc_upper.str.contains("MISS", na=False)
    df["is_ftm"] = ((etype == EVENT_FREE_THROW) & ~ft_miss).astype(int)

    # Rebounds
    is_rebound = etype == EVENT_REBOUND
    is_off_reb = desc_upper.str.contains(r"OFF\.|OFFENSIVE", na=False, regex=True)
    df["is_oreb"] = (is_rebound & is_off_reb).astype(int)
    df["is_dreb"] = (is_rebound & ~is_off_reb).astype(int)

    # Turnovers
    df["is_tov"] = (etype == EVENT_TURNOVER).astype(int)

    # Steals: identified from turnover event descriptions, credited to PLAYER2
    has_steal = desc_upper.str.contains(r"STEAL|STL", na=False, regex=True)
    df["is_stl"] = ((etype == EVENT_TURNOVER) & has_steal).astype(int)

    # Assists: PLAYER2 on made shots (PLAYER2_ID is non-null and non-zero)
    player2_exists = df.get("PLAYER2_ID", pd.Series(0, index=df.index)).fillna(0) != 0
    df["is_ast"] = ((etype == EVENT_MADE_SHOT) & player2_exists).astype(int)

    # Blocks: PLAYER3 on missed shots with block description
    has_block = desc_upper.str.contains(r"BLOCK|BLK", na=False, regex=True)
    df["is_blk"] = ((etype == EVENT_MISSED_SHOT) & has_block).astype(int)

    # Personal fouls
    df["is_pf"] = (etype == EVENT_FOUL).astype(int)

    return df


def transform_pbp_season(raw_pbp: pd.DataFrame) -> pd.DataFrame:
    """
    Full transformation pipeline for a season of raw PBP data.

    Input: Raw PlayByPlayV2 DataFrame.
    Output: Enriched DataFrame ready for Parquet with event classification flags.
    """
    df = raw_pbp.copy()

    # Parse time and score
    df["TIME_ELAPSED"] = parse_game_clock(df["PCTIMESTRING"], df["PERIOD"])
    df["ABS_SCORE_DIFF"] = parse_score_margin(df["SCOREMARGIN"])

    # Parse clock into minutes/seconds remaining
    parts = df["PCTIMESTRING"].str.split(":", expand=True).astype(int)
    df["MINUTES_REMAINING"] = parts[0]
    df["SECONDS_REMAINING"] = parts[1]

    # Classify events
    df = classify_events(df)

    keep_cols = [
        "GAME_ID", "EVENTNUM", "EVENTMSGTYPE", "PERIOD",
        "TIME_ELAPSED", "MINUTES_REMAINING", "SECONDS_REMAINING",
        "ABS_SCORE_DIFF",
        "PLAYER1_ID", "PLAYER1_NAME", "PLAYER1_TEAM_ABBREVIATION",
        "PLAYER2_ID", "PLAYER2_NAME", "PLAYER2_TEAM_ABBREVIATION",
        "PLAYER3_ID", "PLAYER3_NAME",
        "is_fga", "is_fgm", "is_3pa", "is_3pm",
        "is_fta", "is_ftm",
        "is_oreb", "is_dreb",
        "is_ast", "is_tov", "is_stl", "is_blk", "is_pf",
    ]

    # Only keep columns that exist (some may be missing in older data)
    available = [c for c in keep_cols if c in df.columns]
    return df[available]


def process_season(
    season: int,
    input_dir: Path | None = None,
    output_dir: Path | None = None,
    force: bool = False,
) -> Path:
    """
    Read raw PBP Parquet, transform, and save enriched events Parquet.

    Returns path to the output file.
    """
    input_dir = input_dir or settings.DATA_DIR / "raw" / "pbp"
    output_dir = output_dir or settings.DATA_DIR / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)

    in_path = input_dir / f"play_by_play_{season}.parquet"
    out_path = output_dir / f"events_{season}.parquet"

    if out_path.exists() and not force:
        print(f"Processed events already exist: {out_path}")
        return out_path

    if not in_path.exists():
        raise FileNotFoundError(f"Raw PBP not found: {in_path}. Run `fetch` first.")

    print(f"Processing PBP for season {season}...")
    raw = pd.read_parquet(in_path)
    events = transform_pbp_season(raw)
    events.to_parquet(out_path, engine="pyarrow", compression="snappy", index=False)
    print(f"Saved {len(events)} events to {out_path}")
    return out_path
