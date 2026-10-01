"""Transform raw PBP data into enriched, analysis-ready Parquet files."""

import pandas as pd
from pathlib import Path

from backend.config import settings
from backend.metrics.game_state import clock_seconds, elapsed_seconds

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
    return elapsed_seconds(period, clock_seconds(pctimestring))


def parse_score_margin(score_margin: pd.Series, game_ids: pd.Series | None = None,
                       *, absolute: bool = True) -> pd.Series:
    """Parse post-event home margin; forward-fill only within each game.

    Pass absolute=False to retain direction. Callers analyzing performance should
    shift the signed result within each game to use the pre-event state.
    """
    values = pd.to_numeric(score_margin.astype("string").str.upper().replace("TIE", "0"),
                           errors="coerce")
    values = values.ffill() if game_ids is None else values.groupby(game_ids).ffill()
    values = values.fillna(0).astype(int)
    return values.abs() if absolute else values


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
    # NBA descriptions report cumulative "Off:N Def:N" on EVERY rebound.
    # Infer ownership from the last shot/free throw, never from that text.
    team = df.get("PLAYER1_TEAM_ID", df.get("PLAYER1_TEAM_ABBREVIATION"))
    if is_rebound.any() and team is None:
        raise ValueError("Rebound classification requires PLAYER1 team identity")
    if team is not None:
        shooting_team = team.where(df["is_fga"].eq(1) | df["is_fta"].eq(1))
        groups = df.get("GAME_ID", pd.Series(0, index=df.index))
        shooting_team = shooting_team.groupby(groups).ffill()
        is_off_reb = team.eq(shooting_team) & team.notna()
    else:
        is_off_reb = pd.Series(False, index=df.index)
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
    df = raw_pbp.sort_values(["GAME_ID", "EVENTNUM"], kind="stable").copy()

    # Parse time and score
    df["TIME_ELAPSED"] = parse_game_clock(df["PCTIMESTRING"], df["PERIOD"])
    signed = parse_score_margin(df["SCOREMARGIN"], df["GAME_ID"], absolute=False)
    df["HOME_SCORE_MARGIN_BEFORE"] = signed.groupby(df["GAME_ID"]).shift().fillna(0).astype(int)
    df["ABS_SCORE_DIFF"] = df["HOME_SCORE_MARGIN_BEFORE"].abs()

    # Parse clock into minutes/seconds remaining
    parts = df["PCTIMESTRING"].str.split(":", expand=True).astype(int)
    df["MINUTES_REMAINING"] = parts[0]
    df["SECONDS_REMAINING"] = parts[1]

    # Classify events
    df = classify_events(df)

    keep_cols = [
        "GAME_ID", "EVENTNUM", "EVENTMSGTYPE", "PERIOD",
        "TIME_ELAPSED", "MINUTES_REMAINING", "SECONDS_REMAINING",
        "ABS_SCORE_DIFF", "HOME_SCORE_MARGIN_BEFORE", "PLAYER1_TEAM_ID",
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
