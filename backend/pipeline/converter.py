"""Convert legacy processed_pbp/*.csv files to Parquet format."""

import pandas as pd
from pathlib import Path
from tqdm import tqdm

from backend.config import settings
from backend.metrics.game_state import elapsed_seconds
from backend.pipeline.transformer import parse_score_margin


def convert_legacy_csvs(
    source_dir: Path | None = None,
    dest_dir: Path | None = None,
) -> list[Path]:
    """
    Convert all existing shot_detail_pbp_*.csv to Parquet.

    Preserves the original schema: GAME_ID, GAME_DATE, PLAYER_ID, PLAYER_NAME,
    TEAM_NAME, PERIOD, MINUTES_REMAINING, SECONDS_REMAINING, TIME_ELAPSED,
    ABS_SCORE_DIFF, SHOT_ATTEMPTED_FLAG, SHOT_MADE_FLAG, 3PT_ATTEMPTED_FLAG

    Returns list of created Parquet file paths.
    """
    source_dir = source_dir or settings.LEGACY_DIR
    dest_dir = dest_dir or settings.DATA_DIR / "legacy"
    dest_dir.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(source_dir.glob("shot_detail_pbp_*.csv"))
    if not csv_files:
        print(f"No CSV files found in {source_dir}")
        return []

    created = []
    for csv_path in tqdm(csv_files, desc="Converting CSVs to Parquet"):
        season = csv_path.stem.split("_")[-1]
        out_path = dest_dir / f"shot_detail_pbp_{season}.parquet"

        if out_path.exists():
            created.append(out_path)
            continue

        df = pd.read_csv(csv_path)
        df.to_parquet(out_path, engine="pyarrow", compression="snappy", index=False)
        created.append(out_path)

    print(f"Converted {len(created)} files to {dest_dir}")
    return created


def convert_shot_chart_to_legacy(
    season: int,
    source_dir: Path | None = None,
    dest_dir: Path | None = None,
    pbp_dir: Path | None = None,
) -> Path | None:
    """
    Convert raw ShotChartDetail Parquet into legacy format.

    Derives 3PT_ATTEMPTED_FLAG from SHOT_TYPE.
    If PBP data is available, merges ABS_SCORE_DIFF from it;
    otherwise ABS_SCORE_DIFF is set to NaN.

    Returns path to created file, or None if source doesn't exist.
    """
    source_dir = source_dir or settings.DATA_DIR / "raw" / "shots"
    dest_dir = dest_dir or settings.DATA_DIR / "legacy"
    pbp_dir = pbp_dir or settings.DATA_DIR / "raw" / "pbp"
    dest_dir.mkdir(parents=True, exist_ok=True)

    src_path = source_dir / f"shot_chart_{season}.parquet"
    if not src_path.exists():
        print(f"No shot chart data for season {season}")
        return None

    df = pd.read_parquet(src_path)

    # Derive 3PT_ATTEMPTED_FLAG
    df["3PT_ATTEMPTED_FLAG"] = (df["SHOT_TYPE"] == "3PT Field Goal").astype(int)

    # Compute TIME_ELAPSED (seconds from start of game)
    df["TIME_ELAPSED"] = elapsed_seconds(
        df["PERIOD"], df["MINUTES_REMAINING"] * 60 + df["SECONDS_REMAINING"])

    # Convert GAME_ID from string '002XXYYYY' to int (strip leading '00')
    df["GAME_ID"] = df["GAME_ID"].astype(str).str.lstrip("0").astype(int)

    # Try to merge ABS_SCORE_DIFF from PBP data
    pbp_path = pbp_dir / f"play_by_play_{season}.parquet"
    if pbp_path.exists():
        pbp = pd.read_parquet(pbp_path, columns=["GAME_ID", "EVENTNUM", "SCOREMARGIN"])
        pbp["GAME_ID"] = pbp["GAME_ID"].astype(str).str.lstrip("0").astype(int)
        pbp = pbp.sort_values(["GAME_ID", "EVENTNUM"])
        margin = parse_score_margin(pbp["SCOREMARGIN"], pbp["GAME_ID"], absolute=False)
        pbp["ABS_SCORE_DIFF"] = margin.groupby(pbp["GAME_ID"]).shift().fillna(0).abs()
        pbp = pbp[["GAME_ID", "EVENTNUM", "ABS_SCORE_DIFF"]].dropna(subset=["ABS_SCORE_DIFF"])

        df = df.merge(
            pbp,
            left_on=["GAME_ID", "GAME_EVENT_ID"],
            right_on=["GAME_ID", "EVENTNUM"],
            how="left",
        )
        df.drop(columns=["EVENTNUM"], errors="ignore", inplace=True)
        print(f"  Merged ABS_SCORE_DIFF from PBP ({df['ABS_SCORE_DIFF'].notna().sum()}/{len(df)} matched)")
    else:
        df["ABS_SCORE_DIFF"] = float("nan")
        print(f"  No PBP data yet — ABS_SCORE_DIFF set to NaN (run fetch --data-types pbp to fill)")

    # Select and rename to legacy columns
    legacy_cols = [
        "GAME_ID", "GAME_DATE", "PLAYER_ID", "PLAYER_NAME", "TEAM_NAME",
        "PERIOD", "MINUTES_REMAINING", "SECONDS_REMAINING", "TIME_ELAPSED",
        "ABS_SCORE_DIFF", "SHOT_ATTEMPTED_FLAG", "SHOT_MADE_FLAG", "3PT_ATTEMPTED_FLAG",
    ]
    out = df[legacy_cols].copy()

    out_path = dest_dir / f"shot_detail_pbp_{season}.parquet"
    out.to_parquet(out_path, engine="pyarrow", compression="snappy", index=False)
    print(f"Saved {len(out)} shots to {out_path}")
    return out_path
