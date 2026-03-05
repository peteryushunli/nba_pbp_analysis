"""Compute on/off floor efficiency stats from Kaggle PBP-with-lineups data.

Uses DuckDB for efficient processing of the ~13M row CSV.
The Kaggle dataset has a `Possession` column that tracks possession boundaries,
so we don't need to estimate possessions from time.

Algorithm:
1. For each game-possession, get the lineup (A1-A5, H1-H5) and score changes
2. Unpivot to per-player-per-possession rows
3. For each player-game: sum ON-court possessions and points
4. Derive OFF-court = game totals - ON-court
5. Aggregate per player-season and compute ratings (pts per 100 possessions)
"""

import duckdb
import pandas as pd
from pathlib import Path

from backend.config import settings


# Historical league pace (possessions per 48 min, per team) for sanity checks
LEAGUE_PACE = {
    2000: 93.1, 2001: 91.3, 2002: 90.7, 2003: 91.0, 2004: 90.1,
    2005: 90.9, 2006: 90.5, 2007: 92.4, 2008: 92.4, 2009: 91.7,
    2010: 92.7, 2011: 92.1, 2012: 91.3, 2013: 92.0, 2014: 93.9,
    2015: 93.9, 2016: 95.8, 2017: 96.4, 2018: 97.3, 2019: 100.0,
    2020: 100.3, 2021: 99.2, 2022: 98.2, 2023: 99.5,
}


def compute_season_from_date(date_str: str) -> int:
    """Convert date string (MM/DD/YYYY) to NBA season trailing year.

    NBA season runs Oct-Jun. Games in Aug+ belong to the following season.
    e.g., 11/24/2010 -> 2011 (2010-11 season)
          03/15/2011 -> 2011 (2010-11 season)
    """
    parts = date_str.split("/")
    month, year = int(parts[0]), int(parts[2])
    return year + 1 if month >= 8 else year


def process_kaggle_pbp(
    csv_path: Path | None = None,
    output_dir: Path | None = None,
    seasons: list[int] | None = None,
    force: bool = False,
) -> list[Path]:
    """Process the Kaggle PBP CSV into per-season on/off Parquet files.

    Args:
        csv_path: Path to all_games.csv
        output_dir: Where to save output Parquet files
        seasons: If provided, only process these seasons. Otherwise process all.
        force: Re-process even if output exists.

    Returns:
        List of output Parquet file paths.
    """
    csv_path = csv_path or settings.DATA_DIR / "external" / "kaggle_pbp" / "all_games.csv"
    output_dir = output_dir or settings.DATA_DIR / "processed" / "on_off"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not csv_path.exists():
        raise FileNotFoundError(f"Kaggle PBP CSV not found: {csv_path}")

    print(f"Loading PBP data from {csv_path}...")

    # Use a fresh in-memory DuckDB for processing
    conn = duckdb.connect(":memory:")

    # Load CSV into DuckDB (very fast, handles 13M rows efficiently)
    conn.execute(f"""
        CREATE TABLE pbp AS
        SELECT * FROM read_csv('{csv_path}',
            delim=',',
            header=true,
            quote='"',
            escape='"',
            sample_size=10000,
            ignore_errors=true,
            null_padding=true,
            columns={{
                'PlayNum': 'INTEGER',
                'GameID': 'VARCHAR',
                'Date': 'VARCHAR',
                'Period': 'INTEGER',
                'Possession': 'DOUBLE',
                'Time': 'VARCHAR',
                'AwayName': 'VARCHAR',
                'AwayScore': 'VARCHAR',
                'AwayEvent': 'VARCHAR',
                'HomeName': 'VARCHAR',
                'HomeScore': 'VARCHAR',
                'HomeEvent': 'VARCHAR',
                'AwayIn': 'VARCHAR',
                'AwayOut': 'VARCHAR',
                'HomeIn': 'VARCHAR',
                'HomeOut': 'VARCHAR',
                'ActivePlayers': 'VARCHAR',
                'A1': 'VARCHAR',
                'A2': 'VARCHAR',
                'A3': 'VARCHAR',
                'A4': 'VARCHAR',
                'A5': 'VARCHAR',
                'H1': 'VARCHAR',
                'H2': 'VARCHAR',
                'H3': 'VARCHAR',
                'H4': 'VARCHAR',
                'H5': 'VARCHAR'
            }}
        )
    """)

    total_rows = conn.execute("SELECT COUNT(*) FROM pbp").fetchone()[0]
    print(f"Loaded {total_rows:,} rows")

    # Add season column
    conn.execute("""
        ALTER TABLE pbp ADD COLUMN season INTEGER;
    """)
    conn.execute("""
        UPDATE pbp
        SET season = CASE
            WHEN CAST(SPLIT_PART("Date", '/', 1) AS INTEGER) >= 8
            THEN CAST(SPLIT_PART("Date", '/', 3) AS INTEGER) + 1
            ELSE CAST(SPLIT_PART("Date", '/', 3) AS INTEGER)
        END
    """)

    # Get available seasons
    available = [
        row[0] for row in
        conn.execute("SELECT DISTINCT season FROM pbp ORDER BY season").fetchall()
    ]
    print(f"Available seasons: {available}")

    if seasons:
        to_process = [s for s in seasons if s in available]
    else:
        to_process = available

    output_paths = []

    for season in to_process:
        out_path = output_dir / f"on_off_{season}.parquet"
        if out_path.exists() and not force:
            print(f"  Season {season}: already exists, skipping (use --force)")
            output_paths.append(out_path)
            continue

        print(f"\nProcessing season {season}...")
        df = _process_season(conn, season)

        if df is not None and len(df) > 0:
            df.to_parquet(str(out_path), engine="pyarrow", compression="snappy", index=False)
            print(f"  Saved {len(df)} players to {out_path}")
            output_paths.append(out_path)
        else:
            print(f"  No data for season {season}")

    conn.close()
    return output_paths


def _process_season(conn: duckdb.DuckDBPyConnection, season: int) -> pd.DataFrame | None:
    """Process a single season's on/off stats."""

    # Step 1: Summarize each possession — one row per (game, possession)
    # Score is cumulative, so use LAG to get points scored during each possession
    poss_df = conn.execute("""
        WITH poss_end AS (
            SELECT
                "GameID",
                "Possession",
                FIRST("AwayName" ORDER BY "PlayNum") AS away_team,
                FIRST("HomeName" ORDER BY "PlayNum") AS home_team,
                FIRST("A1" ORDER BY "PlayNum") AS A1,
                FIRST("A2" ORDER BY "PlayNum") AS A2,
                FIRST("A3" ORDER BY "PlayNum") AS A3,
                FIRST("A4" ORDER BY "PlayNum") AS A4,
                FIRST("A5" ORDER BY "PlayNum") AS A5,
                FIRST("H1" ORDER BY "PlayNum") AS H1,
                FIRST("H2" ORDER BY "PlayNum") AS H2,
                FIRST("H3" ORDER BY "PlayNum") AS H3,
                FIRST("H4" ORDER BY "PlayNum") AS H4,
                FIRST("H5" ORDER BY "PlayNum") AS H5,
                MAX(CAST("AwayScore" AS INTEGER)) AS end_away,
                MAX(CAST("HomeScore" AS INTEGER)) AS end_home,
            FROM pbp
            WHERE season = ?
              AND "Possession" IS NOT NULL
              AND "A1" IS NOT NULL
            GROUP BY "GameID", "Possession"
        )
        SELECT
            "GameID",
            "Possession",
            away_team,
            home_team,
            A1, A2, A3, A4, A5,
            H1, H2, H3, H4, H5,
            end_away - COALESCE(
                LAG(end_away) OVER (PARTITION BY "GameID" ORDER BY "Possession"),
                0
            ) AS away_pts,
            end_home - COALESCE(
                LAG(end_home) OVER (PARTITION BY "GameID" ORDER BY "Possession"),
                0
            ) AS home_pts,
        FROM poss_end
    """, [season]).fetchdf()

    if poss_df.empty:
        return None

    n_games = poss_df["GameID"].nunique()
    n_poss = len(poss_df)
    print(f"  {n_games} games, {n_poss:,} possessions")

    # Step 2: Unpivot — create per-player-per-possession rows
    # Away players: offense = away_pts, defense = home_pts
    # Home players: offense = home_pts, defense = away_pts
    player_rows = []
    for side, cols, off_col, def_col, team_col in [
        ("away", ["A1", "A2", "A3", "A4", "A5"], "away_pts", "home_pts", "away_team"),
        ("home", ["H1", "H2", "H3", "H4", "H5"], "home_pts", "away_pts", "home_team"),
    ]:
        for col in cols:
            subset = poss_df[["GameID", col, team_col, off_col, def_col]].copy()
            subset.columns = ["game_id", "player_id", "team", "off_pts", "def_pts"]
            player_rows.append(subset)

    player_poss = pd.concat(player_rows, ignore_index=True)
    player_poss = player_poss.dropna(subset=["player_id"])

    # Step 3: Per-player-per-game ON-court stats
    player_game_on = (
        player_poss
        .groupby(["player_id", "team", "game_id"])
        .agg(
            on_poss=("off_pts", "size"),  # count of possessions on court
            on_off_pts=("off_pts", "sum"),
            on_def_pts=("def_pts", "sum"),
        )
        .reset_index()
    )

    # Step 4: Game totals (one row per game, from possession-level data)
    game_totals = (
        poss_df
        .groupby("GameID")
        .agg(
            total_poss=("Possession", "size"),
            total_away_pts=("away_pts", "sum"),
            total_home_pts=("home_pts", "sum"),
            away_team=("away_team", "first"),
            home_team=("home_team", "first"),
        )
        .reset_index()
        .rename(columns={"GameID": "game_id"})
    )

    # Step 5: Join to compute OFF-court stats
    merged = player_game_on.merge(game_totals, on="game_id", how="left")

    # Off-court possessions = total game possessions - player's on-court possessions
    merged["off_poss"] = merged["total_poss"] - merged["on_poss"]

    # Off-court points: team total - what was scored when player was ON
    # Need to know if player is on away or home team
    is_away = merged["team"] == merged["away_team"]
    merged["off_off_pts"] = 0.0
    merged["off_def_pts"] = 0.0

    merged.loc[is_away, "off_off_pts"] = (
        merged.loc[is_away, "total_away_pts"] - merged.loc[is_away, "on_off_pts"]
    )
    merged.loc[is_away, "off_def_pts"] = (
        merged.loc[is_away, "total_home_pts"] - merged.loc[is_away, "on_def_pts"]
    )
    merged.loc[~is_away, "off_off_pts"] = (
        merged.loc[~is_away, "total_home_pts"] - merged.loc[~is_away, "on_off_pts"]
    )
    merged.loc[~is_away, "off_def_pts"] = (
        merged.loc[~is_away, "total_away_pts"] - merged.loc[~is_away, "on_def_pts"]
    )

    # Step 6: Aggregate per player per season (across all games)
    # Player might play for multiple teams — use mode (most common)
    player_season = (
        merged
        .groupby("player_id")
        .agg(
            team=("team", lambda x: x.value_counts().index[0]),  # most common team
            gp=("game_id", "nunique"),
            on_poss_total=("on_poss", "sum"),
            on_off_pts_total=("on_off_pts", "sum"),
            on_def_pts_total=("on_def_pts", "sum"),
            off_poss_total=("off_poss", "sum"),
            off_off_pts_total=("off_off_pts", "sum"),
            off_def_pts_total=("off_def_pts", "sum"),
        )
        .reset_index()
    )

    # Step 7: Compute ratings (pts per 100 possessions)
    # Each possession in the data represents one team's possession.
    # Total poss / 2 ≈ one team's possessions.
    # But since we track each team's pts separately, we use total poss directly.
    # Actually: a game has ~200 total possessions (100 per team).
    # When a player is on court for 100 possessions, that's ~50 offensive + 50 defensive.
    # Rating = pts / (poss/2) * 100

    MIN_POSS = 50  # Minimum possessions to compute rating

    player_season["on_team_poss"] = player_season["on_poss_total"] / 2.0
    player_season["off_team_poss"] = player_season["off_poss_total"] / 2.0

    # Compute ratings where we have enough data
    player_season["off_rating_on"] = (
        player_season["on_off_pts_total"] / player_season["on_team_poss"] * 100
    ).where(player_season["on_team_poss"] >= MIN_POSS / 2)

    player_season["def_rating_on"] = (
        player_season["on_def_pts_total"] / player_season["on_team_poss"] * 100
    ).where(player_season["on_team_poss"] >= MIN_POSS / 2)

    player_season["off_rating_off"] = (
        player_season["off_off_pts_total"] / player_season["off_team_poss"] * 100
    ).where(player_season["off_team_poss"] >= MIN_POSS / 2)

    player_season["def_rating_off"] = (
        player_season["off_def_pts_total"] / player_season["off_team_poss"] * 100
    ).where(player_season["off_team_poss"] >= MIN_POSS / 2)

    # Drop rows with NaN ratings (not enough data)
    player_season = player_season.dropna(subset=["off_rating_on", "off_rating_off",
                                                   "def_rating_on", "def_rating_off"])

    # Differentials
    player_season["off_diff"] = player_season["off_rating_on"] - player_season["off_rating_off"]
    player_season["def_diff"] = player_season["def_rating_on"] - player_season["def_rating_off"]

    # Approximate minutes from possessions (league average ~100 poss/48 min per team)
    player_season["minutes_on"] = player_season["on_team_poss"] / 100 * 48

    # Add season column and player_name (BR ID for now, will map later)
    player_season["season"] = season
    player_season["player_name"] = player_season["player_id"]  # placeholder
    player_season["position"] = "Unknown"  # will be filled by position mapping

    # Select output columns
    output = player_season[[
        "player_id", "player_name", "team", "position", "season",
        "gp", "minutes_on",
        "off_rating_on", "off_rating_off", "off_diff",
        "def_rating_on", "def_rating_off", "def_diff",
    ]].copy()

    # Round ratings
    for col in ["off_rating_on", "off_rating_off", "off_diff",
                "def_rating_on", "def_rating_off", "def_diff", "minutes_on"]:
        output[col] = output[col].round(1)

    print(f"  {len(output)} qualifying players (min {MIN_POSS} possessions)")

    return output
