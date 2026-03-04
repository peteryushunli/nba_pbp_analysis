"""Fetch NBA data from stats.nba.com via nba_api."""

import pandas as pd
from pathlib import Path
from tqdm import tqdm

from nba_api.stats.endpoints import (
    LeagueGameFinder,
    PlayByPlayV2,
    ShotChartDetail,
    BoxScoreTraditionalV2,
)
from backend.pipeline.rate_limiter import throttled_request
from backend.config import settings


def season_string(season: int) -> str:
    """Convert trailing year (e.g. 2025) to NBA format '2024-25'."""
    return f"{season - 1}-{str(season)[2:]}"


def game_id_prefix(season: int) -> str:
    """
    NBA game IDs start with '002XXYYYY' where XX encodes the season.
    Regular season prefix for 2024-25 is '0022400'.
    """
    season_code = str(season - 1)[2:]  # e.g. 2025 -> '24'
    return f"002{season_code}"


def fetch_game_ids(season: int, season_type: str = "Regular Season") -> pd.DataFrame:
    """
    Get all game IDs for a season.

    Returns DataFrame with GAME_ID, GAME_DATE, MATCHUP, TEAM_ID, TEAM_ABBREVIATION.
    Deduplicates to one row per game.
    """
    finder = throttled_request(
        LeagueGameFinder,
        season_nullable=season_string(season),
        league_id_nullable="00",
        season_type_nullable=season_type,
    )
    df = finder.get_data_frames()[0]
    cols = ["GAME_ID", "GAME_DATE", "MATCHUP", "TEAM_ID", "TEAM_ABBREVIATION"]
    return df[cols].drop_duplicates(subset=["GAME_ID"])


def fetch_pbp_for_game(game_id: str) -> pd.DataFrame:
    """Fetch full play-by-play for a single game."""
    pbp = throttled_request(PlayByPlayV2, game_id=game_id)
    return pbp.get_data_frames()[0]


def fetch_pbp_for_season(
    season: int,
    output_dir: Path | None = None,
    force: bool = False,
) -> Path:
    """
    Fetch PBP for all games in a season and save as Parquet.

    Supports resumption: skips games already saved to the output file
    unless force=True.

    Returns path to the output Parquet file.
    """
    output_dir = output_dir or settings.DATA_DIR / "raw" / "pbp"
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"play_by_play_{season}.parquet"

    if out_path.exists() and not force:
        print(f"PBP data already exists: {out_path} (use --force to re-fetch)")
        return out_path

    game_ids_df = fetch_game_ids(season)
    unique_games = game_ids_df["GAME_ID"].unique()

    all_pbp = []
    for game_id in tqdm(unique_games, desc=f"Fetching PBP {season}"):
        try:
            df = fetch_pbp_for_game(game_id)
            df["SEASON"] = season
            all_pbp.append(df)
        except Exception as e:
            print(f"  Failed to fetch PBP for {game_id}: {e}")
            continue

    if not all_pbp:
        raise RuntimeError(f"No PBP data fetched for season {season}")

    combined = pd.concat(all_pbp, ignore_index=True)
    combined.to_parquet(out_path, engine="pyarrow", compression="snappy", index=False)
    print(f"Saved {len(combined)} PBP events to {out_path}")
    return out_path


def fetch_shot_chart_for_season(
    season: int,
    output_dir: Path | None = None,
    force: bool = False,
) -> Path:
    """
    Fetch ShotChartDetail for all players in a season (single bulk call).

    Returns path to the output Parquet file.
    """
    output_dir = output_dir or settings.DATA_DIR / "raw" / "shots"
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"shot_chart_{season}.parquet"

    if out_path.exists() and not force:
        print(f"Shot chart already exists: {out_path}")
        return out_path

    shots = throttled_request(
        ShotChartDetail,
        player_id=0,
        team_id=0,
        context_measure_simple="FGA",
        season_nullable=season_string(season),
        season_type_all_star="Regular Season",
    )
    df = shots.get_data_frames()[0]
    df["SEASON"] = season
    df.to_parquet(out_path, engine="pyarrow", compression="snappy", index=False)
    print(f"Saved {len(df)} shots to {out_path}")
    return out_path


def fetch_box_scores_for_season(
    season: int,
    output_dir: Path | None = None,
    force: bool = False,
) -> Path:
    """
    Fetch BoxScoreTraditionalV2 for every game in the season.

    Returns path to the output Parquet file.
    """
    output_dir = output_dir or settings.DATA_DIR / "raw" / "box_scores"
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"box_scores_{season}.parquet"

    if out_path.exists() and not force:
        print(f"Box scores already exist: {out_path}")
        return out_path

    game_ids_df = fetch_game_ids(season)
    unique_games = game_ids_df["GAME_ID"].unique()

    all_box = []
    for game_id in tqdm(unique_games, desc=f"Fetching box scores {season}"):
        try:
            box = throttled_request(BoxScoreTraditionalV2, game_id=game_id)
            player_stats = box.get_data_frames()[0]
            player_stats["SEASON"] = season
            all_box.append(player_stats)
        except Exception as e:
            print(f"  Failed to fetch box score for {game_id}: {e}")
            continue

    if not all_box:
        raise RuntimeError(f"No box scores fetched for season {season}")

    combined = pd.concat(all_box, ignore_index=True)
    combined.to_parquet(out_path, engine="pyarrow", compression="snappy", index=False)
    print(f"Saved {len(combined)} box score rows to {out_path}")
    return out_path


def fetch_game_index_for_season(
    season: int,
    output_dir: Path | None = None,
    force: bool = False,
) -> Path:
    """
    Save the game index (list of games) for a season as Parquet.

    Returns path to the output Parquet file.
    """
    output_dir = output_dir or settings.DATA_DIR / "raw" / "games"
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"game_index_{season}.parquet"

    if out_path.exists() and not force:
        print(f"Game index already exists: {out_path}")
        return out_path

    df = fetch_game_ids(season)
    df["SEASON"] = season
    df.to_parquet(out_path, engine="pyarrow", compression="snappy", index=False)
    print(f"Saved {len(df)} game entries to {out_path}")
    return out_path
