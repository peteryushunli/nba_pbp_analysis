"""Season and entity listing endpoints."""

from fastapi import APIRouter, Query

from backend.db.connection import get_connection
from backend.db.queries import (
    LEGACY_SEASONS,
    LEGACY_PLAYERS,
    LEGACY_TEAMS,
    season_game_id_range,
    season_filter_sql,
)
from backend.models.responses import SeasonInfo, EntityList

router = APIRouter(tags=["seasons"])


@router.get("/seasons", response_model=list[SeasonInfo])
def list_seasons():
    """List all available seasons with game counts."""
    conn = get_connection()

    seasons = []

    # Check legacy data
    try:
        result = conn.execute(LEGACY_SEASONS).fetchdf()
        for _, row in result.iterrows():
            s = int(row["season"])
            seasons.append(SeasonInfo(
                season=s,
                display_name=f"{s - 1}-{str(s)[2:]}",
                game_count=int(row["game_count"]),
            ))
    except Exception:
        pass

    # Check processed events data
    try:
        result = conn.execute("""
            SELECT
                CAST(SUBSTR(CAST(GAME_ID AS VARCHAR), 2, 2) AS INTEGER) + 2001 AS season,
                COUNT(DISTINCT GAME_ID) AS game_count
            FROM events
            WHERE is_fga = 1
            GROUP BY season
            ORDER BY season
        """).fetchdf()
        existing = {s.season for s in seasons}
        for _, row in result.iterrows():
            s = int(row["season"])
            if s not in existing:
                seasons.append(SeasonInfo(
                    season=s,
                    display_name=f"{s - 1}-{str(s)[2:]}",
                    game_count=int(row["game_count"]),
                ))
    except Exception:
        pass

    return sorted(seasons, key=lambda s: s.season)


@router.get("/seasons/{season}/entities", response_model=EntityList)
def list_entities(season: int):
    """List players and teams for a given season."""
    conn = get_connection()
    lo, hi = season_game_id_range(season)
    sf = season_filter_sql(season)

    players = []
    teams = []

    # Try legacy data first
    try:
        player_df = conn.execute(LEGACY_PLAYERS, [lo, hi]).fetchdf()
        players = [
            {"id": int(row["id"]), "name": row["name"], "team": row["team"]}
            for _, row in player_df.iterrows()
        ]
        team_df = conn.execute(LEGACY_TEAMS, [lo, hi]).fetchdf()
        teams = [{"name": row["name"]} for _, row in team_df.iterrows()]
    except Exception:
        pass

    # Try events data if legacy didn't yield results
    if not players:
        try:
            player_df = conn.execute(f"""
                SELECT DISTINCT
                    PLAYER1_ID AS id,
                    PLAYER1_NAME AS name,
                    PLAYER1_TEAM_ABBREVIATION AS team
                FROM events
                WHERE PLAYER1_ID IS NOT NULL AND is_fga = 1
                    {sf}
                ORDER BY name
            """).fetchdf()
            players = [
                {"id": int(row["id"]), "name": row["name"], "team": row["team"]}
                for _, row in player_df.iterrows()
            ]
            team_df = conn.execute(f"""
                SELECT DISTINCT PLAYER1_TEAM_ABBREVIATION AS name
                FROM events
                WHERE PLAYER1_TEAM_ABBREVIATION IS NOT NULL
                    {sf}
                ORDER BY name
            """).fetchdf()
            teams = [{"name": row["name"]} for _, row in team_df.iterrows()]
        except Exception:
            pass

    return EntityList(season=season, players=players, teams=teams)
