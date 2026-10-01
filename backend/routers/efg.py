"""eFG% heatmap and weighted ranking endpoints."""

from fastapi import APIRouter, Query
from typing import Optional

from backend.db.connection import get_connection
from backend.db.queries import (
    LEGACY_EFG_HEATMAP,
    LEGACY_LEAGUE_AVERAGES,
    LEGACY_WEIGHTED_EFG,
    EVENTS_EFG_HEATMAP,
    EVENTS_LEAGUE_AVERAGES,
    season_filter_sql,
    season_game_id_range,
)
from backend.metrics.efg import compute_efg
from backend.metrics.sample_size import apply_padding, is_below_threshold, confidence_level
from backend.models.responses import EfgHeatmapResponse, HeatmapCell, WeightedRankEntry

router = APIRouter(tags=["eFG%"])


def _has_view(conn, view_name: str) -> bool:
    """Check if a view exists in DuckDB."""
    try:
        conn.execute(f"SELECT 1 FROM {view_name} LIMIT 0")
        return True
    except Exception:
        return False


@router.get("/efg/heatmap", response_model=EfgHeatmapResponse)
def get_efg_heatmap(
    season: int = Query(..., ge=2003, le=2026),
    player_name: Optional[str] = Query(None),
    team_name: Optional[str] = Query(None),
):
    """Get eFG% heatmap data (time buckets x score buckets)."""
    conn = get_connection()

    parameters = []
    if player_name:
        parameters.append(player_name)
    if team_name:
        parameters.append(team_name)

    # Decide which data source to use
    use_legacy = _has_view(conn, "legacy_shots_bucketed")
    use_events = _has_view(conn, "events_bucketed")

    if use_legacy:
        sf = season_filter_sql(season)
        player_filter = "AND PLAYER_NAME = ?" if player_name else ""
        team_filter = "AND TEAM_NAME = ?" if team_name else ""

        query = LEGACY_EFG_HEATMAP.format(
            season_filter=sf,
            player_filter=player_filter,
            team_filter=team_filter,
        )
        lg_query = LEGACY_LEAGUE_AVERAGES.format(season_filter=sf)
    elif use_events:
        sf = season_filter_sql(season)
        player_filter = "AND PLAYER1_NAME = ?" if player_name else ""
        team_filter = "AND PLAYER1_TEAM_ABBREVIATION = ?" if team_name else ""

        query = EVENTS_EFG_HEATMAP.format(
            season_filter=sf,
            player_filter=player_filter,
            team_filter=team_filter,
        )
        lg_query = EVENTS_LEAGUE_AVERAGES.format(season_filter=sf)
    else:
        return EfgHeatmapResponse(
            season=season, entity_name=None, entity_type="league",
            cells=[], league_avg_efg=0.0,
        )

    result = conn.execute(query, parameters).fetchdf()

    # League average eFG%
    lg_result = conn.execute(lg_query).fetchdf()
    if len(lg_result) > 0 and lg_result["lg_fga"].iloc[0] > 0:
        league_avg_efg = float(
            (lg_result["lg_fgm"].iloc[0] + 0.5 * lg_result["lg_fg3m"].iloc[0])
            / lg_result["lg_fga"].iloc[0]
        )
    else:
        league_avg_efg = 0.52  # fallback

    # Build response cells
    cells = []
    for _, row in result.iterrows():
        raw_efg = float(row["efg_pct"]) if row["efg_pct"] is not None else None
        n = int(row["sample_size"])
        padded = apply_padding(raw_efg, n, league_avg_efg, stat_type="efg")

        cells.append(HeatmapCell(
            time_bucket=row["TIME_BUCKET"],
            score_bucket=row["SCORE_BUCKET"],
            efg_pct=raw_efg,
            efg_pct_padded=padded,
            fgm=int(row["fgm"]),
            fga=int(row["fga"]),
            fg3m=int(row["fg3m"]),
            sample_size=n,
            is_below_threshold=is_below_threshold(n),
            confidence=confidence_level(n),
        ))

    entity_type = "player" if player_name else ("team" if team_name else "league")
    entity_name = player_name or team_name

    return EfgHeatmapResponse(
        season=season,
        entity_name=entity_name,
        entity_type=entity_type,
        cells=cells,
        league_avg_efg=round(league_avg_efg, 4),
    )


@router.get("/efg/weighted-ranking", response_model=list[WeightedRankEntry])
def get_weighted_ranking(
    season: int = Query(..., ge=2003, le=2026),
    min_fga: int = Query(200, ge=0, description="Minimum FGA to include"),
):
    """Get weighted eFG% ranking for all players in a season."""
    conn = get_connection()
    lo, hi = season_game_id_range(season)

    if not _has_view(conn, "legacy_shots_bucketed"):
        return []

    result = conn.execute(LEGACY_WEIGHTED_EFG, [lo, hi, min_fga]).fetchdf()

    entries = []
    for _, row in result.iterrows():
        entries.append(WeightedRankEntry(
            player_name=row["player_name"],
            weighted_efg=float(row["weighted_efg"]) if row["weighted_efg"] else 0.0,
            raw_efg=float(row["raw_efg"]) if row["raw_efg"] else 0.0,
            raw_diff=float(row["raw_diff"]) if row["raw_diff"] else 0.0,
            weighted_fga=float(row["weighted_fga"]),
            weighted_fgm=float(row["weighted_fgm"]),
            total_fga=int(row["total_fga"]),
            total_fgm=int(row["total_fgm"]),
        ))

    return entries
