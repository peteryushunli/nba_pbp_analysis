"""On/Off floor efficiency endpoints."""

from fastapi import APIRouter, Query

from backend.db.connection import get_connection
from backend.metrics.percentiles import compute_on_off_percentiles
from backend.models.responses import OnOffPlayerRow, OnOffResponse

router = APIRouter(tags=["on-off"])


@router.get("/on-off", response_model=OnOffResponse)
def get_on_off(
    season: int = Query(..., ge=2000, le=2030),
    position: str = Query("All", description="Position filter: All, PG, SG, SF, PF, C"),
    min_gp: int = Query(20, ge=0, description="Minimum games played"),
):
    """
    Get on/off floor efficiency stats for all qualifying players in a season.

    Returns raw pts/100poss values and percentile ranks within the selected
    position group.
    """
    conn = get_connection()

    # Query on_off data joined with positions
    try:
        df = conn.execute("""
            SELECT
                player_id,
                player_name,
                team,
                COALESCE(position, 'Unknown') AS position,
                gp,
                minutes_on,
                off_rating_on,
                off_rating_off,
                off_diff,
                def_rating_on,
                def_rating_off,
                def_diff
            FROM on_off_stats
            WHERE season = ?
              AND gp >= ?
            ORDER BY player_name
        """, [season, min_gp]).fetchdf()
    except Exception:
        return OnOffResponse(
            season=season,
            position_filter=position,
            min_gp=min_gp,
            player_count=0,
            players=[],
        )

    if df.empty:
        return OnOffResponse(
            season=season,
            position_filter=position,
            min_gp=min_gp,
            player_count=0,
            players=[],
        )

    # Compute percentiles within position group
    df = compute_on_off_percentiles(df, position=position)

    # Filter to requested position after percentile computation
    if position and position != "All":
        df = df[df["position"] == position]

    players = []
    for _, row in df.iterrows():
        players.append(OnOffPlayerRow(
            player_id=str(row["player_id"]),
            player_name=row["player_name"],
            team=row["team"],
            position=row["position"],
            gp=int(row["gp"]),
            minutes_on=round(float(row["minutes_on"]), 1),
            off_rating_on=round(float(row["off_rating_on"]), 1),
            off_diff=round(float(row["off_diff"]), 1),
            def_rating_on=round(float(row["def_rating_on"]), 1),
            def_diff=round(float(row["def_diff"]), 1),
            net_on=round(float(row["net_on"]), 1),
            net_diff=round(float(row["net_diff"]), 1),
            off_diff_pctl=round(float(row.get("off_diff_pctl", 50)), 1),
            def_diff_pctl=round(float(row.get("def_diff_pctl", 50)), 1),
            net_diff_pctl=round(float(row.get("net_diff_pctl", 50)), 1),
        ))

    return OnOffResponse(
        season=season,
        position_filter=position,
        min_gp=min_gp,
        player_count=len(players),
        players=players,
    )


@router.get("/on-off/seasons")
def list_on_off_seasons():
    """List seasons that have on/off data available."""
    conn = get_connection()
    try:
        df = conn.execute("""
            SELECT DISTINCT season
            FROM on_off_stats
            ORDER BY season
        """).fetchdf()
        return [int(s) for s in df["season"]]
    except Exception:
        return []
