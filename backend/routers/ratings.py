"""ORtg/DRtg ratings endpoints."""

from fastapi import APIRouter, Query
from typing import Optional

from backend.db.connection import get_connection
from backend.db.queries import (
    SITUATIONAL_PLAYER_STATS,
    SITUATIONAL_ASSISTS,
    SITUATIONAL_STEALS,
    SITUATIONAL_BLOCKS,
    TEAM_STATS,
    EVENTS_LEAGUE_AVERAGES,
    season_filter_sql,
)
from backend.metrics.oliver_ratings import (
    PlayerBoxStats, TeamBoxStats, OpponentBoxStats, LeagueStats,
    calculate_ortg, calculate_drtg,
)
from backend.metrics.possessions import estimate_possessions
from backend.metrics.sample_size import apply_padding, is_below_threshold
from backend.models.responses import RatingsResponse, PlayerRating

router = APIRouter(tags=["ratings"])


@router.get("/ratings", response_model=RatingsResponse, deprecated=True)
def get_ratings(
    season: int = Query(..., ge=2003, le=2026),
    player_name: Optional[str] = Query(None),
    team_name: Optional[str] = Query(None),
    time_bucket: Optional[str] = Query(None),
    score_bucket: Optional[str] = Query(None),
    min_possessions: int = Query(20, ge=0),
):
    """
    Legacy approximate individual ratings, with inferred minutes/opponents.
    Use /team-situations for measured team ORtg/DRtg by signed game state.

    Requires processed events data (from `nba-pipeline fetch` + `process`).
    """
    conn = get_connection()

    # Check if events data exists
    try:
        conn.execute("SELECT 1 FROM events_bucketed LIMIT 0")
    except Exception:
        return RatingsResponse(
            season=season, time_bucket=time_bucket,
            score_bucket=score_bucket, players=[],
        )

    season_filter = season_filter_sql(season)
    time_filter = "AND TIME_BUCKET = ?" if time_bucket else ""
    score_filter = "AND SCORE_BUCKET = ?" if score_bucket else ""
    parameters = [value for value in [time_bucket, score_bucket] if value]
    filters = {
        "season_filter": season_filter,
        "time_filter": time_filter,
        "score_filter": score_filter,
    }

    # Query player stats
    player_df = conn.execute(SITUATIONAL_PLAYER_STATS.format(**filters), parameters).fetchdf()
    if player_df.empty:
        return RatingsResponse(
            season=season, time_bucket=time_bucket,
            score_bucket=score_bucket, players=[],
        )

    # Query secondary stats (assists, steals, blocks attributed to other players)
    assists_df = conn.execute(SITUATIONAL_ASSISTS.format(**filters), parameters).fetchdf()
    steals_df = conn.execute(SITUATIONAL_STEALS.format(**filters), parameters).fetchdf()
    blocks_df = conn.execute(SITUATIONAL_BLOCKS.format(**filters), parameters).fetchdf()

    # Merge secondary stats
    player_df = player_df.merge(assists_df, on="player_id", how="left")
    player_df = player_df.merge(steals_df, on="player_id", how="left")
    player_df = player_df.merge(blocks_df, on="player_id", how="left")
    player_df = player_df.fillna(0)

    # Team stats
    team_df = conn.execute(TEAM_STATS.format(**filters), parameters).fetchdf()

    # League averages (unfiltered for the season)
    lg_df = conn.execute(EVENTS_LEAGUE_AVERAGES.format(
        season_filter=season_filter
    )).fetchdf()

    if lg_df.empty or lg_df["lg_fga"].iloc[0] == 0:
        return RatingsResponse(
            season=season, time_bucket=time_bucket,
            score_bucket=score_bucket, players=[],
        )

    league = LeagueStats(
        fga=int(lg_df["lg_fga"].iloc[0]),
        fgm=int(lg_df["lg_fgm"].iloc[0]),
        fg3m=int(lg_df["lg_fg3m"].iloc[0]),
        fta=int(lg_df["lg_fta"].iloc[0]),
        ftm=int(lg_df["lg_ftm"].iloc[0]),
        oreb=int(lg_df["lg_oreb"].iloc[0]),
        dreb=int(lg_df["lg_dreb"].iloc[0]),
        reb=int(lg_df["lg_reb"].iloc[0]),
        pts=int(lg_df["lg_pts"].iloc[0]),
        tov=int(lg_df["lg_tov"].iloc[0]),
    )

    # League average ORtg/DRtg (approximate: ~110)
    lg_poss = estimate_possessions(league.fga, league.oreb, league.tov, league.fta)
    lg_ortg = 100 * league.pts / lg_poss if lg_poss > 0 else 110.0

    # Build team stats lookup
    team_lookup = {}
    for _, trow in team_df.iterrows():
        t = trow["team"]
        team_lookup[t] = trow

    # Calculate ratings for each player
    results = []
    for _, row in player_df.iterrows():
        team_abbr = row["team"]
        trow = team_lookup.get(team_abbr)
        if trow is None:
            continue

        # Estimate minutes proportionally
        team_events = int(trow.get("team_fga", 0)) + int(trow.get("team_fta", 0)) + int(trow.get("team_tov", 0))
        player_events = int(row["fga"]) + int(row["fta"]) + int(row["tov"])
        if team_events > 0:
            mp_fraction = player_events / team_events
        else:
            mp_fraction = 0
        # Scale to approximate minutes (48 min * games * 5 players)
        team_mp = 48.0 * int(row["games"]) * 5
        player_mp = mp_fraction * team_mp

        player_stats = PlayerBoxStats(
            player_id=int(row["player_id"]),
            player_name=row["player_name"],
            team=team_abbr,
            fgm=int(row["fgm"]),
            fga=int(row["fga"]),
            fg3m=int(row["fg3m"]),
            fg3a=int(row["fg3a"]),
            ftm=int(row["ftm"]),
            fta=int(row["fta"]),
            oreb=int(row["oreb"]),
            dreb=int(row["dreb"]),
            ast=int(row.get("ast", 0)),
            tov=int(row["tov"]),
            stl=int(row.get("stl", 0)),
            blk=int(row.get("blk", 0)),
            pf=int(row["pf"]),
            pts=int(row["pts"]),
            mp=player_mp,
            games=int(row["games"]),
        )

        # Build team stats object from team aggregate + team-level assists/steals/blocks
        team_ast_df = conn.execute(f"""
            SELECT COALESCE(SUM(is_ast), 0) AS team_ast
            FROM events_bucketed
            WHERE is_ast = 1
                AND PLAYER2_TEAM_ABBREVIATION = '{team_abbr}'
                {season_filter} {time_filter} {score_filter}
        """).fetchdf()
        team_stl_df = conn.execute(f"""
            SELECT COALESCE(SUM(is_stl), 0) AS team_stl
            FROM events_bucketed
            WHERE is_stl = 1
                AND PLAYER2_TEAM_ABBREVIATION = '{team_abbr}'
                {season_filter} {time_filter} {score_filter}
        """).fetchdf()
        team_blk_df = conn.execute(f"""
            SELECT COALESCE(SUM(is_blk), 0) AS team_blk
            FROM events_bucketed
            WHERE is_blk = 1
                {season_filter} {time_filter} {score_filter}
        """).fetchdf()

        team_box = TeamBoxStats(
            team=team_abbr,
            fgm=int(trow["team_fgm"]),
            fga=int(trow["team_fga"]),
            fg3m=int(trow["team_fg3m"]),
            ftm=int(trow["team_ftm"]),
            fta=int(trow["team_fta"]),
            oreb=int(trow["team_oreb"]),
            dreb=int(trow["team_dreb"]),
            ast=int(team_ast_df["team_ast"].iloc[0]),
            tov=int(trow["team_tov"]),
            stl=int(team_stl_df["team_stl"].iloc[0]),
            blk=int(team_blk_df["team_blk"].iloc[0]),
            pf=int(trow["team_pf"]),
            pts=int(trow["team_pts"]),
            mp=team_mp,
        )

        # Opponent stats: aggregate all other teams' stats
        opp_rows = [r for t, r in team_lookup.items() if t != team_abbr]
        if opp_rows:
            opponent = OpponentBoxStats(
                fgm=sum(int(r["team_fgm"]) for r in opp_rows),
                fga=sum(int(r["team_fga"]) for r in opp_rows),
                fg3m=sum(int(r["team_fg3m"]) for r in opp_rows),
                ftm=sum(int(r["team_ftm"]) for r in opp_rows),
                fta=sum(int(r["team_fta"]) for r in opp_rows),
                oreb=sum(int(r["team_oreb"]) for r in opp_rows),
                dreb=sum(int(r["team_dreb"]) for r in opp_rows),
                tov=sum(int(r["team_tov"]) for r in opp_rows),
                pts=sum(int(r["team_pts"]) for r in opp_rows),
            )
        else:
            continue

        ortg = calculate_ortg(player_stats, team_box, league)
        drtg = calculate_drtg(player_stats, team_box, opponent)

        poss = int(estimate_possessions(
            player_stats.fga, player_stats.oreb, player_stats.tov, player_stats.fta
        ))

        padded_ortg = apply_padding(ortg, poss, lg_ortg, stat_type="rating") if ortg else None
        padded_drtg = apply_padding(drtg, poss, lg_ortg, stat_type="rating") if drtg else None

        results.append(PlayerRating(
            player_id=player_stats.player_id,
            player_name=player_stats.player_name,
            team=team_abbr,
            ortg=ortg,
            drtg=drtg,
            ortg_padded=padded_ortg,
            drtg_padded=padded_drtg,
            possessions=poss,
            games=player_stats.games,
            is_below_threshold=is_below_threshold(poss, stat_type="rating"),
        ))

    # Filter by player/team if requested
    if player_name:
        results = [r for r in results if r.player_name == player_name]
    elif team_name:
        results = [r for r in results if r.team == team_name]

    # Sort by ORtg descending
    results.sort(key=lambda r: r.ortg or 0, reverse=True)

    return RatingsResponse(
        season=season,
        time_bucket=time_bucket,
        score_bucket=score_bucket,
        players=results,
    )
