"""Named SQL query templates for DuckDB."""


def season_game_id_range(season: int) -> tuple[int, int]:
    """
    Return (lo, hi) GAME_ID range for a season.

    Legacy 8-digit BIGINT IDs: 2XXYYYY.
    e.g. season 2004 (2003-04) -> 20300000..20399999
    e.g. season 2022 (2021-22) -> 22100000..22199999
    """
    season_code = str(season - 1)[2:]  # 2025 -> '24'
    lo = int(f"2{season_code}00000")
    hi = int(f"2{season_code}99999")
    return lo, hi


def season_filter_sql(season: int, game_id_col: str = "GAME_ID") -> str:
    """
    Build a SQL filter clause for a season that works with both
    BIGINT and VARCHAR GAME_ID columns.

    Uses a numeric range check which is fastest for BIGINT columns.
    """
    lo, hi = season_game_id_range(season)
    return f"AND TRY_CAST({game_id_col} AS BIGINT) BETWEEN {lo} AND {hi}"


# ============================================================
# eFG% Heatmap (works with both legacy_shots_bucketed and events_bucketed)
# ============================================================

LEGACY_EFG_HEATMAP = """
    SELECT
        TIME_BUCKET,
        SCORE_BUCKET,
        SUM(SHOT_MADE_FLAG) AS fgm,
        SUM(SHOT_ATTEMPTED_FLAG) AS fga,
        SUM(CASE WHEN "3PT_ATTEMPTED_FLAG" = 1 AND SHOT_MADE_FLAG = 1 THEN 1 ELSE 0 END) AS fg3m,
        SUM(SHOT_ATTEMPTED_FLAG) AS sample_size,
        ROUND(
            (SUM(SHOT_MADE_FLAG) + 0.5 * SUM(CASE WHEN "3PT_ATTEMPTED_FLAG" = 1 AND SHOT_MADE_FLAG = 1 THEN 1 ELSE 0 END))
            / NULLIF(SUM(SHOT_ATTEMPTED_FLAG), 0),
            4
        ) AS efg_pct
    FROM legacy_shots_bucketed
    WHERE 1=1
        {season_filter}
        {player_filter}
        {team_filter}
    GROUP BY TIME_BUCKET, SCORE_BUCKET
    ORDER BY SCORE_BUCKET, TIME_BUCKET
"""

EVENTS_EFG_HEATMAP = """
    SELECT
        TIME_BUCKET,
        SCORE_BUCKET,
        SUM(is_fgm) AS fgm,
        SUM(is_fga) AS fga,
        SUM(is_3pm) AS fg3m,
        SUM(is_fga) AS sample_size,
        ROUND(
            (SUM(is_fgm) + 0.5 * SUM(is_3pm)) / NULLIF(SUM(is_fga), 0),
            4
        ) AS efg_pct
    FROM events_bucketed
    WHERE 1=1
        {season_filter}
        {player_filter}
        {team_filter}
    GROUP BY TIME_BUCKET, SCORE_BUCKET
    ORDER BY SCORE_BUCKET, TIME_BUCKET
"""

# ============================================================
# League averages
# ============================================================

LEGACY_LEAGUE_AVERAGES = """
    SELECT
        SUM(SHOT_ATTEMPTED_FLAG) AS lg_fga,
        SUM(SHOT_MADE_FLAG) AS lg_fgm,
        SUM(CASE WHEN "3PT_ATTEMPTED_FLAG" = 1 AND SHOT_MADE_FLAG = 1 THEN 1 ELSE 0 END) AS lg_fg3m
    FROM legacy_shots_bucketed
    WHERE 1=1
        {season_filter}
"""

EVENTS_LEAGUE_AVERAGES = """
    SELECT
        SUM(is_fga) AS lg_fga,
        SUM(is_fgm) AS lg_fgm,
        SUM(is_3pm) AS lg_fg3m,
        SUM(is_fta) AS lg_fta,
        SUM(is_ftm) AS lg_ftm,
        SUM(is_oreb) AS lg_oreb,
        SUM(is_dreb) AS lg_dreb,
        SUM(is_oreb) + SUM(is_dreb) AS lg_reb,
        SUM(is_tov) AS lg_tov,
        SUM(is_fgm * 2 + is_3pm + is_ftm) AS lg_pts
    FROM events_bucketed
    WHERE 1=1
        {season_filter}
"""

# ============================================================
# Situational stats for Oliver's ratings
# ============================================================

SITUATIONAL_PLAYER_STATS = """
    SELECT
        PLAYER1_ID AS player_id,
        PLAYER1_NAME AS player_name,
        PLAYER1_TEAM_ABBREVIATION AS team,
        SUM(is_fga) AS fga,
        SUM(is_fgm) AS fgm,
        SUM(is_3pa) AS fg3a,
        SUM(is_3pm) AS fg3m,
        SUM(is_fta) AS fta,
        SUM(is_ftm) AS ftm,
        SUM(is_oreb) AS oreb,
        SUM(is_dreb) AS dreb,
        SUM(is_tov) AS tov,
        SUM(is_pf) AS pf,
        SUM(is_fgm * 2 + is_3pm + is_ftm) AS pts,
        COUNT(DISTINCT GAME_ID) AS games
    FROM events_bucketed
    WHERE PLAYER1_ID IS NOT NULL
        {season_filter}
        {time_filter}
        {score_filter}
    GROUP BY player_id, player_name, team
"""

SITUATIONAL_ASSISTS = """
    SELECT
        PLAYER2_ID AS player_id,
        SUM(is_ast) AS ast
    FROM events_bucketed
    WHERE is_ast = 1
        {season_filter}
        {time_filter}
        {score_filter}
    GROUP BY PLAYER2_ID
"""

SITUATIONAL_STEALS = """
    SELECT
        PLAYER2_ID AS player_id,
        SUM(is_stl) AS stl
    FROM events_bucketed
    WHERE is_stl = 1
        {season_filter}
        {time_filter}
        {score_filter}
    GROUP BY PLAYER2_ID
"""

SITUATIONAL_BLOCKS = """
    SELECT
        PLAYER3_ID AS player_id,
        SUM(is_blk) AS blk
    FROM events_bucketed
    WHERE is_blk = 1
        {season_filter}
        {time_filter}
        {score_filter}
    GROUP BY PLAYER3_ID
"""

TEAM_STATS = """
    SELECT
        PLAYER1_TEAM_ABBREVIATION AS team,
        SUM(is_fga) AS team_fga,
        SUM(is_fgm) AS team_fgm,
        SUM(is_3pa) AS team_fg3a,
        SUM(is_3pm) AS team_fg3m,
        SUM(is_fta) AS team_fta,
        SUM(is_ftm) AS team_ftm,
        SUM(is_oreb) AS team_oreb,
        SUM(is_dreb) AS team_dreb,
        SUM(is_tov) AS team_tov,
        SUM(is_pf) AS team_pf,
        SUM(is_fgm * 2 + is_3pm + is_ftm) AS team_pts
    FROM events_bucketed
    WHERE PLAYER1_TEAM_ABBREVIATION IS NOT NULL
        {season_filter}
        {time_filter}
        {score_filter}
    GROUP BY team
"""

# ============================================================
# Season/entity listing
# ============================================================

LEGACY_SEASONS = """
    SELECT
        CAST(SUBSTR(LPAD(CAST(TRY_CAST(GAME_ID AS BIGINT) AS VARCHAR), 10, '0'), 4, 2) AS INTEGER) + 2001 AS season,
        COUNT(DISTINCT GAME_ID) AS game_count
    FROM legacy_shots
    GROUP BY season
    ORDER BY season
"""

LEGACY_PLAYERS = """
    SELECT DISTINCT PLAYER_ID AS id, PLAYER_NAME AS name, TEAM_NAME AS team
    FROM legacy_shots
    WHERE GAME_ID BETWEEN $1 AND $2
    ORDER BY name
"""

LEGACY_TEAMS = """
    SELECT DISTINCT TEAM_NAME AS name
    FROM legacy_shots
    WHERE GAME_ID BETWEEN $1 AND $2
    ORDER BY name
"""

# ============================================================
# Weighted eFG% ranking
# ============================================================

LEGACY_WEIGHTED_EFG = """
    WITH player_cells AS (
        SELECT
            PLAYER_NAME,
            TIME_BUCKET,
            SCORE_BUCKET,
            SUM(SHOT_MADE_FLAG) AS fgm,
            SUM(SHOT_ATTEMPTED_FLAG) AS fga,
            SUM(CASE WHEN "3PT_ATTEMPTED_FLAG" = 1 AND SHOT_MADE_FLAG = 1 THEN 1 ELSE 0 END) AS fg3m
        FROM legacy_shots_bucketed
        WHERE GAME_ID BETWEEN $1 AND $2
        GROUP BY PLAYER_NAME, TIME_BUCKET, SCORE_BUCKET
    ),
    weighted AS (
        SELECT
            p.PLAYER_NAME,
            p.TIME_BUCKET,
            p.SCORE_BUCKET,
            p.fgm,
            p.fga,
            p.fg3m,
            w.WEIGHT,
            (p.fgm + 0.5 * p.fg3m) / NULLIF(p.fga, 0) AS cell_efg,
            p.fga * w.WEIGHT AS weighted_fga,
            (p.fgm + 0.5 * p.fg3m) * w.WEIGHT AS weighted_fgm
        FROM player_cells p
        JOIN efg_weights w ON p.TIME_BUCKET = w.TIME_BUCKET AND p.SCORE_BUCKET = w.SCORE_BUCKET
    )
    SELECT
        PLAYER_NAME AS player_name,
        ROUND(SUM(weighted_fgm) / NULLIF(SUM(weighted_fga), 0), 4) AS weighted_efg,
        ROUND(SUM(fgm + 0.5 * fg3m) / NULLIF(SUM(fga), 0), 4) AS raw_efg,
        ROUND(SUM(weighted_fgm) / NULLIF(SUM(weighted_fga), 0) - SUM(fgm + 0.5 * fg3m) / NULLIF(SUM(fga), 0), 4) AS raw_diff,
        ROUND(SUM(weighted_fga), 1) AS weighted_fga,
        ROUND(SUM(weighted_fgm), 1) AS weighted_fgm,
        SUM(fga) AS total_fga,
        SUM(fgm) AS total_fgm
    FROM weighted
    GROUP BY PLAYER_NAME
    HAVING SUM(fga) >= $3
    ORDER BY weighted_efg DESC
"""
