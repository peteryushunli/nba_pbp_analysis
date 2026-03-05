"""DuckDB view definitions over Parquet files."""

import duckdb
import pandas as pd
from pathlib import Path

from backend.config import settings


def create_all_views(conn: duckdb.DuckDBPyConnection):
    """Create all views over Parquet data files."""
    data_dir = settings.DATA_DIR

    # --- Processed events (from transformer) ---
    processed_dir = data_dir / "processed"
    if processed_dir.exists() and list(processed_dir.glob("events_*.parquet")):
        conn.execute(f"""
            CREATE OR REPLACE VIEW events AS
            SELECT * FROM read_parquet('{processed_dir}/events_*.parquet',
                                       union_by_name=true)
        """)

        # Bucketed events: adds TIME_BUCKET and SCORE_BUCKET
        conn.execute("""
            CREATE OR REPLACE VIEW events_bucketed AS
            SELECT *,
                CASE
                    WHEN PERIOD <= 4 THEN MINUTES_REMAINING + (4 - PERIOD) * 12
                    ELSE MINUTES_REMAINING
                END AS RAW_MINUTES_REMAINING,
                CASE
                    WHEN ABS_SCORE_DIFF <= 5 THEN '0-5'
                    WHEN ABS_SCORE_DIFF <= 10 THEN '6-10'
                    WHEN ABS_SCORE_DIFF <= 15 THEN '11-15'
                    WHEN ABS_SCORE_DIFF <= 20 THEN '16-20'
                    ELSE '21+'
                END AS SCORE_BUCKET,
                CASE
                    WHEN PERIOD <= 4 THEN
                        CASE
                            WHEN MINUTES_REMAINING + (4 - PERIOD) * 12 <= 4 THEN '4-0'
                            WHEN MINUTES_REMAINING + (4 - PERIOD) * 12 <= 8 THEN '8-5'
                            WHEN MINUTES_REMAINING + (4 - PERIOD) * 12 <= 12 THEN '12-9'
                            WHEN MINUTES_REMAINING + (4 - PERIOD) * 12 <= 16 THEN '16-13'
                            WHEN MINUTES_REMAINING + (4 - PERIOD) * 12 <= 20 THEN '20-17'
                            WHEN MINUTES_REMAINING + (4 - PERIOD) * 12 <= 24 THEN '24-21'
                            WHEN MINUTES_REMAINING + (4 - PERIOD) * 12 <= 28 THEN '28-25'
                            WHEN MINUTES_REMAINING + (4 - PERIOD) * 12 <= 32 THEN '32-29'
                            WHEN MINUTES_REMAINING + (4 - PERIOD) * 12 <= 36 THEN '36-33'
                            WHEN MINUTES_REMAINING + (4 - PERIOD) * 12 <= 40 THEN '40-37'
                            WHEN MINUTES_REMAINING + (4 - PERIOD) * 12 <= 44 THEN '44-41'
                            ELSE '48-45'
                        END
                    ELSE '4-0'
                END AS TIME_BUCKET
            FROM events
        """)

    # --- Legacy shot detail (from converted CSVs) ---
    legacy_dir = data_dir / "legacy"
    if legacy_dir.exists() and list(legacy_dir.glob("*.parquet")):
        conn.execute(f"""
            CREATE OR REPLACE VIEW legacy_shots AS
            SELECT * FROM read_parquet('{legacy_dir}/shot_detail_pbp_*.parquet',
                                       union_by_name=true)
        """)

        # Bucketed legacy shots (matches existing notebook logic)
        conn.execute("""
            CREATE OR REPLACE VIEW legacy_shots_bucketed AS
            SELECT *,
                CASE
                    WHEN PERIOD > 4 THEN 4
                    ELSE PERIOD
                END AS PERIOD_CAPPED,
                CASE
                    WHEN ABS_SCORE_DIFF <= 5 THEN '0-5'
                    WHEN ABS_SCORE_DIFF <= 10 THEN '6-10'
                    WHEN ABS_SCORE_DIFF <= 15 THEN '11-15'
                    WHEN ABS_SCORE_DIFF <= 20 THEN '16-20'
                    ELSE '21+'
                END AS SCORE_BUCKET,
                CASE
                    WHEN MINUTES_REMAINING + (4 - LEAST(PERIOD, 4)) * 12 <= 4 THEN '4-0'
                    WHEN MINUTES_REMAINING + (4 - LEAST(PERIOD, 4)) * 12 <= 8 THEN '8-5'
                    WHEN MINUTES_REMAINING + (4 - LEAST(PERIOD, 4)) * 12 <= 12 THEN '12-9'
                    WHEN MINUTES_REMAINING + (4 - LEAST(PERIOD, 4)) * 12 <= 16 THEN '16-13'
                    WHEN MINUTES_REMAINING + (4 - LEAST(PERIOD, 4)) * 12 <= 20 THEN '20-17'
                    WHEN MINUTES_REMAINING + (4 - LEAST(PERIOD, 4)) * 12 <= 24 THEN '24-21'
                    WHEN MINUTES_REMAINING + (4 - LEAST(PERIOD, 4)) * 12 <= 28 THEN '28-25'
                    WHEN MINUTES_REMAINING + (4 - LEAST(PERIOD, 4)) * 12 <= 32 THEN '32-29'
                    WHEN MINUTES_REMAINING + (4 - LEAST(PERIOD, 4)) * 12 <= 36 THEN '36-33'
                    WHEN MINUTES_REMAINING + (4 - LEAST(PERIOD, 4)) * 12 <= 40 THEN '40-37'
                    WHEN MINUTES_REMAINING + (4 - LEAST(PERIOD, 4)) * 12 <= 44 THEN '44-41'
                    ELSE '48-45'
                END AS TIME_BUCKET
            FROM legacy_shots
        """)

    # --- Raw data views ---
    shots_dir = data_dir / "raw" / "shots"
    if shots_dir.exists() and list(shots_dir.glob("*.parquet")):
        conn.execute(f"""
            CREATE OR REPLACE VIEW shot_chart AS
            SELECT * FROM read_parquet('{shots_dir}/shot_chart_*.parquet',
                                       union_by_name=true)
        """)

    box_dir = data_dir / "raw" / "box_scores"
    if box_dir.exists() and list(box_dir.glob("*.parquet")):
        conn.execute(f"""
            CREATE OR REPLACE VIEW box_scores AS
            SELECT * FROM read_parquet('{box_dir}/box_scores_*.parquet',
                                       union_by_name=true)
        """)

    games_dir = data_dir / "raw" / "games"
    if games_dir.exists() and list(games_dir.glob("*.parquet")):
        conn.execute(f"""
            CREATE OR REPLACE VIEW games AS
            SELECT * FROM read_parquet('{games_dir}/game_index_*.parquet',
                                       union_by_name=true)
        """)

    # --- On/Off efficiency stats ---
    on_off_dir = data_dir / "processed" / "on_off"
    if on_off_dir.exists() and list(on_off_dir.glob("*.parquet")):
        conn.execute(f"""
            CREATE OR REPLACE VIEW on_off_stats AS
            SELECT * FROM read_parquet('{on_off_dir}/on_off_*.parquet',
                                       union_by_name=true)
        """)

    # --- eFG weight matrix ---
    _load_weight_matrix(conn)


def _load_weight_matrix(conn: duckdb.DuckDBPyConnection):
    """Load the clutch weight matrix from eFG_Weight.csv into a table."""
    weight_path = settings.WEIGHT_CSV
    if not weight_path.exists():
        return

    conn.execute("""
        CREATE OR REPLACE TABLE efg_weights (
            SCORE_BUCKET VARCHAR,
            TIME_BUCKET VARCHAR,
            WEIGHT DOUBLE
        )
    """)

    weights = pd.read_csv(weight_path)
    time_cols = [c for c in weights.columns if c != "Abs Score Difference"]

    rows = []
    for _, row in weights.iterrows():
        score_bucket = row["Abs Score Difference"]
        for tc in time_cols:
            rows.append({
                "SCORE_BUCKET": score_bucket,
                "TIME_BUCKET": tc,
                "WEIGHT": float(row[tc]),
            })

    weight_df = pd.DataFrame(rows)
    conn.execute("INSERT INTO efg_weights SELECT * FROM weight_df")
