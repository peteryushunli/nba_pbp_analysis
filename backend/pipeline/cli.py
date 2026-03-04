"""CLI for the NBA data pipeline."""

from typing import Optional
import typer

app = typer.Typer(help="NBA PBP data pipeline")


@app.command()
def fetch(
    seasons: list[int] = typer.Option([2025], help="Season trailing years, e.g. 2025 for 2024-25"),
    data_types: list[str] = typer.Option(
        ["all"],
        help="Data types to fetch: pbp, shots, box_scores, games, or all",
    ),
    force: bool = typer.Option(False, help="Re-fetch even if files exist"),
):
    """Fetch raw data from NBA API and save as Parquet."""
    from backend.pipeline.fetcher import (
        fetch_pbp_for_season,
        fetch_shot_chart_for_season,
        fetch_box_scores_for_season,
        fetch_game_index_for_season,
    )

    types = set(data_types)
    fetch_all = "all" in types

    for season in seasons:
        typer.echo(f"\n{'='*50}")
        typer.echo(f"Season {season - 1}-{str(season)[2:]}")
        typer.echo(f"{'='*50}")

        if fetch_all or "games" in types:
            fetch_game_index_for_season(season, force=force)

        if fetch_all or "shots" in types:
            fetch_shot_chart_for_season(season, force=force)

        if fetch_all or "pbp" in types:
            fetch_pbp_for_season(season, force=force)

        if fetch_all or "box_scores" in types:
            fetch_box_scores_for_season(season, force=force)


@app.command()
def process(
    seasons: list[int] = typer.Option([2025], help="Season trailing years"),
    force: bool = typer.Option(False, help="Re-process even if output exists"),
):
    """Transform raw PBP Parquet into enriched event stream."""
    from backend.pipeline.transformer import process_season

    for season in seasons:
        process_season(season, force=force)


@app.command()
def migrate_legacy():
    """Convert existing processed_pbp/*.csv files to Parquet."""
    from backend.pipeline.converter import convert_legacy_csvs
    convert_legacy_csvs()


@app.command()
def full_refresh(
    seasons: list[int] = typer.Option([2025], help="Season trailing years"),
):
    """Fetch + process in one step."""
    from backend.pipeline.fetcher import (
        fetch_pbp_for_season,
        fetch_shot_chart_for_season,
        fetch_box_scores_for_season,
        fetch_game_index_for_season,
    )
    from backend.pipeline.transformer import process_season

    for season in seasons:
        typer.echo(f"\n{'='*50}")
        typer.echo(f"Full refresh: {season - 1}-{str(season)[2:]}")
        typer.echo(f"{'='*50}")

        fetch_game_index_for_season(season, force=True)
        fetch_shot_chart_for_season(season, force=True)
        fetch_pbp_for_season(season, force=True)
        fetch_box_scores_for_season(season, force=True)
        process_season(season, force=True)


if __name__ == "__main__":
    app()
