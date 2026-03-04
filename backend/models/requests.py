"""Pydantic request/query parameter models."""

from pydantic import BaseModel, Field


class EfgHeatmapParams(BaseModel):
    season: int = Field(..., ge=2003, le=2026)
    player_name: str | None = None
    team_name: str | None = None


class RatingsParams(BaseModel):
    season: int = Field(..., ge=2003, le=2026)
    player_name: str | None = None
    team_name: str | None = None
    time_bucket: str | None = None
    score_bucket: str | None = None
    min_possessions: int = Field(default=20, ge=0)
