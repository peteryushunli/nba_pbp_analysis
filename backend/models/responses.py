"""Pydantic response models for the API."""

from pydantic import BaseModel


class HeatmapCell(BaseModel):
    time_bucket: str
    score_bucket: str
    efg_pct: float | None
    efg_pct_padded: float | None
    fgm: int
    fga: int
    fg3m: int
    sample_size: int
    is_below_threshold: bool
    confidence: str  # "low", "medium", "high"


class EfgHeatmapResponse(BaseModel):
    season: int
    entity_name: str | None
    entity_type: str  # "player", "team", or "league"
    cells: list[HeatmapCell]
    league_avg_efg: float


class PlayerRating(BaseModel):
    player_id: int
    player_name: str
    team: str
    ortg: float | None
    drtg: float | None
    ortg_padded: float | None
    drtg_padded: float | None
    possessions: int
    games: int
    is_below_threshold: bool


class RatingsResponse(BaseModel):
    season: int
    time_bucket: str | None
    score_bucket: str | None
    players: list[PlayerRating]


class SeasonInfo(BaseModel):
    season: int
    display_name: str
    game_count: int


class EntityList(BaseModel):
    season: int
    players: list[dict]
    teams: list[dict]


class WeightedRankEntry(BaseModel):
    player_name: str
    weighted_efg: float
    raw_efg: float
    raw_diff: float
    weighted_fga: float
    weighted_fgm: float
    total_fga: int
    total_fgm: int


class OnOffPlayerRow(BaseModel):
    player_id: str
    player_name: str
    team: str
    position: str  # G, F, C
    gp: int
    minutes_on: float

    # Raw values (pts per 100 possessions)
    off_rating_on: float
    off_diff: float
    def_rating_on: float
    def_diff: float
    net_on: float
    net_diff: float

    # Percentiles on differentials (0-100, 100 = best)
    off_diff_pctl: float
    def_diff_pctl: float
    net_diff_pctl: float


class OnOffResponse(BaseModel):
    season: int
    position_filter: str
    min_gp: int
    player_count: int
    players: list[OnOffPlayerRow]
