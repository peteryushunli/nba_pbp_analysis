export interface SeasonInfo {
  season: number;
  display_name: string;
  game_count: number;
}

export interface EntityList {
  season: number;
  players: { id: number; name: string; team: string }[];
  teams: { name: string }[];
}

export interface HeatmapCell {
  time_bucket: string;
  score_bucket: string;
  efg_pct: number | null;
  efg_pct_padded: number | null;
  fgm: number;
  fga: number;
  fg3m: number;
  sample_size: number;
  is_below_threshold: boolean;
  confidence: "low" | "medium" | "high";
}

export interface EfgHeatmapResponse {
  season: number;
  entity_name: string | null;
  entity_type: "player" | "team" | "league";
  cells: HeatmapCell[];
  league_avg_efg: number;
}

export interface PlayerRating {
  player_id: number;
  player_name: string;
  team: string;
  ortg: number | null;
  drtg: number | null;
  ortg_padded: number | null;
  drtg_padded: number | null;
  possessions: number;
  games: number;
  is_below_threshold: boolean;
}

export interface RatingsResponse {
  season: number;
  time_bucket: string | null;
  score_bucket: string | null;
  players: PlayerRating[];
}

export interface WeightedRankEntry {
  player_name: string;
  weighted_efg: number;
  raw_efg: number;
  raw_diff: number;
  weighted_fga: number;
  weighted_fgm: number;
  total_fga: number;
  total_fgm: number;
}
