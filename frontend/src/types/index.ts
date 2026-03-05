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

export interface OnOffPlayerRow {
  player_id: string;
  player_name: string;
  team: string;
  position: string;
  gp: number;
  minutes_on: number;

  off_rating_on: number;
  off_diff: number;
  def_rating_on: number;
  def_diff: number;
  net_on: number;
  net_diff: number;

  off_diff_pctl: number;
  def_diff_pctl: number;
  net_diff_pctl: number;
}

export interface OnOffResponse {
  season: number;
  position_filter: string;
  min_gp: number;
  player_count: number;
  players: OnOffPlayerRow[];
}
