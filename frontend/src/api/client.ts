import type {
  SeasonInfo,
  EntityList,
  EfgHeatmapResponse,
  RatingsResponse,
  WeightedRankEntry,
} from "../types";

const BASE = import.meta.env.DEV ? "http://127.0.0.1:8000/api" : "/api";

async function fetchJson<T>(url: string): Promise<T> {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`API error: ${res.status} ${res.statusText}`);
  return res.json();
}

export function fetchSeasons(): Promise<SeasonInfo[]> {
  return fetchJson(`${BASE}/seasons`);
}

export function fetchEntities(season: number): Promise<EntityList> {
  return fetchJson(`${BASE}/seasons/${season}/entities`);
}

export function fetchEfgHeatmap(params: {
  season: number;
  player_name?: string;
  team_name?: string;
}): Promise<EfgHeatmapResponse> {
  const q = new URLSearchParams();
  q.set("season", String(params.season));
  if (params.player_name) q.set("player_name", params.player_name);
  if (params.team_name) q.set("team_name", params.team_name);
  return fetchJson(`${BASE}/efg/heatmap?${q}`);
}

export function fetchRatings(params: {
  season: number;
  player_name?: string;
  team_name?: string;
  time_bucket?: string;
  score_bucket?: string;
}): Promise<RatingsResponse> {
  const q = new URLSearchParams();
  q.set("season", String(params.season));
  if (params.player_name) q.set("player_name", params.player_name);
  if (params.team_name) q.set("team_name", params.team_name);
  if (params.time_bucket) q.set("time_bucket", params.time_bucket);
  if (params.score_bucket) q.set("score_bucket", params.score_bucket);
  return fetchJson(`${BASE}/ratings?${q}`);
}

export function fetchWeightedRanking(params: {
  season: number;
  min_fga?: number;
}): Promise<WeightedRankEntry[]> {
  const q = new URLSearchParams();
  q.set("season", String(params.season));
  if (params.min_fga !== undefined) q.set("min_fga", String(params.min_fga));
  return fetchJson(`${BASE}/efg/weighted-ranking?${q}`);
}
