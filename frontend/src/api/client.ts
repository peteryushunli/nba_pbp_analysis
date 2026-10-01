import type { OnOffResponse, SeasonInfo, EntityList } from "../types";

const BASE = import.meta.env.DEV ? "http://127.0.0.1:8000/api" : "/api";

async function fetchJson<T>(url: string): Promise<T> {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`API error: ${res.status} ${res.statusText}`);
  return res.json();
}

export function fetchOnOffSeasons(): Promise<number[]> {
  return fetchJson(`${BASE}/on-off/seasons`);
}

export function fetchOnOff(params: {
  season: number;
  position?: string;
  min_gp?: number;
}): Promise<OnOffResponse> {
  const q = new URLSearchParams();
  q.set("season", String(params.season));
  if (params.position) q.set("position", params.position);
  if (params.min_gp !== undefined) q.set("min_gp", String(params.min_gp));
  return fetchJson(`${BASE}/on-off?${q}`);
}

export function fetchTeamSituations(): Promise<import("../pages/TeamSituationsPage").TeamSituationsData> {
  return fetchJson(`${BASE}/team-situations`);
}

export function fetchSeasons(): Promise<SeasonInfo[]> {
  return fetchJson(`${BASE}/seasons`);
}

export function fetchEntities(season: number): Promise<EntityList> {
  return fetchJson(`${BASE}/seasons/${season}/entities`);
}
