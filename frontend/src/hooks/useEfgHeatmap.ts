import { useQuery } from "@tanstack/react-query";
import { fetchEfgHeatmap, fetchWeightedRanking } from "../api/client";

export function useEfgHeatmap(params: {
  season: number;
  player_name?: string;
  team_name?: string;
}) {
  return useQuery({
    queryKey: ["efg-heatmap", params],
    queryFn: () => fetchEfgHeatmap(params),
    enabled: params.season > 0,
    staleTime: 5 * 60 * 1000,
  });
}

export function useWeightedRanking(params: {
  season: number;
  min_fga?: number;
}) {
  return useQuery({
    queryKey: ["weighted-ranking", params],
    queryFn: () => fetchWeightedRanking(params),
    enabled: params.season > 0,
    staleTime: 5 * 60 * 1000,
  });
}
