import { useQuery } from "@tanstack/react-query";
import { fetchSeasons, fetchEntities } from "../api/client";

export function useSeasons() {
  return useQuery({
    queryKey: ["seasons"],
    queryFn: fetchSeasons,
    staleTime: 10 * 60 * 1000,
  });
}

export function useEntities(season: number) {
  return useQuery({
    queryKey: ["entities", season],
    queryFn: () => fetchEntities(season),
    enabled: season > 0,
    staleTime: 10 * 60 * 1000,
  });
}
