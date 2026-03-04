import { useQuery } from "@tanstack/react-query";
import { fetchRatings } from "../api/client";

export function useRatings(params: {
  season: number;
  player_name?: string;
  team_name?: string;
  time_bucket?: string;
  score_bucket?: string;
}) {
  return useQuery({
    queryKey: ["ratings", params],
    queryFn: () => fetchRatings(params),
    enabled: params.season > 0,
    staleTime: 5 * 60 * 1000,
  });
}
