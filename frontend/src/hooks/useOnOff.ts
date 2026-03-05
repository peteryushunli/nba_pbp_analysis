import { useQuery } from "@tanstack/react-query";
import { fetchOnOff, fetchOnOffSeasons } from "../api/client";

export function useOnOffSeasons() {
  return useQuery({
    queryKey: ["on-off-seasons"],
    queryFn: fetchOnOffSeasons,
    staleTime: 10 * 60 * 1000,
  });
}

export function useOnOff(params: {
  season: number;
  position?: string;
  min_gp?: number;
}) {
  return useQuery({
    queryKey: ["on-off", params],
    queryFn: () => fetchOnOff(params),
    enabled: params.season > 0,
    staleTime: 5 * 60 * 1000,
  });
}
