import { useState } from "react";
import { SeasonSelector } from "../components/SeasonSelector";
import { EntitySelector } from "../components/EntitySelector";
import { HeatmapChart } from "../components/HeatmapChart";
import { useEfgHeatmap } from "../hooks/useEfgHeatmap";

export function EfgHeatmapPage() {
  const [season, setSeason] = useState(2023);
  const [mode, setMode] = useState<"player" | "team">("player");
  const [selectedName, setSelectedName] = useState<string | undefined>();
  const [usePadded, setUsePadded] = useState(false);

  const params = {
    season,
    ...(mode === "player" && selectedName ? { player_name: selectedName } : {}),
    ...(mode === "team" && selectedName ? { team_name: selectedName } : {}),
  };

  const { data, isLoading, error } = useEfgHeatmap(params);

  return (
    <div>
      <h1 className="text-2xl font-bold mb-4">eFG% Heatmap</h1>

      <div className="flex flex-wrap gap-4 items-center mb-6">
        <SeasonSelector value={season} onChange={setSeason} />
        <EntitySelector
          season={season}
          mode={mode}
          onModeChange={setMode}
          onSelect={setSelectedName}
          selectedName={selectedName}
        />
        <label className="flex items-center gap-2 text-sm">
          <input
            type="checkbox"
            checked={usePadded}
            onChange={(e) => setUsePadded(e.target.checked)}
          />
          Show padded values
        </label>
      </div>

      {data && (
        <div className="mb-2 text-sm text-gray-600">
          {data.entity_name
            ? `${data.entity_name} (${data.entity_type})`
            : "League Average"}{" "}
          | League eFG: {(data.league_avg_efg * 100).toFixed(1)}%
        </div>
      )}

      {isLoading && <div className="text-gray-500">Loading...</div>}
      {error && (
        <div className="text-red-500">Error: {(error as Error).message}</div>
      )}

      {data && data.cells.length > 0 && (
        <HeatmapChart
          cells={data.cells}
          usePadded={usePadded}
          leagueAvg={data.league_avg_efg}
        />
      )}

      {data && data.cells.length === 0 && (
        <div className="text-gray-500 mt-4">
          No shot data found for this selection.
        </div>
      )}

      <div className="mt-4 text-xs text-gray-400">
        Cells are colored by eFG% (blue=cold, red=hot). Hover for details.
        {usePadded &&
          " Padded values blend toward league average for small samples."}
      </div>
    </div>
  );
}
