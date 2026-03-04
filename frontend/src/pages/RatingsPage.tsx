import { useState } from "react";
import { SeasonSelector } from "../components/SeasonSelector";
import { EntitySelector } from "../components/EntitySelector";
import { RatingsTable } from "../components/RatingsTable";
import { useRatings } from "../hooks/useRatings";

const TIME_BUCKETS = [
  "48-45", "44-41", "40-37", "36-33", "32-29", "28-25",
  "24-21", "20-17", "16-13", "12-9", "8-5", "4-0",
];
const SCORE_BUCKETS = ["0-5", "6-10", "11-15", "16-20", "21+"];

export function RatingsPage() {
  const [season, setSeason] = useState(2023);
  const [mode, setMode] = useState<"player" | "team">("team");
  const [selectedName, setSelectedName] = useState<string | undefined>();
  const [timeBucket, setTimeBucket] = useState<string | undefined>();
  const [scoreBucket, setScoreBucket] = useState<string | undefined>();
  const [usePadded, setUsePadded] = useState(false);

  const params = {
    season,
    ...(mode === "player" && selectedName ? { player_name: selectedName } : {}),
    ...(mode === "team" && selectedName ? { team_name: selectedName } : {}),
    ...(timeBucket ? { time_bucket: timeBucket } : {}),
    ...(scoreBucket ? { score_bucket: scoreBucket } : {}),
  };

  const { data, isLoading, error } = useRatings(params);

  return (
    <div>
      <h1 className="text-2xl font-bold mb-4">Offensive & Defensive Ratings</h1>

      <div className="flex flex-wrap gap-4 items-center mb-4">
        <SeasonSelector value={season} onChange={setSeason} />
        <EntitySelector
          season={season}
          mode={mode}
          onModeChange={setMode}
          onSelect={setSelectedName}
          selectedName={selectedName}
        />
      </div>

      <div className="flex flex-wrap gap-4 items-center mb-6">
        <label className="text-sm">
          Time:
          <select
            className="ml-1 border border-gray-300 rounded px-2 py-1 text-sm"
            value={timeBucket || ""}
            onChange={(e) => setTimeBucket(e.target.value || undefined)}
          >
            <option value="">All</option>
            {TIME_BUCKETS.map((tb) => (
              <option key={tb} value={tb}>
                {tb} min
              </option>
            ))}
          </select>
        </label>

        <label className="text-sm">
          Score diff:
          <select
            className="ml-1 border border-gray-300 rounded px-2 py-1 text-sm"
            value={scoreBucket || ""}
            onChange={(e) => setScoreBucket(e.target.value || undefined)}
          >
            <option value="">All</option>
            {SCORE_BUCKETS.map((sb) => (
              <option key={sb} value={sb}>
                {sb} pts
              </option>
            ))}
          </select>
        </label>

        <label className="flex items-center gap-2 text-sm">
          <input
            type="checkbox"
            checked={usePadded}
            onChange={(e) => setUsePadded(e.target.checked)}
          />
          Show padded values
        </label>
      </div>

      {isLoading && <div className="text-gray-500">Loading...</div>}
      {error && (
        <div className="text-red-500">Error: {(error as Error).message}</div>
      )}

      {data && data.players.length > 0 && (
        <RatingsTable players={data.players} usePadded={usePadded} />
      )}

      {data && data.players.length === 0 && (
        <div className="text-gray-500 mt-4">
          No ratings data available. Make sure you've fetched and processed PBP
          data using the pipeline CLI.
        </div>
      )}
    </div>
  );
}
