import { useState } from "react";
import { useOnOff, useOnOffSeasons } from "../hooks/useOnOff";
import { OnOffTable } from "../components/OnOffTable";

const POSITIONS = ["All", "G", "F", "C"];

export function OnOffPage() {
  const { data: seasons, isLoading: seasonsLoading } = useOnOffSeasons();
  const [season, setSeason] = useState(0);
  const [position, setPosition] = useState("All");
  const [minGp, setMinGp] = useState(40);
  const [searchQuery, setSearchQuery] = useState("");

  // Auto-select latest season when seasons load
  const effectiveSeason =
    season > 0 ? season : seasons && seasons.length > 0 ? seasons[seasons.length - 1] : 0;

  const { data, isLoading, error } = useOnOff({
    season: effectiveSeason,
    position,
    min_gp: minGp,
  });

  return (
    <div className="space-y-4">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold text-gray-900">
          On/Off Floor Efficiency
        </h1>
        <p className="text-sm text-gray-500 mt-1">
          Team pts/100 possessions with player on vs. off the floor. Percentiles within position group.
        </p>
      </div>

      {/* Controls */}
      <div className="flex flex-wrap items-end gap-4 bg-white rounded-lg border border-gray-200 p-4">
        {/* Season */}
        <div>
          <label className="block text-xs font-medium text-gray-500 mb-1">
            Season
          </label>
          <select
            className="border border-gray-300 rounded px-3 py-1.5 text-sm bg-white"
            value={effectiveSeason}
            onChange={(e) => setSeason(Number(e.target.value))}
            disabled={seasonsLoading}
          >
            {seasonsLoading && <option>Loading...</option>}
            {seasons?.map((s) => (
              <option key={s} value={s}>
                {s - 1}-{String(s).slice(2)}
              </option>
            ))}
          </select>
        </div>

        {/* Position */}
        <div>
          <label className="block text-xs font-medium text-gray-500 mb-1">
            Position
          </label>
          <div className="flex rounded overflow-hidden border border-gray-300">
            {POSITIONS.map((pos) => (
              <button
                key={pos}
                className={`px-3 py-1.5 text-sm font-medium transition-colors ${
                  position === pos
                    ? "bg-gray-800 text-white"
                    : "bg-white text-gray-600 hover:bg-gray-100"
                } ${pos !== "All" ? "border-l border-gray-300" : ""}`}
                onClick={() => setPosition(pos)}
              >
                {pos}
              </button>
            ))}
          </div>
        </div>

        {/* Min GP */}
        <div>
          <label className="block text-xs font-medium text-gray-500 mb-1">
            Min GP
          </label>
          <input
            type="number"
            className="border border-gray-300 rounded px-3 py-1.5 text-sm w-20"
            value={minGp}
            min={0}
            max={82}
            onChange={(e) => setMinGp(Number(e.target.value) || 0)}
          />
        </div>

        {/* Search */}
        <div className="flex-1 min-w-48">
          <label className="block text-xs font-medium text-gray-500 mb-1">
            Search
          </label>
          <input
            type="text"
            className="border border-gray-300 rounded px-3 py-1.5 text-sm w-full"
            placeholder="Player or team..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
          />
        </div>

        {/* Count */}
        {data && (
          <div className="text-xs text-gray-400 self-end pb-1">
            {data.player_count} players
          </div>
        )}
      </div>

      {/* Table */}
      {isLoading && (
        <div className="text-center py-12 text-gray-400">Loading...</div>
      )}
      {error && (
        <div className="text-center py-12 text-red-500">
          Error loading data: {(error as Error).message}
        </div>
      )}
      {data && !isLoading && (
        <div className="bg-white rounded-lg border border-gray-200 overflow-hidden">
          <OnOffTable players={data.players} searchQuery={searchQuery} />
        </div>
      )}

      {!seasonsLoading && seasons?.length === 0 && (
        <div className="text-center py-12 text-gray-400">
          No on/off data available. Run the processing pipeline first.
        </div>
      )}
    </div>
  );
}
