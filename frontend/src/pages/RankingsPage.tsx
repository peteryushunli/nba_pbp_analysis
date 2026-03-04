import { useState } from "react";
import { SeasonSelector } from "../components/SeasonSelector";
import { useWeightedRanking } from "../hooks/useEfgHeatmap";

export function RankingsPage() {
  const [season, setSeason] = useState(2023);
  const [minFga, setMinFga] = useState(200);

  const { data, isLoading } = useWeightedRanking({ season, min_fga: minFga });

  return (
    <div>
      <h1 className="text-2xl font-bold mb-4">Weighted eFG% Rankings</h1>

      <div className="flex flex-wrap gap-4 items-center mb-6">
        <SeasonSelector value={season} onChange={setSeason} />
        <label className="text-sm">
          Min FGA:
          <input
            type="number"
            className="ml-1 border border-gray-300 rounded px-2 py-1 text-sm w-20"
            value={minFga}
            onChange={(e) => setMinFga(Number(e.target.value))}
            min={0}
          />
        </label>
      </div>

      <p className="text-sm text-gray-600 mb-4">
        Weighted eFG% applies clutch weights (higher weight for close games late
        in the 4th quarter) to each player's shooting performance across all
        time/score buckets.
      </p>

      {isLoading && <div className="text-gray-500">Loading...</div>}

      {data && data.length > 0 && (
        <div className="overflow-x-auto">
          <table className="min-w-full divide-y divide-gray-200">
            <thead className="bg-gray-50">
              <tr>
                <th className="px-3 py-2 text-left text-xs font-medium text-gray-500 uppercase">
                  #
                </th>
                <th className="px-3 py-2 text-left text-xs font-medium text-gray-500 uppercase">
                  Player
                </th>
                <th className="px-3 py-2 text-left text-xs font-medium text-gray-500 uppercase">
                  Weighted eFG%
                </th>
                <th className="px-3 py-2 text-left text-xs font-medium text-gray-500 uppercase">
                  Raw eFG%
                </th>
                <th className="px-3 py-2 text-left text-xs font-medium text-gray-500 uppercase">
                  Diff
                </th>
                <th className="px-3 py-2 text-left text-xs font-medium text-gray-500 uppercase">
                  FGA
                </th>
              </tr>
            </thead>
            <tbody className="bg-white divide-y divide-gray-200">
              {data.map((p, i) => (
                <tr key={p.player_name}>
                  <td className="px-3 py-2 text-sm text-gray-400">{i + 1}</td>
                  <td className="px-3 py-2 text-sm">{p.player_name}</td>
                  <td className="px-3 py-2 text-sm font-mono font-bold">
                    {(p.weighted_efg * 100).toFixed(1)}%
                  </td>
                  <td className="px-3 py-2 text-sm font-mono">
                    {(p.raw_efg * 100).toFixed(1)}%
                  </td>
                  <td
                    className={`px-3 py-2 text-sm font-mono ${
                      p.raw_diff > 0
                        ? "text-green-600"
                        : p.raw_diff < 0
                          ? "text-red-600"
                          : ""
                    }`}
                  >
                    {p.raw_diff > 0 ? "+" : ""}
                    {(p.raw_diff * 100).toFixed(1)}%
                  </td>
                  <td className="px-3 py-2 text-sm font-mono">{p.total_fga}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {data && data.length === 0 && (
        <div className="text-gray-500">No data for this season.</div>
      )}
    </div>
  );
}
