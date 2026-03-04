import { useState } from "react";
import type { PlayerRating } from "../types";

interface Props {
  players: PlayerRating[];
  usePadded: boolean;
}

type SortKey = "player_name" | "ortg" | "drtg" | "possessions" | "games";

export function RatingsTable({ players, usePadded }: Props) {
  const [sortKey, setSortKey] = useState<SortKey>("ortg");
  const [sortDesc, setSortDesc] = useState(true);

  const handleSort = (key: SortKey) => {
    if (sortKey === key) {
      setSortDesc(!sortDesc);
    } else {
      setSortKey(key);
      setSortDesc(key !== "player_name");
    }
  };

  const sorted = [...players].sort((a, b) => {
    let av: number | string, bv: number | string;
    switch (sortKey) {
      case "player_name":
        av = a.player_name;
        bv = b.player_name;
        break;
      case "ortg":
        av = (usePadded ? a.ortg_padded : a.ortg) ?? -999;
        bv = (usePadded ? b.ortg_padded : b.ortg) ?? -999;
        break;
      case "drtg":
        av = (usePadded ? a.drtg_padded : a.drtg) ?? 999;
        bv = (usePadded ? b.drtg_padded : b.drtg) ?? 999;
        break;
      case "possessions":
        av = a.possessions;
        bv = b.possessions;
        break;
      case "games":
        av = a.games;
        bv = b.games;
        break;
    }
    if (av < bv) return sortDesc ? 1 : -1;
    if (av > bv) return sortDesc ? -1 : 1;
    return 0;
  });

  const th = (label: string, key: SortKey) => (
    <th
      className="px-3 py-2 text-left text-xs font-medium text-gray-500 uppercase cursor-pointer hover:text-gray-700"
      onClick={() => handleSort(key)}
    >
      {label} {sortKey === key ? (sortDesc ? "v" : "^") : ""}
    </th>
  );

  return (
    <div className="overflow-x-auto">
      <table className="min-w-full divide-y divide-gray-200">
        <thead className="bg-gray-50">
          <tr>
            {th("Player", "player_name")}
            <th className="px-3 py-2 text-left text-xs font-medium text-gray-500 uppercase">
              Team
            </th>
            {th("ORtg", "ortg")}
            {th("DRtg", "drtg")}
            {th("Poss", "possessions")}
            {th("Games", "games")}
          </tr>
        </thead>
        <tbody className="bg-white divide-y divide-gray-200">
          {sorted.slice(0, 100).map((p) => {
            const ortg = usePadded ? p.ortg_padded : p.ortg;
            const drtg = usePadded ? p.drtg_padded : p.drtg;
            return (
              <tr
                key={p.player_id}
                className={p.is_below_threshold ? "opacity-50" : ""}
              >
                <td className="px-3 py-2 text-sm">{p.player_name}</td>
                <td className="px-3 py-2 text-sm text-gray-500">{p.team}</td>
                <td className="px-3 py-2 text-sm font-mono">
                  {ortg !== null ? ortg.toFixed(1) : "-"}
                </td>
                <td className="px-3 py-2 text-sm font-mono">
                  {drtg !== null ? drtg.toFixed(1) : "-"}
                </td>
                <td className="px-3 py-2 text-sm font-mono">{p.possessions}</td>
                <td className="px-3 py-2 text-sm font-mono">{p.games}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
