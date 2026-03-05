import { useState, useMemo, Fragment } from "react";
import type { OnOffPlayerRow } from "../types";

type SortKey = keyof OnOffPlayerRow;

interface Props {
  players: OnOffPlayerRow[];
  searchQuery: string;
}

/**
 * Percentile → background color.
 * Blue (bad, 0) → White (neutral, 50) → Orange (good, 100)
 */
function percentileColor(pctl: number): string {
  const t = Math.max(0, Math.min(1, pctl / 100));
  if (t < 0.5) {
    // Blue → White
    const s = t / 0.5;
    const r = Math.round(59 + s * (255 - 59));
    const g = Math.round(130 + s * (255 - 130));
    const b = Math.round(206 + s * (255 - 206));
    return `rgb(${r},${g},${b})`;
  } else {
    // White → Orange
    const s = (t - 0.5) / 0.5;
    const r = Math.round(255);
    const g = Math.round(255 - s * (255 - 140));
    const b = Math.round(255 - s * (255 - 26));
    return `rgb(${r},${g},${b})`;
  }
}

function textColorForBg(pctl: number): string {
  return pctl < 12 || pctl > 88 ? "white" : "#1a1a1a";
}

// Each stat group has a rating column (plain) and a +/- column (percentile-colored)
const STAT_GROUPS: {
  group: string;
  ratingKey: SortKey;
  diffKey: SortKey;
  pctlKey: SortKey;
}[] = [
  {
    group: "Offense",
    ratingKey: "off_rating_on",
    diffKey: "off_diff",
    pctlKey: "off_diff_pctl",
  },
  {
    group: "Defense",
    ratingKey: "def_rating_on",
    diffKey: "def_diff",
    pctlKey: "def_diff_pctl",
  },
  {
    group: "Net",
    ratingKey: "net_on",
    diffKey: "net_diff",
    pctlKey: "net_diff_pctl",
  },
];

export function OnOffTable({ players, searchQuery }: Props) {
  const [sortKey, setSortKey] = useState<SortKey>("net_diff_pctl");
  const [sortDesc, setSortDesc] = useState(true);

  const handleSort = (key: SortKey) => {
    if (sortKey === key) {
      setSortDesc(!sortDesc);
    } else {
      setSortKey(key);
      setSortDesc(true);
    }
  };

  const sortIndicator = (key: SortKey) => {
    if (sortKey !== key) return "";
    return sortDesc ? " \u25BC" : " \u25B2";
  };

  const filtered = useMemo(() => {
    if (!searchQuery) return players;
    const normalize = (s: string) =>
      s.normalize("NFD").replace(/[\u0300-\u036f]/g, "").toLowerCase();
    const q = normalize(searchQuery);
    return players.filter(
      (p) =>
        normalize(p.player_name).includes(q) ||
        p.team.toLowerCase().includes(q),
    );
  }, [players, searchQuery]);

  const sorted = useMemo(() => {
    return [...filtered].sort((a, b) => {
      const aVal = a[sortKey];
      const bVal = b[sortKey];
      if (typeof aVal === "string" && typeof bVal === "string") {
        return sortDesc ? bVal.localeCompare(aVal) : aVal.localeCompare(bVal);
      }
      const diff = (aVal as number) - (bVal as number);
      return sortDesc ? -diff : diff;
    });
  }, [filtered, sortKey, sortDesc]);

  return (
    <div className="overflow-x-auto">
      <table className="w-full border-collapse text-sm">
        <thead>
          {/* Group header row */}
          <tr className="bg-gray-800 text-white">
            <th
              colSpan={4}
              className="px-2 py-1.5 text-left font-medium border-r border-gray-600"
            >
              Player Info
            </th>
            {STAT_GROUPS.map((g, i) => (
              <th
                key={g.group}
                colSpan={3}
                className={`px-2 py-1.5 text-center font-medium ${
                  i < STAT_GROUPS.length - 1 ? "border-r border-gray-600" : ""
                }`}
              >
                {g.group}
              </th>
            ))}
          </tr>

          {/* Sub-header row */}
          <tr className="bg-gray-700 text-gray-200 text-xs">
            <th
              className="px-2 py-1.5 text-left cursor-pointer hover:bg-gray-600 w-44"
              onClick={() => handleSort("player_name")}
            >
              Player{sortIndicator("player_name")}
            </th>
            <th
              className="px-2 py-1.5 text-left cursor-pointer hover:bg-gray-600 w-14"
              onClick={() => handleSort("team")}
            >
              Team{sortIndicator("team")}
            </th>
            <th
              className="px-2 py-1.5 text-center cursor-pointer hover:bg-gray-600 w-10"
              onClick={() => handleSort("position")}
            >
              Pos{sortIndicator("position")}
            </th>
            <th
              className="px-2 py-1.5 text-center cursor-pointer hover:bg-gray-600 w-10 border-r border-gray-600"
              onClick={() => handleSort("gp")}
            >
              GP{sortIndicator("gp")}
            </th>

            {STAT_GROUPS.map((g, gi) => (
              <Fragment key={g.group}>
                <th
                  className="px-2 py-1.5 text-center cursor-pointer hover:bg-gray-600 w-16"
                  onClick={() => handleSort(g.ratingKey)}
                >
                  Rtg{sortIndicator(g.ratingKey)}
                </th>
                <th
                  className="px-2 py-1.5 text-center cursor-pointer hover:bg-gray-600 w-16"
                  onClick={() => handleSort(g.diffKey)}
                >
                  +/-{sortIndicator(g.diffKey)}
                </th>
                <th
                  className={`px-2 py-1.5 text-center cursor-pointer hover:bg-gray-600 w-16 ${
                    gi < STAT_GROUPS.length - 1
                      ? "border-r border-gray-600"
                      : ""
                  }`}
                  onClick={() => handleSort(g.pctlKey)}
                >
                  Pctl{sortIndicator(g.pctlKey)}
                </th>
              </Fragment>
            ))}
          </tr>
        </thead>

        <tbody>
          {sorted.map((row, idx) => (
            <tr
              key={row.player_id}
              className={`${idx % 2 === 0 ? "bg-white" : "bg-gray-50"} hover:bg-blue-50`}
            >
              <td className="px-2 py-1.5 font-medium text-gray-900 whitespace-nowrap">
                {row.player_name}
              </td>
              <td className="px-2 py-1.5 text-gray-600">{row.team}</td>
              <td className="px-2 py-1.5 text-center text-gray-600">
                {row.position}
              </td>
              <td className="px-2 py-1.5 text-center text-gray-600 border-r border-gray-200">
                {row.gp}
              </td>

              {STAT_GROUPS.map((g, gi) => {
                const ratingVal = row[g.ratingKey] as number;
                const diffVal = row[g.diffKey] as number;
                const pctlVal = row[g.pctlKey] as number;
                return (
                  <Fragment key={g.group}>
                    <td className="px-2 py-1.5 text-center font-mono text-sm text-gray-900">
                      {ratingVal.toFixed(1)}
                    </td>
                    <td className="px-2 py-1.5 text-center font-mono text-sm text-gray-900">
                      {diffVal > 0 ? "+" : ""}
                      {diffVal.toFixed(1)}
                    </td>
                    <td
                      className={`px-2 py-1.5 text-center font-mono text-sm ${
                        gi < STAT_GROUPS.length - 1
                          ? "border-r border-gray-200"
                          : ""
                      }`}
                      style={{
                        backgroundColor: percentileColor(pctlVal),
                        color: textColorForBg(pctlVal),
                      }}
                    >
                      {Math.round(pctlVal)}
                    </td>
                  </Fragment>
                );
              })}
            </tr>
          ))}

          {sorted.length === 0 && (
            <tr>
              <td
                colSpan={13}
                className="px-4 py-8 text-center text-gray-400"
              >
                No players found
              </td>
            </tr>
          )}
        </tbody>
      </table>
    </div>
  );
}
