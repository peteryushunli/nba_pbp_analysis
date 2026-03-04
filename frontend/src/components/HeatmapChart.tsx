import { ResponsiveHeatMap } from "@nivo/heatmap";
import type { HeatmapCell } from "../types";

interface Props {
  cells: HeatmapCell[];
  usePadded: boolean;
  leagueAvg: number;
}

const TIME_BUCKETS = [
  "48-45", "44-41", "40-37", "36-33", "32-29", "28-25",
  "24-21", "20-17", "16-13", "12-9", "8-5", "4-0",
];
const SCORE_BUCKETS = ["21+", "16-20", "11-15", "6-10", "0-5"];

export function HeatmapChart({ cells, usePadded, leagueAvg }: Props) {
  // Build lookup
  const lookup = new Map<string, HeatmapCell>();
  for (const c of cells) {
    lookup.set(`${c.score_bucket}|${c.time_bucket}`, c);
  }

  // Build Nivo data format: array of { id: scoreBucket, data: [{x: timeBucket, y: efg}] }
  const data = SCORE_BUCKETS.map((sb) => ({
    id: sb,
    data: TIME_BUCKETS.map((tb) => {
      const cell = lookup.get(`${sb}|${tb}`);
      const val = cell
        ? usePadded
          ? cell.efg_pct_padded
          : cell.efg_pct
        : null;
      const pct = val !== null && val !== undefined ? Math.round(val * 1000) / 10 : null;
      return {
        x: tb,
        y: pct,
      };
    }),
  }));

  return (
    <div style={{ height: 400 }}>
      <ResponsiveHeatMap
        data={data}
        margin={{ top: 40, right: 90, bottom: 60, left: 70 }}
        valueFormat={(v) => (v !== null ? `${v.toFixed(1)}%` : "")}
        axisTop={{
          tickSize: 5,
          tickPadding: 5,
          legend: "Minutes Remaining",
          legendPosition: "middle",
          legendOffset: -30,
        }}
        axisLeft={{
          tickSize: 5,
          tickPadding: 5,
          legend: "Score Diff",
          legendPosition: "middle",
          legendOffset: -55,
        }}
        colors={(cell) => {
          const v = cell.value ?? 50;
          const min = 25, max = 75;
          const t = Math.max(0, Math.min(1, (v - min) / (max - min)));
          // blue → white → red
          if (t < 0.5) {
            const s = t / 0.5;
            const r = Math.round(59 + s * (247 - 59));
            const g = Math.round(130 + s * (247 - 130));
            const b = Math.round(206 + s * (247 - 206));
            return `rgb(${r},${g},${b})`;
          } else {
            const s = (t - 0.5) / 0.5;
            const r = Math.round(247 + s * (178 - 247));
            const g = Math.round(247 - s * (247 - 24));
            const b = Math.round(247 - s * (247 - 43));
            return `rgb(${r},${g},${b})`;
          }
        }}
        emptyColor="#f0f0f0"
        legends={[
          {
            anchor: "right",
            translateX: 30,
            length: 300,
            thickness: 12,
            direction: "column",
            tickSize: 5,
            tickSpacing: 4,
            title: "eFG%",
            titleAlign: "start",
            titleOffset: 4,
          },
        ]}
        tooltip={({ cell: nivoCell }) => {
          const sb = nivoCell.serieId;
          const tb = nivoCell.data.x;
          const c = lookup.get(`${sb}|${tb}`);
          if (!c) return null;
          return (
            <div className="bg-white shadow-lg rounded p-2 text-xs border">
              <div className="font-bold">
                Time: {tb} | Score: {sb}
              </div>
              <div>
                Raw eFG: {c.efg_pct !== null ? (c.efg_pct * 100).toFixed(1) : "N/A"}%
              </div>
              <div>
                Padded eFG:{" "}
                {c.efg_pct_padded !== null
                  ? (c.efg_pct_padded * 100).toFixed(1)
                  : "N/A"}
                %
              </div>
              <div>
                {c.fgm}/{c.fga} FGA ({c.fg3m} 3PM)
              </div>
              <div>
                Confidence:{" "}
                <span
                  className={
                    c.confidence === "high"
                      ? "text-green-600"
                      : c.confidence === "medium"
                        ? "text-yellow-600"
                        : "text-red-600"
                  }
                >
                  {c.confidence}
                </span>
              </div>
            </div>
          );
        }}
      />
    </div>
  );
}
