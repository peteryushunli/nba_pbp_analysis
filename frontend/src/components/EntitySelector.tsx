import { useState, useMemo } from "react";
import { useEntities } from "../hooks/useSeasons";

interface Props {
  season: number;
  mode: "player" | "team";
  onModeChange: (mode: "player" | "team") => void;
  onSelect: (name: string | undefined) => void;
  selectedName?: string;
}

export function EntitySelector({
  season,
  mode,
  onModeChange,
  onSelect,
  selectedName,
}: Props) {
  const { data } = useEntities(season);
  const [search, setSearch] = useState("");

  const items = useMemo(() => {
    if (!data) return [];
    const list =
      mode === "player"
        ? data.players.map((p) => `${p.name} (${p.team})`)
        : data.teams.map((t) => t.name);
    if (!search) return list;
    const lower = search.toLowerCase();
    return list.filter((item) => item.toLowerCase().includes(lower));
  }, [data, mode, search]);

  const handleSelect = (value: string) => {
    if (value === "") {
      onSelect(undefined);
      setSearch("");
    } else {
      // Strip team abbreviation for player names
      const name = mode === "player" ? value.replace(/\s*\(.*\)$/, "") : value;
      onSelect(name);
      setSearch("");
    }
  };

  return (
    <div className="flex gap-2 items-center">
      <div className="flex rounded overflow-hidden border border-gray-300">
        <button
          className={`px-3 py-1.5 text-sm ${
            mode === "player" ? "bg-blue-600 text-white" : "bg-white text-gray-700"
          }`}
          onClick={() => {
            onModeChange("player");
            onSelect(undefined);
          }}
        >
          Player
        </button>
        <button
          className={`px-3 py-1.5 text-sm ${
            mode === "team" ? "bg-blue-600 text-white" : "bg-white text-gray-700"
          }`}
          onClick={() => {
            onModeChange("team");
            onSelect(undefined);
          }}
        >
          Team
        </button>
      </div>

      <div className="relative">
        <input
          type="text"
          className="border border-gray-300 rounded px-3 py-1.5 text-sm w-64"
          placeholder={`Search ${mode}s...`}
          value={selectedName || search}
          onChange={(e) => {
            setSearch(e.target.value);
            if (selectedName) onSelect(undefined);
          }}
        />
        {search && !selectedName && items.length > 0 && (
          <ul className="absolute z-10 bg-white border border-gray-300 rounded mt-1 max-h-60 overflow-auto w-full shadow-lg">
            <li
              className="px-3 py-1.5 text-sm hover:bg-gray-100 cursor-pointer text-gray-500 italic"
              onClick={() => handleSelect("")}
            >
              League Average
            </li>
            {items.slice(0, 50).map((item) => (
              <li
                key={item}
                className="px-3 py-1.5 text-sm hover:bg-gray-100 cursor-pointer"
                onClick={() => handleSelect(item)}
              >
                {item}
              </li>
            ))}
          </ul>
        )}
      </div>

      {selectedName && (
        <button
          className="text-sm text-gray-500 hover:text-red-500"
          onClick={() => {
            onSelect(undefined);
            setSearch("");
          }}
        >
          Clear
        </button>
      )}
    </div>
  );
}
