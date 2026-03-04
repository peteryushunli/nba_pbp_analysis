import { useSeasons } from "../hooks/useSeasons";

interface Props {
  value: number;
  onChange: (season: number) => void;
}

export function SeasonSelector({ value, onChange }: Props) {
  const { data: seasons, isLoading } = useSeasons();

  return (
    <select
      className="border border-gray-300 rounded px-3 py-1.5 text-sm bg-white"
      value={value}
      onChange={(e) => onChange(Number(e.target.value))}
      disabled={isLoading}
    >
      {!seasons && <option>Loading...</option>}
      {seasons?.map((s) => (
        <option key={s.season} value={s.season}>
          {s.display_name} ({s.game_count} games)
        </option>
      ))}
    </select>
  );
}
