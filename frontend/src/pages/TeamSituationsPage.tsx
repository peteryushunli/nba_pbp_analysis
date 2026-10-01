import { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { fetchTeamSituations } from '../api/client';

export interface SituationCell {
  season: number; team: string; margin_bucket: string; time_bucket: string;
  off_n: number; def_n: number; games: number;
  off_relative: number | null; def_relative: number | null; relative_net: number | null;
}
interface Profile extends Omit<SituationCell, 'margin_bucket' | 'time_bucket'> {
  profile: string; off_rating: number | null; def_rating: number | null;
  lift_vs_usual: number | null; ci_low: number | null; ci_high: number | null; eligible: boolean;
}
interface Contrast {
  season: number; team: string; contrast: string; gap: number;
  ci_low: number; ci_high: number; eligible: boolean; schedule_adjusted_gap: number;
}
export interface TeamSituationsData {
  cells: SituationCell[]; profiles: Profile[]; contrasts: Contrast[];
  margins: string[]; times: string[]; coverage: Record<string, { games: number; source_games: number; date_min: string; date_max: string }>;
}
const signed = (n: number | null) => n === null || !Number.isFinite(n) ? '—' : `${n > 0 ? '+' : ''}${n.toFixed(1)}`;
const rating = (n: number | null) => n === null ? '—' : n.toFixed(1);
const cellClass = 'p-2 text-right whitespace-nowrap tabular-nums';

export function TeamSituationsPage() {
  const { data, isLoading, error } = useQuery({ queryKey: ['team-situations'], queryFn: fetchTeamSituations });
  const [season, setSeason] = useState(0);
  const [team, setTeam] = useState('MIN');
  const [contrast, setContrast] = useState('Ahead vs behind');
  const [selected, setSelected] = useState<SituationCell | null>(null);
  if (isLoading) return <p>Loading team situations…</p>;
  if (error) return <p role="alert">Could not load analysis: {error.message}</p>;
  if (!data) return null;
  const seasons = [...new Set(data.profiles.map(p => p.season))].sort((a, b) => b - a);
  const effectiveSeason = season || seasons[0];
  const teams = [...new Set(data.profiles.map(p => p.team))].sort();
  const profiles = data.profiles.filter(p => p.season === effectiveSeason && p.team === team);
  const cells = data.cells.filter(p => p.season === effectiveSeason && p.team === team);
  const ranks = data.contrasts.filter(p => p.season === effectiveSeason && p.contrast === contrast).sort((a, b) => b.gap - a.gap);
  const coverage = data.coverage[effectiveSeason];
  const detail = selected?.season === effectiveSeason && selected.team === team ? selected : null;
  return <div className="space-y-5">
    <div><h1 className="text-2xl font-semibold">Team efficiency by game state</h1>
      <p className="text-sm text-gray-600 mt-1">Signed margin at possession start. League comparisons match season, margin and time remaining.</p></div>
    <div className="flex flex-wrap gap-4">
      <label>Season <select className="border rounded px-2 py-1 bg-white" value={effectiveSeason} onChange={e => setSeason(+e.target.value)}>
        {seasons.map(s => <option key={s} value={s}>{s - 1}–{String(s).slice(-2)}</option>)}</select></label>
      <label>Team <select className="border rounded px-2 py-1 bg-white" value={team} onChange={e => setTeam(e.target.value)}>
        {teams.map(t => <option key={t}>{t}</option>)}</select></label>
    </div>
    <p className="text-sm text-gray-600">{coverage.games.toLocaleString()} reconciled games of {coverage.source_games.toLocaleString()} source games across the league. Regular-season source dates: {coverage.date_min} to {coverage.date_max}.</p>
    <section><h2 className="text-lg font-medium">League-relative net rating · points per 100 possessions</h2>
      <div className="overflow-x-auto mt-2"><table className="w-full text-sm bg-white" aria-label="Net rating by signed margin and minutes remaining">
        <thead><tr><th className="p-2 text-left">Team margin</th>{data.times.map(t => <th key={t} className="p-2 whitespace-nowrap">{t}</th>)}</tr></thead>
        <tbody>{data.margins.slice().reverse().map(m => <tr key={m}><th scope="row" className="p-2 text-left whitespace-nowrap">{m}</th>
          {data.times.map(t => { const c = cells.find(c => c.margin_bucket === m && c.time_bucket === t);
            const small = c && (Math.min(c.off_n, c.def_n) < 50 || c.games < 5);
            return <td key={t} className={`text-center ${c?.relative_net == null || small ? 'bg-gray-100' : c.relative_net >= 0 ? 'bg-blue-100' : 'bg-orange-100'}`}>
              {c?.relative_net == null ? '—' : <button className="w-full p-2 hover:underline" onClick={() => setSelected(c)} aria-label={`${m}, ${t} minutes; net ${signed(c.relative_net)}; ${c.off_n} offense and ${c.def_n} defense possessions`}>{signed(c.relative_net)}{small ? '*' : ''}</button>}</td>;
          })}</tr>)}</tbody></table></div>
      <p className="text-sm text-gray-600 mt-2">Positive = better. * = fewer than 50 possessions on either side or 5 games. Minutes remaining; overtime separate.</p>
      <p className="text-sm mt-2" aria-live="polite">{detail ? `${detail.margin_bucket}, ${detail.time_bucket} minutes: relative offense ${signed(detail.off_relative)}, defense ${signed(detail.def_relative)}; ${detail.off_n} offensive / ${detail.def_n} defensive possessions in ${detail.games} games.` : 'Select a cell for offense, defense and sample counts.'}</p>
    </section>
    <section><h2 className="text-lg font-medium">{team} profiles</h2><div className="overflow-x-auto"><table className="w-full text-sm">
      <thead><tr>{['State', 'ORtg / DRtg', 'Relative offense / defense', 'Relative net [95% CI]', 'Vs usual', 'O / D possessions', 'Games'].map(h => <th key={h} className="p-2 text-left">{h}</th>)}</tr></thead>
      <tbody>{profiles.map(p => <tr key={p.profile} className="border-t border-gray-200"><th className="p-2 text-left font-normal">{p.profile}{p.eligible ? '' : ' †'}</th>
        <td className={cellClass}>{rating(p.off_rating)} / {rating(p.def_rating)}</td><td className={cellClass}>{signed(p.off_relative)} / {signed(p.def_relative)}</td>
        <td className={cellClass}>{signed(p.relative_net)} [{signed(p.ci_low)}, {signed(p.ci_high)}]</td><td className={cellClass}>{signed(p.lift_vs_usual)}</td><td className={cellClass}>{p.off_n} / {p.def_n}</td><td className={cellClass}>{p.games}</td></tr>)}</tbody></table></div>
      <p className="text-sm text-gray-600">Lower raw DRtg is better; positive relative defense is better. Vs usual subtracts the team's overall relative net. † = below 300 O, 300 D or 20 games.</p></section>
    <section><h2 className="text-lg font-medium">Leading versus pressure</h2><label className="block my-2">Comparison <select className="border rounded px-2 py-1 bg-white" value={contrast} onChange={e => setContrast(e.target.value)}>{[...new Set(data.contrasts.map(c => c.contrast))].map(c => <option key={c}>{c}</option>)}</select></label>
      <div className="overflow-x-auto"><table className="w-full text-sm"><thead><tr>{['Team', 'Relative gap [95% CI]', 'Opponent-adjusted gap'].map(h => <th key={h} className="p-2 text-left">{h}</th>)}</tr></thead>
        <tbody>{ranks.map(g => <tr key={g.team} className="border-t border-gray-200"><th className="p-2 text-left font-normal">{g.team}{g.eligible ? '' : ' †'}</th><td className={cellClass}>{signed(g.gap)} [{signed(g.ci_low)}, {signed(g.ci_high)}]</td><td className={cellClass}>{signed(g.schedule_adjusted_gap)}</td></tr>)}</tbody></table></div>
      <p className="text-sm text-gray-600 mt-2">Positive gap = better while ahead. Intervals resample entire games with fixed league benchmarks. These exploratory comparisons do not establish a causal or stable psychological trait. Opponent adjustment uses full-season strength. Lineups, reserves, late fouling and incomplete coverage can influence results.</p></section>
  </div>;
}
