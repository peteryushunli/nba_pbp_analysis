"""Run with: python -m backend.analysis.run --download --seasons 2023 2024 2025 2026."""
import argparse
from pathlib import Path
import json
import hashlib
import pandas as pd
from backend.analysis.download import download_archives, source_kinds, SOURCE
from backend.analysis.possessions import load_season
from backend.analysis.situational import analyze, MARGINS, TIMES

ROOT = Path(__file__).resolve().parents[2]


def markdown_table(df: pd.DataFrame, columns: list[str]) -> str:
    d = df[columns].copy()
    for c in columns:
        if pd.api.types.is_float_dtype(d[c]):
            d[c] = d[c].map(lambda x: (f'{x:.1f}' if c in ['off_rating','def_rating'] else f'{x:+.2f}') if pd.notna(x) else '—')
    return '\n'.join(['| '+' | '.join(str(c) for c in columns)+' |','| '+' | '.join(['---']*len(columns))+' |'] +
                     ['| '+' | '.join(str(x) for x in row)+' |' for row in d.to_numpy()])


def write_report(profiles: pd.DataFrame, contrasts: pd.DataFrame, audits: dict, out: Path, draws: int) -> None:
    latest = int(profiles.season.max())
    g = contrasts[contrasts.season.eq(latest) & contrasts.contrast.eq('Ahead vs behind')].sort_values('gap',ascending=False)
    clutch = contrasts[contrasts.season.eq(latest)&contrasts.contrast.eq('Ahead vs clutch')].sort_values('gap',ascending=False)
    positive = contrasts[contrasts.eligible & contrasts.positive_ci]
    latest_positive = g[g.eligible & g.positive_ci]
    finding = ('No eligible team has' if latest_positive.empty else f'{len(latest_positive)} eligible teams have')
    example_teams = g[g.eligible].head(2).team.tolist()
    coverage_dates = f"{audits[latest]['date_min'][:10]} to {audits[latest]['date_max'][:10]}"
    stability = contrasts[contrasts.contrast.eq('Ahead vs behind')].pivot(index='team',columns='season',values='gap').corr()
    coverage = pd.DataFrame([{'season':f'{int(s)-1}–{str(s)[-2:]}',
        'source_games':a['games_in_archive'],'analyzed_games':a['games_analyzed'],
        'quarantined_games':len(a['quarantined_games']),'possessions':a['possessions_analyzed']}
        for s,a in audits.items()])
    report = f'''# Team efficiency by score margin and time remaining

Analyzed regular seasons {min(audits)-1}–{str(min(audits))[-2:]} through {latest-1}–{str(latest)[-2:]}. Latest-season source dates: {coverage_dates}. Season labels use the ending year; 2026 means 2025–26.

## Findings

In {latest-1}–{str(latest)[-2:]}, {finding.lower()} a leading-by-11+ versus trailing-by-11+ gap with a 95% interval entirely above zero. Different situational profiles alone do not establish a stable “frontrunner” identity. Across the analyzed seasons and all three tested contrasts, {len(positive)} eligible positive gaps have intervals above zero; these are exploratory intervals before accounting for multiple comparisons.

The largest current-season estimates are below. A positive gap means better league-relative net efficiency while ahead. Teams below the eligibility floor remain visible and marked. Compare the intervals and opponent-adjusted gaps before interpreting a ranking.

{markdown_table(g[g.eligible].head(5),['team','gap','ci_low','ci_high','schedule_adjusted_gap','eligible'])}

At the other end, these teams have the largest league-relative advantage while trailing by 11+ compared with leading by 11+. A negative gap favors performance from behind. This is still a descriptive comparison with exploratory intervals, and trailing by a wide margin is different from a close late game.

{markdown_table(g[g.eligible].tail(5).sort_values('gap'),['team','gap','ci_low','ci_high','schedule_adjusted_gap','eligible'])}

The big-lead versus clutch contrast asks a different question: how leading by 11+ compares with playing within five points in the final five minutes or overtime. It uses separate samples and intervals.

{markdown_table(clutch[clutch.eligible].head(5),['team','gap','ci_low','ci_high','schedule_adjusted_gap','eligible'])}

Profiles for the two largest eligible ahead-versus-behind estimates show the offense and defense components, usual baseline, and sample sizes. “Trailing” and “under pressure” are different states.

{markdown_table(profiles[profiles.season.eq(latest)&profiles.team.isin(example_teams)&profiles.profile.isin(['Overall','Ahead 11+','Behind 11+','Clutch (last 5, within 5)'])],['team','profile','off_rating','def_rating','relative_net','lift_vs_usual','off_n','def_n'])}

Year-to-year correlations of the ahead-versus-behind point estimates are below. Roster changes, exposure, and measurement noise all contribute; this does not establish that team identity has no effect.

{markdown_table(stability.reset_index(),['season']+list(stability.columns))}

## Definitions and comparison

- ORtg = team points / offensive possessions × 100; DRtg = opponent points / defensive possessions × 100. Lower raw DRtg is better. Net is their difference. Count offense and defense separately; never halve a combined possession count.
- Assign signed margin and clock at the **start of each possession**, from the focal team's perspective. Keep the full possession together if it crosses a margin/time boundary. A +11 lead and −11 deficit are separate states.
- Margin cells: behind 21+, behind 11–20, behind 6–10, within 5, ahead 6–10, ahead 11–20, ahead 21+. The central bin includes both small leads and small deficits. This is a coarse conditional comparison, not exact-point matching.
- Time cells: four-minute intervals through most of regulation, then 8–5, 5–2 and 2–0 minutes. Overtime is separate. Clock boundaries belong to the later/less-time-remaining bin. All overtime is clutch-eligible when within five; its period length is five minutes.
- For each team, estimate league ORtg and DRtg within the **same season, signed margin cell and time cell**, excluding the focal team's observations. Benchmarks weight possessions, not team averages. Aggregate expected points using that team's actual exposure. An extreme cell with no other-team exposure falls back to the same-season/time benchmark; exported fallback counts identify these rare cases.
- Relative offense = observed ORtg − expected ORtg. Relative defense = expected DRtg − observed DRtg. Relative net = their sum. **Positive relative values always mean better**. A +5 relative net means five points per 100 possessions better than the conditional league benchmark.
- Lift vs usual = situational relative net − that team's season-wide matched relative net. This separates overall strength from a situational change. The difference between two states already cancels that usual baseline.
- The opponent sensitivity adds the opponent's full-season defensive/offensive deviation from league average to expected rates. It is a descriptive, additive check using the same season's data, not a causal model, cross-validation or a fitted opponent/state interaction.
- Shrunk relative rates add 200 zero-residual possessions to each side. Raw values and counts remain exported. This is a sensitivity aid, not an estimated optimal prior.

## Sampling and interpretation

Resample entire games within each team-season, preserving offense, defense and all states together. Use {draws:,} draws and seed 7 for 95% percentile intervals. League benchmarks remain fixed, so intervals condition on the observed league reference. Contrasts use the same bootstrap game weights for both states. A profile is eligible at ≥300 offensive possessions, ≥300 defensive possessions, and ≥20 distinct games; this is a display floor, not proof of reliability. Intervals are not corrected for searching across teams/seasons/contrasts.

State exposure is endogenous: good teams lead more, opponents change tactics, trailing teams may face reserves, and injuries/lineups can vary by state. Relative matching controls coarse game state, not all those selection effects. Teams retain their own within-state time/margin mix; the comparison does not force a common exposure distribution. Do not interpret a large positive gap as a psychological trait or causal effect of leading.

Late intentional fouling can affect offense and defense. The exported contrast excluding the final two regulation minutes is a sensitivity check; it does not identify actual fouls or remove all garbage time. Wide margins earlier in games remain included. Conditional ORtg/DRtg use different offensive/defensive state samples, so their difference is not directly a probability of winning or coming back.

## Data quality and provenance

{markdown_table(coverage,list(coverage.columns))}

Source: [shufinskiy/nba_data](https://github.com/shufinskiy/nba_data), regular-season `pbpstats` possessions plus `nbastats` event logs through 2024–25, and the `cdnnba_2025` action archive for 2025–26, retrieved October 1, 2026. Archive filenames use the **starting year**; public analysis arguments use the **ending year**. [Possession margin semantics](https://github.com/dblackrun/pbpstats/blob/main/pbpstats/resources/possessions/possession.py) are from the offensive team's perspective at possession start. [NBA rating definitions](https://www.nba.com/stats/help/glossary) use points per 100 possessions.

For the legacy format, the source repeats possession totals once for each event. Deduplicate the complete possession payload excluding DESCRIPTION/URL. Count each resulting possession once, including zero-point possessions. This follows the archive's possession boundaries; exact agreement with NBA's internal possession-count convention is not claimed.

Use made field-goal totals and match each made free throw to the NBA event log by game, period, description and possession clock interval. Credit technical free throws to the actual scoring team even when it is defending. Reconcile each team's total points with classified NBA scoring events and each game's final cumulative score. Quarantine the whole game if either fails or a made FT cannot be uniquely matched. Full IDs, team coverage, dates, source URLs and SHA-256 hashes are in `team_situations_audit.json`. Event-only game IDs missing from the possession source are recorded separately. Quarantine counts in the table refer to games present in the possession archive.

For 2025–26, reconstruct possessions with [pbpstats 1.3.11 live rules](https://pbpstats.readthedocs.io/en/latest/quickstart.html), preserving offensive rebounds, free throws and period boundaries. Skip lineup inference because this analysis uses team possessions. Omit between-period substitutions/timeouts preceding the new period's start marker: CDN labels these with the new period but the old possession team. Quarantine games with basketball actions before that marker, ambiguous ownership or parsing errors. Check classified scoring against every cumulative scoreboard total and possession scoring against per-team event totals; record every exclusion reason in the audit. These checks share the CDN source, unlike the legacy two-source reconciliation. Cross-season differences may reflect provider/boundary changes as well as team changes. CDN timestamps and displayed source-date ranges are UTC.

Do not present these as complete official-season totals: source gaps and exclusions may introduce selection bias. Cached normalized possessions are in `data/processed/situational/`; source files are local and git-ignored.

## Reproduce and inspect

```sh
python3 -m venv .venv
.venv/bin/pip install -e '.[dev]'
.venv/bin/python -m backend.analysis.run --download --seasons 2023 2024 2025 2026
.venv/bin/python -m pytest -q
```

The three CSVs contain all cells, profiles and contrast intervals. `team_situations.json` is the compact app/explorer dataset. Inputs and output paths can be set with `--data-dir` and `--output-dir`; season arguments always use ending years. Source SHA-256 hashes invalidate stale normalized caches automatically. Increment the normalization version or delete the corresponding cache after changing normalization logic.
'''
    (out/'team_situations.md').write_text(report)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seasons',nargs='+',type=int,default=[2023,2024,2025,2026])
    parser.add_argument('--download',action='store_true')
    parser.add_argument('--data-dir',type=Path,default=ROOT/'data')
    parser.add_argument('--output-dir',type=Path,default=ROOT/'reports')
    parser.add_argument('--visualization-output',type=Path,default=None)
    parser.add_argument('--draws',type=int,default=1000)
    args=parser.parse_args()
    seasons=sorted(set(args.seasons))
    if args.download:download_archives(seasons,args.data_dir)
    frames,audits=[],{}
    for s in seasons:
        p,a=load_season(s,args.data_dir);frames.append(p)
        a['sources']=[]
        for kind in source_kinds(s):
            csv=args.data_dir/'external'/kind/f'{kind}_{s-1}.csv'
            a['sources'].append({'url':f'{SOURCE}/{kind}_{s-1}.tar.xz',
                                'csv_sha256':hashlib.sha256(csv.read_bytes()).hexdigest()})
        audits[s]=a
    cells,profiles,contrasts=analyze(pd.concat(frames,ignore_index=True),draws=args.draws)
    args.output_dir.mkdir(parents=True,exist_ok=True)
    for name,df in [('cells',cells),('profiles',profiles),('contrasts',contrasts)]:
        df.to_csv(args.output_dir/f'team_situations_{name}.csv',index=False)
    (args.output_dir/'team_situations_audit.json').write_text(json.dumps(audits,indent=2))
    # Only fields useful for exploration; precision retained in CSVs.
    cell_fields=['season','team','margin_bucket','time_bucket','off_n','def_n','games','off_relative','def_relative','relative_net']
    profile_fields=['season','team','profile','off_n','def_n','games','off_rating','def_rating','off_relative','def_relative',
        'relative_net','lift_vs_usual','ci_low','ci_high','eligible','schedule_adjusted_relative_net']
    records=lambda df:json.loads(df.round(2).to_json(orient='records'))
    payload={'cells':records(cells[cell_fields]),'profiles':records(profiles[profile_fields]),
        'contrasts':records(contrasts),'margins':MARGINS,'times':TIMES,
        'coverage':{str(s):{'games':a['games_analyzed'],'source_games':a['games_in_archive'],'date_min':a['date_min'][:10],'date_max':a['date_max'][:10]} for s,a in audits.items()}}
    (args.output_dir/'team_situations.json').write_text(json.dumps(payload,separators=(',',':')))
    write_report(profiles,contrasts,audits,args.output_dir,args.draws)
    if args.visualization_output:
        # Arrays avoid repeating field names in the inline data. Encode categorical
        # cell axes as indices so four seasons remain comfortably below 1 MB.
        payload['cells'] = [[c['season'],c['team'],MARGINS.index(c['margin_bucket']),TIMES.index(c['time_bucket']),
            c['off_n'],c['def_n'],c['games'],c['off_relative'],c['def_relative'],c['relative_net']]
            for c in payload['cells']]
        for key in ['profiles','contrasts']:
            fields = list(payload[key][0])
            payload[key+'_fields'] = fields
            payload[key] = [[row[f] for f in fields] for row in payload[key]]
        template = Path(__file__).with_name('explorer.html').read_text()
        args.visualization_output.parent.mkdir(parents=True,exist_ok=True)
        fragment = template.replace('__DATA__',json.dumps(payload,separators=(',',':')))
        if len(fragment.encode()) >= 1_000_000:
            raise ValueError('Inline explorer exceeds 1 MB; reduce dataset size')
        args.visualization_output.write_text(fragment)
    print(f'Analyzed {sum(a["games_analyzed"] for a in audits.values()):,} games. Results: {args.output_dir}')

if __name__=='__main__':main()
