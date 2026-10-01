# Team efficiency by score margin and time remaining

Analyzed regular seasons 2022–23 through 2025–26. Latest-season source dates: 2025-10-21 to 2026-04-13. Season labels use the ending year; 2026 means 2025–26.

## Findings

In 2025–26, no eligible team has a leading-by-11+ versus trailing-by-11+ gap with a 95% interval entirely above zero. Different situational profiles alone do not establish a stable “frontrunner” identity. Across the analyzed seasons and all three tested contrasts, 0 eligible positive gaps have intervals above zero; these are exploratory intervals before accounting for multiple comparisons.

The largest current-season estimates are below. A positive gap means better league-relative net efficiency while ahead. Teams below the eligibility floor remain visible and marked. Compare the intervals and opponent-adjusted gaps before interpreting a ranking.

| team | gap | ci_low | ci_high | schedule_adjusted_gap | eligible |
| --- | --- | --- | --- | --- | --- |
| UTA | +4.23 | -19.01 | +19.91 | -1.47 | True |
| ATL | +3.56 | -7.70 | +13.57 | -2.16 | True |
| WAS | +2.78 | -14.05 | +13.20 | -1.59 | True |
| MIL | +2.75 | -7.52 | +12.02 | -5.18 | True |
| MIA | +1.54 | -11.21 | +12.70 | -3.65 | True |

At the other end, these teams have the largest league-relative advantage while trailing by 11+ compared with leading by 11+. A negative gap favors performance from behind. This is still a descriptive comparison with exploratory intervals, and trailing by a wide margin is different from a close late game.

| team | gap | ci_low | ci_high | schedule_adjusted_gap | eligible |
| --- | --- | --- | --- | --- | --- |
| DAL | -22.36 | -39.83 | -10.82 | -25.61 | True |
| CLE | -22.21 | -36.53 | -10.30 | -27.35 | True |
| SAS | -19.47 | -37.94 | -7.06 | -24.59 | True |
| DEN | -15.87 | -34.77 | +2.42 | -21.12 | True |
| DET | -14.09 | -30.41 | -2.98 | -18.36 | True |

The big-lead versus clutch contrast asks a different question: how leading by 11+ compares with playing within five points in the final five minutes or overtime. It uses separate samples and intervals.

| team | gap | ci_low | ci_high | schedule_adjusted_gap | eligible |
| --- | --- | --- | --- | --- | --- |
| HOU | +15.65 | -5.44 | +37.33 | +11.98 | True |
| PHX | +8.24 | -10.67 | +27.32 | +5.48 | True |
| DEN | +2.13 | -16.89 | +22.07 | -1.42 | True |
| POR | -4.04 | -22.91 | +16.93 | -8.39 | True |
| CHI | -5.50 | -32.03 | +17.51 | -8.08 | True |

Profiles for the two largest eligible ahead-versus-behind estimates show the offense and defense components, usual baseline, and sample sizes. “Trailing” and “under pressure” are different states.

| team | profile | off_rating | def_rating | relative_net | lift_vs_usual | off_n | def_n |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ATL | Overall | 115.0 | 112.8 | +2.42 | +0.00 | 8450 | 8436 |
| UTA | Overall | 112.2 | 120.2 | -8.52 | +0.00 | 8248 | 8247 |
| ATL | Ahead 11+ | 113.9 | 112.0 | +2.72 | +0.30 | 1728 | 1912 |
| UTA | Ahead 11+ | 116.2 | 119.1 | -3.15 | +5.37 | 606 | 726 |
| ATL | Behind 11+ | 113.7 | 114.4 | -0.84 | -3.25 | 1337 | 1187 |
| UTA | Behind 11+ | 114.7 | 121.2 | -7.37 | +1.15 | 2570 | 2307 |
| ATL | Clutch (last 5, within 5) | 111.9 | 112.1 | -0.14 | -2.56 | 270 | 281 |
| UTA | Clutch (last 5, within 5) | 103.9 | 110.0 | -6.13 | +2.39 | 255 | 250 |

Year-to-year correlations of the ahead-versus-behind point estimates are below. Roster changes, exposure, and measurement noise all contribute; this does not establish that team identity has no effect.

| season | 2023 | 2024 | 2025 | 2026 |
| --- | --- | --- | --- | --- |
| 2023 | +1.00 | -0.19 | +0.05 | -0.25 |
| 2024 | -0.19 | +1.00 | +0.21 | -0.12 |
| 2025 | +0.05 | +0.21 | +1.00 | -0.21 |
| 2026 | -0.25 | -0.12 | -0.21 | +1.00 |

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

Resample entire games within each team-season, preserving offense, defense and all states together. Use 1,000 draws and seed 7 for 95% percentile intervals. League benchmarks remain fixed, so intervals condition on the observed league reference. Contrasts use the same bootstrap game weights for both states. A profile is eligible at ≥300 offensive possessions, ≥300 defensive possessions, and ≥20 distinct games; this is a display floor, not proof of reliability. Intervals are not corrected for searching across teams/seasons/contrasts.

State exposure is endogenous: good teams lead more, opponents change tactics, trailing teams may face reserves, and injuries/lineups can vary by state. Relative matching controls coarse game state, not all those selection effects. Teams retain their own within-state time/margin mix; the comparison does not force a common exposure distribution. Do not interpret a large positive gap as a psychological trait or causal effect of leading.

Late intentional fouling can affect offense and defense. The exported contrast excluding the final two regulation minutes is a sensitivity check; it does not identify actual fouls or remove all garbage time. Wide margins earlier in games remain included. Conditional ORtg/DRtg use different offensive/defensive state samples, so their difference is not directly a probability of winning or coming back.

## Data quality and provenance

| season | source_games | analyzed_games | quarantined_games | possessions |
| --- | --- | --- | --- | --- |
| 2022–23 | 1230 | 1188 | 42 | 236221 |
| 2023–24 | 1228 | 1205 | 23 | 237755 |
| 2024–25 | 1230 | 1213 | 17 | 240445 |
| 2025–26 | 1230 | 1217 | 13 | 245451 |

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
