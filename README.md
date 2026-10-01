# NBA play-by-play analysis

This repo contains historical shot-quality notebooks, a FastAPI/React app for
player on/off data, and a reproducible team efficiency analysis by game state.

Start with [the team-situations findings](reports/team_situations.md). It analyzes
2022–23 through 2025–26 with signed possession-start margins, time remaining,
league-relative ORtg/DRtg, game-bootstrap intervals, and source reconciliation.
The complete data outputs and excluded-game audit are in `reports/`.

[The playoff backtest](reports/playoff_prediction.md) extends this to 2015–16
through 2025–26. It tests whether regular-season situation profiles improve
series predictions beyond seed, wins, scoring margin and schedule strength,
and measures playoff overperformance against the starting bracket's expected
series wins. Each postseason is predicted using only earlier seasons.

## Run the app

```sh
python3 -m venv .venv
.venv/bin/pip install -e '.[dev]'
.venv/bin/python -m uvicorn backend.main:app --reload
npm ci --prefix frontend
npm run dev --prefix frontend
```

Open http://localhost:5173/team-situations for the new team analysis. The existing
on/off view remains at `/`. The new view uses checked-in aggregate results, so
it works without downloading the raw data. Raw archives are git-ignored.

## Refresh the analysis

```sh
.venv/bin/python -m backend.analysis.run --download --seasons 2023 2024 2025 2026
```

Season arguments use the **ending year**. The public archive and original
`processed_pbp/shot_detail_pbp_YYYY.csv` names use **starting years**: the legacy
file called `2022` contains 2022–23. Prefer game IDs over filenames or calendar
month heuristics, especially for the 2020 bubble.

The 2025–26 archive uses NBA CDN actions rather than the older possession CSVs.
The loader reconstructs possessions with pinned pbpstats live rules and excludes
games with parsing or scoring disagreements; the audit records every reason.
Cross-season comparisons can also reflect this provider change.

The runner writes findings, CSVs, a compact JSON dataset, and source fingerprints.
Run without `--download` to use local inputs. For an inline explorer, pass
`--visualization-output /absolute/writable/path/team-game-state.html`.

Refresh the playoff analysis and its audited historical inputs:

```sh
.venv/bin/python -m backend.analysis.playoff_run --download
```

The playoff runner uses a separate canonical historical possession cache.
The 2026 playoff archive was incomplete; checked-in official NBA game results
in `backend/analysis/metadata/playoffs_2026.json` complete the bracket and are
validated against every overlapping archived game.

## Checks

```sh
.venv/bin/python -m pytest -q
npm run build --prefix frontend
npm run lint --prefix frontend
```

## Repo review and targeted refactor

- `eFG_shot_viz.ipynb`, `Shot_Rank.ipynb`, and `nba_shot_viz.py` already explored
  time versus absolute margin. They measure shooting efficiency, not full
  possession efficiency, and discard the lead/deficit sign. Their old exports
  cannot answer the frontrunner question directly.
- Shared regulation/overtime clock math now lives in `backend/metrics/game_state.py`.
  Backend score fills are confined to each game, retain signed pre-event home
  margins, and classify rebounds using shot-team ownership rather than cumulative
  `Off:N Def:N` text. Reprocess prior enriched events with `--force` to adopt this.
- Shot-chart conversion now matches pre-event margins, including missed shots,
  and handles overtime consistently. Notebook helpers copy inputs and aggregate
  numeric shot counts explicitly; the cleaned frame is passed to bucketing.
- Season queries support leading-zero string and numeric NBA IDs. Player and
  situation SQL filters bind values, including names with apostrophes. Missing
  frontend season/entity client functions were restored so the whole app builds.
- The new team analysis separates collection, normalization, metrics, reporting,
  and presentation. Possession points are independently reconciled; uncertainty
  and thin samples are visible. A synthetic suite covers attribution, technical
  FTs, duplicates, game boundaries, clock/season rules and conditional benchmarks.
- `/api/ratings` is retained but marked deprecated: its individual Oliver ratings
  infer minutes from event shares and approximate opponent statistics. It is not
  validated for player situational inference. `/api/team-situations` provides
  the measured team analysis.

## Remaining limitations

The old on/off processor still estimates each team's possessions by halving all
possessions and infers minutes from a fixed pace; its calendar-month season rule
also needs special handling for the 2020 bubble. Reliable player-level refactoring
needs the missing lineup dataset and independent minute/box-score validation.
The legacy second-level score interpolation can use a later observed margin;
regenerate from raw possession/event data for inference. The new analysis bypasses
both paths and preserves the historical notebooks for reference.

The existing frontend lockfile reports 15 dependency advisories during `npm ci`
(11 high). Dependency upgrades were not mixed into the analytical changes.
