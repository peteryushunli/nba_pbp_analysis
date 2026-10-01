"""Normalize the NBA CDN action archive using pbpstats' live possession rules.

Only team possessions are needed: skip lineup inference, which would otherwise
require starter overrides. All scoring must reconcile before admitting a game.
"""
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
import hashlib
import json
import re

import pandas as pd
from pbpstats.data_loader.live.enhanced_pbp.loader import LiveEnhancedPbpLoader
from pbpstats.data_loader.nba_possession_loader import NbaPossessionLoader
from pbpstats.resources.enhanced_pbp import FieldGoal, FreeThrow
from pbpstats.resources.possessions.possession import Possession

VERSION = 3
COLUMNS = ['game_id', 'season', 'possession_id', 'offense', 'defense', 'period',
           'seconds_remaining', 'regulation_remaining', 'margin', 'points', 'other_points']


class TeamPbpLoader(LiveEnhancedPbpLoader):
    def _set_period_start_items(self):
        for i in self.start_period_indices:
            self.items[i].team_starting_with_ball = self.items[i].get_team_starting_with_ball()


class TeamPossessionLoader(NbaPossessionLoader):
    def __init__(self, actions: list[dict], game_id: str):
        source = SimpleNamespace(file_directory=None,
            load_data=lambda _: {'game': {'actions': actions}})
        self.events = TeamPbpLoader(game_id, source).items
        self.items = [Possession(e) for e in self._split_events_by_possession()]
        self._add_extra_attrs_to_all_possessions()


def seconds(clock: str) -> float:
    match = re.fullmatch(r'PT(\d+)M(\d+(?:\.\d+)?)S', clock)
    if not match:
        raise ValueError(f'Invalid CDN clock: {clock}')
    return int(match[1]) * 60 + float(match[2])


def normalize_live_game(frame: pd.DataFrame, season: int) -> list[dict]:
    game_id = str(int(frame.gameId.iloc[0])).zfill(10)
    if not game_id.startswith(f'002{str(season-1)[-2:]}'):
        raise ValueError('Game ID disagrees with requested regular season')
    frame = frame.sort_values('orderNumber', kind='stable')
    if not ((frame.actionType.eq('game') & frame.subType.eq('end')).any()):
        raise ValueError('Game has no final marker')
    teams = {int(t): str(g.teamTricode.dropna().iloc[0])
             for t, g in frame.dropna(subset=['teamId', 'teamTricode']).groupby('teamId')}
    if len(teams) != 2:
        raise ValueError('Game must have two teams')
    actions = [{k:v for k,v in row.items() if v is not None}
               for row in frame.astype(object).where(frame.notna(), None).to_dict('records')]
    # CDN puts between-period substitutions before Period Start while labeling
    # them with the NEW period and the OLD possession team. For team analysis,
    # omit those administrative rows so ownership/start margin use the opening
    # ball team. Preserve the order of all basketball actions.
    ordered = []
    for period in sorted(frame.period.unique()):
        period_actions = [a for a in actions if a['period'] == period]
        starts = [i for i,a in enumerate(period_actions)
                  if a['actionType'] == 'period' and a.get('subType') == 'start']
        if len(starts) != 1:
            raise ValueError('Period must have one start marker')
        index = starts[0]
        if any(a['actionType'] not in ['substitution','timeout'] for a in period_actions[:index]):
            raise ValueError('Basketball action precedes period start')
        ordered.extend(period_actions[index:])
    actions = ordered
    loader = TeamPossessionLoader(actions, game_id)
    expected = defaultdict(int)
    for action in actions:
        if action.get('shotResult') == 'Made':
            value = {'2pt': 2, '3pt': 3, 'freethrow': 1}.get(action['actionType'])
            if value is None or int(action.get('teamId', 0)) not in teams:
                raise ValueError('Unclassified scoring action')
            expected[int(action['teamId'])] += value
    # Each classified score must also match the running scoreboard total.
    cumulative = 0
    for action in actions:
        if action.get('shotResult') == 'Made':
            cumulative += {'2pt': 2, '3pt': 3, 'freethrow': 1}[action['actionType']]
        if cumulative != int(action['scoreHome']) + int(action['scoreAway']):
            raise ValueError('Scoring events disagree with cumulative scoreboard')
    rows, scored = [], defaultdict(int)
    for i, p in enumerate(loader.items):
        scoring = [e for e in p.events if isinstance(e, (FieldGoal, FreeThrow)) and e.is_made]
        offense = p.offense_team_id
        if offense not in teams:
            if scoring:
                raise ValueError('Scoring possession has no team')
            continue  # e.g. the non-possession Game End marker
        defense = next(t for t in teams if t != offense)
        own = other = 0
        for e in scoring:
            scored[e.team_id] += e.shot_value
            if e.team_id == offense:
                own += e.shot_value
            elif e.team_id == defense:
                if not isinstance(e, FreeThrow):
                    raise ValueError('Defending team scored a field goal inside possession')
                other += e.shot_value
            else:
                raise ValueError('Unknown scoring team')
        clock = seconds(p.start_time)
        if not 0 <= clock <= (720 if p.period <= 4 else 300):
            raise ValueError('Invalid possession clock')
        rows.append(dict(game_id=int(game_id), season=season, possession_id=i,
            offense=teams[offense], defense=teams[defense], period=p.period,
            seconds_remaining=clock, regulation_remaining=(4-p.period)*720+clock if p.period <= 4 else 0,
            margin=p.start_score_margin, points=own, other_points=other))
    if dict(scored) != dict(expected):
        raise ValueError('Possession points disagree with per-team event points')
    if not rows:
        raise ValueError('No possessions')
    return rows


def load_live_season(season: int, data_dir: Path) -> tuple[pd.DataFrame, dict]:
    source = data_dir/'external'/'cdnnba'/f'cdnnba_{season-1}.csv'
    cache = data_dir/'processed'/'situational'/f'possessions_{season}.parquet'
    audit_path = cache.with_suffix('.json')
    fingerprints = {source.name: hashlib.sha256(source.read_bytes()).hexdigest()}
    if cache.exists() and audit_path.exists():
        audit = json.loads(audit_path.read_text())
        if audit.get('live_normalization_version') == VERSION and audit.get('source_fingerprints') == fingerprints:
            return pd.read_parquet(cache), audit
    raw = pd.read_csv(source, low_memory=False)
    rows, excluded = [], {}
    for game, frame in raw.groupby('gameId', sort=False):
        try:
            rows.extend(normalize_live_game(frame, season))
        except (ValueError, KeyError, AttributeError, TypeError, IndexError, RuntimeError) as exc:
            excluded[int(game)] = f'{type(exc).__name__}: {str(exc)[:220]}'
    p = pd.DataFrame(rows, columns=COLUMNS)
    if p.empty:
        raise ValueError('No CDN games passed reconciliation')
    audit = {'archive_rows':len(raw), 'unique_possessions':len(p), 'games_in_archive':raw.gameId.nunique(),
        'quarantined_games':sorted(excluded), 'quarantine_reasons':excluded, 'event_only_games':[],
        'games_analyzed':p.game_id.nunique(), 'possessions_analyzed':len(p),
        'opponent_free_throws':int(p.other_points.sum()), 'date_min':raw.timeActual.min(),
        'date_max':raw.timeActual.max(), 'team_game_counts':p.groupby('offense').game_id.nunique().to_dict(),
        'score_reconciliation':'CDN classified per-team scoring and every cumulative scoreboard total (same source)',
        'live_normalization_version':VERSION, 'pbpstats_version':'1.3.11', 'source_fingerprints':fingerprints}
    cache.parent.mkdir(parents=True, exist_ok=True)
    p.to_parquet(cache, index=False)
    audit_path.write_text(json.dumps(audit, indent=2))
    return p, audit
