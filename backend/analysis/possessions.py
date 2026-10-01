"""Normalize the public pbpstats archive and reconcile against NBA event scores.

Archive rows repeat a possession once per event. Never count CSV rows as possessions.
"""
from pathlib import Path
import json
import hashlib
import numpy as np
import pandas as pd
from backend.metrics.game_state import clock_seconds, season_from_game_id
from backend.analysis.scores import final_scores

def ft_identity(s: pd.Series) -> pd.Series:
    s=s.str.replace(r' Free Throw.*',' Free Throw',regex=True).str.normalize('NFKD').str.encode('ascii',errors='ignore').str.decode('ascii')
    s=s.str.replace(r'^[A-Z]\.\s*','',regex=True)
    return s.str.replace(r'\s+(?:Jr\.?|Sr\.?|II|III|IV) Free Throw',' Free Throw',regex=True)


def normalize_possessions(archive: pd.DataFrame, events: pd.DataFrame, robust_ft: bool = False) -> tuple[pd.DataFrame, dict]:
    keys = [c for c in archive.columns if c not in ['DESCRIPTION', 'URL']]
    archive = archive.copy()
    archive['possession_id'] = archive.groupby(keys, dropna=False, sort=False).ngroup()
    p = archive.drop_duplicates('possession_id').copy()
    teams = p.groupby('GAMEID').OPPONENT.unique()
    if not teams.map(len).eq(2).all():
        raise ValueError('Every game must have exactly two teams')
    opponents = teams.to_dict()
    p['offense'] = [next(t for t in opponents[g] if t != d)
                    for g, d in zip(p.GAMEID, p.OPPONENT)]
    p = p.rename(columns={'OPPONENT': 'defense', 'GAMEID': 'game_id',
                          'PERIOD': 'period', 'STARTSCOREDIFFERENTIAL': 'margin'})
    p['season'] = season_from_game_id(p.game_id)
    p['seconds_remaining'] = clock_seconds(p.STARTTIME)
    limit = np.where(p.period <= 4, 720, 300)
    if not p.seconds_remaining.between(0, limit).all():
        raise ValueError('Invalid possession start clock')
    p['regulation_remaining'] = np.where(p.period <= 4,
        (4 - p.period) * 720 + p.seconds_remaining, 0)
    p['points'] = 2 * p.FG2M + 3 * p.FG3M
    p['other_points'] = 0

    # Made FT descriptions contain the shooter's cumulative points, providing
    # a unique game/period/event key even when the archive has no video URL.
    e = events.drop_duplicates().copy() if robust_ft else events.copy()
    e['description'] = (e.HOMEDESCRIPTION.fillna('') + e.VISITORDESCRIPTION.fillna('')).str.strip()
    ft = e[e.EVENTMSGTYPE.eq(3) & ~e.description.str.contains('MISS')].copy()
    if robust_ft:
        # Historical NBA logs occasionally reverse FT sequence numbers or
        # mislabel cumulative player points. Match shooter + period + clock,
        # verify the made count, then reconcile the whole game independently.
        ft['description'] = ft_identity(ft.description)
    lookup_keys = ['GAME_ID', 'PERIOD', 'description']
    ft['ft_seconds'] = clock_seconds(ft.PCTIMESTRING)
    a = archive[archive.DESCRIPTION.fillna('').str.contains('Free Throw') &
                ~archive.DESCRIPTION.fillna('').str.contains('MISS')]
    a = a.drop_duplicates(['possession_id', 'DESCRIPTION'])
    if robust_ft:
        a = a.copy()
        a['technical']=a.DESCRIPTION.str.contains('Technical',case=False)
        a['expected_team']=a.possession_id.map(p.set_index('possession_id').offense)
        a['DESCRIPTION'] = ft_identity(a.DESCRIPTION)
        expected_ft = a.groupby('possession_id').size()
        a = a.drop_duplicates(['possession_id','DESCRIPTION'])
        ft = ft.drop_duplicates(['GAME_ID','EVENTNUM'])
    matched = a.merge(ft[lookup_keys + ['PLAYER1_TEAM_ABBREVIATION', 'ft_seconds', 'EVENTNUM']],
        left_on=['GAMEID', 'PERIOD', 'DESCRIPTION'], right_on=lookup_keys,
        how='left', validate='many_to_many')
    matched = matched[matched.ft_seconds.between(clock_seconds(matched.ENDTIME),
                                                 clock_seconds(matched.STARTTIME))]
    if robust_ft:
        matched=matched[matched.technical | matched.PLAYER1_TEAM_ABBREVIATION.eq(matched.expected_team)]
    match_keys = ['possession_id', 'DESCRIPTION']
    missing = a.merge(matched[match_keys].drop_duplicates(), on=match_keys, how='left', indicator=True)
    invalid_games = set(missing.loc[missing._merge.eq('left_only'), 'GAMEID'])
    if robust_ft:
        invalid_games.update(matched.loc[matched.duplicated(['GAME_ID','EVENTNUM'],keep=False),'GAMEID'])
        counts = matched.groupby('possession_id').size().reindex(expected_ft.index,fill_value=0)
        invalid_games.update(a.loc[a.possession_id.isin(counts[counts.ne(expected_ft)].index),'GAMEID'])
    else:
        invalid_games.update(matched.loc[matched.duplicated(match_keys, keep=False), 'GAMEID'])
    matched = matched[~matched.GAMEID.isin(invalid_games)]
    matched = matched.merge(p[['possession_id', 'offense']], on='possession_id', validate='many_to_one')
    own = matched.PLAYER1_TEAM_ABBREVIATION.eq(matched.offense)
    own_count = matched[own].groupby('possession_id').size()
    other_count = matched[~own].groupby('possession_id').size()
    p['points'] += p.possession_id.map(own_count).fillna(0).astype(int)
    p['other_points'] = p.possession_id.map(other_count).fillna(0).astype(int)

    # Validate totals independently using BOTH classified NBA scoring events
    # and final cumulative score. Quarantine whole games with any discrepancy.
    e['pts'] = np.select([e.EVENTMSGTYPE.eq(1), e.EVENTMSGTYPE.eq(3) &
                         ~e.description.str.contains('MISS')],
                         [2 + e.description.str.contains('3PT').astype(int), 1], default=0)
    nba = e[e.pts.gt(0)].groupby(['GAME_ID', 'PLAYER1_TEAM_ABBREVIATION']).pts.sum()
    scored = pd.concat([
        p[['game_id', 'offense', 'points']].rename(columns={'offense':'team'}),
        p[['game_id', 'defense', 'other_points']].rename(columns={'defense':'team','other_points':'points'})
    ]).groupby(['game_id', 'team']).points.sum()
    nba.index = nba.index.set_names(['game_id', 'team'])
    diff = scored.subtract(nba, fill_value=0)
    bad = set(diff[diff.ne(0)].index.get_level_values(0)) | invalid_games
    final = final_scores(e)
    final_sum = final.str.split(' - ', expand=True).astype(int).sum(axis=1)
    raw_sum = e.groupby('GAME_ID').pts.sum()
    bad.update(raw_sum[raw_sum.ne(final_sum)].index)
    archive_games = set(p.game_id)
    audit = {'archive_rows': len(archive), 'unique_possessions': len(p),
             'games_in_archive': p.game_id.nunique(), 'quarantined_games': sorted(int(g) for g in bad & archive_games),
             'event_only_games': sorted(int(g) for g in set(e.GAME_ID) - archive_games),
             'opponent_free_throws': int(p.other_points.sum()),
             'unmatched_or_ambiguous_ft_games': sorted(int(g) for g in invalid_games),
             'date_min': str(p.GAMEDATE.min()), 'date_max': str(p.GAMEDATE.max()),
             'score_reconciliation': 'per-team event points and final cumulative game scores'}
    p = p[~p.game_id.isin(bad)].copy()
    audit['games_analyzed'] = p.game_id.nunique()
    audit['possessions_analyzed'] = len(p)
    audit['team_game_counts'] = p.groupby('offense').game_id.nunique().to_dict()
    return p[['game_id', 'season', 'possession_id', 'offense', 'defense', 'period',
              'seconds_remaining', 'regulation_remaining', 'margin', 'points', 'other_points']], audit


def load_season(season: int, data_dir: Path, robust_ft: bool = False) -> tuple[pd.DataFrame, dict]:
    """Season ending year, e.g. 2025 means source archive pbpstats_2024."""
    if season >= 2026:
        from backend.analysis.live_possessions import load_live_season
        return load_live_season(season, data_dir)
    suffix = '_canonical' if robust_ft else ''
    version = 6 if robust_ft else 3
    cache = data_dir / 'processed' / 'situational' / f'possessions_{season}{suffix}.parquet'
    audit_path = cache.with_suffix('.json')
    start = season - 1
    sources = [data_dir / 'external' / kind / f'{kind}_{start}.csv' for kind in ['pbpstats','nbastats']]
    fingerprints = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    if cache.exists() and audit_path.exists():
        audit = json.loads(audit_path.read_text())
        if audit.get('normalization_version') == version and audit.get('source_fingerprints') == fingerprints:
            return pd.read_parquet(cache), audit
    archive = pd.read_csv(data_dir / 'external' / 'pbpstats' / f'pbpstats_{start}.csv', low_memory=False)
    events = pd.read_csv(data_dir / 'external' / 'nbastats' / f'nbastats_{start}.csv', low_memory=False)
    if robust_ft:
        # Keep every fully reconciled exact match; use tolerant annotation
        # matching only as a fallback, so it cannot reduce valid coverage.
        p,audit=normalize_possessions(archive,events.drop_duplicates())
        excluded=set(audit['quarantined_games'])
        if excluded:
            extra,extra_audit=normalize_possessions(archive[archive.GAMEID.isin(excluded)],
                events[events.GAME_ID.isin(excluded)],robust_ft=True)
            p=pd.concat([p,extra],ignore_index=True)
            audit['canonical_recovered_games']=sorted(int(g) for g in extra.game_id.unique())
            accepted=set(p.game_id)
            audit['quarantined_games']=sorted(excluded-accepted)
            audit['initial_unmatched_ft_games']=audit['unmatched_or_ambiguous_ft_games']
            ft_bad=set(audit['initial_unmatched_ft_games'])|set(extra_audit['unmatched_or_ambiguous_ft_games'])
            audit['unmatched_or_ambiguous_ft_games']=sorted(ft_bad & set(audit['quarantined_games']))
            audit['games_analyzed']=p.game_id.nunique();audit['possessions_analyzed']=len(p)
            audit['team_game_counts']=p.groupby('offense').game_id.nunique().to_dict()
            audit['opponent_free_throws']=int(p.other_points.sum())
    else:
        p, audit = normalize_possessions(archive, events)
    if set(p.season) != {season}:
        raise ValueError('Game IDs disagree with requested season')
    audit['normalization_version'] = version
    audit['ft_matching'] = 'shooter, period, clock interval and reconciled made counts' if robust_ft else 'exact event description'
    audit['source_fingerprints'] = fingerprints
    cache.parent.mkdir(parents=True, exist_ok=True)
    p.to_parquet(cache, index=False)
    audit_path.write_text(json.dumps(audit, indent=2))
    return p, audit
