"""Audited game scores, schedule-adjusted strength and the actual playoff bracket."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from backend.analysis.scores import final_scores

EAST = {'ATL','BOS','BKN','CHA','CHI','CLE','DET','IND','MIA','MIL','NYK','ORL','PHI','TOR','WAS'}
GAME_VERSION = 4


def game_scores(events: pd.DataFrame, provider: str, season: int, postseason: bool) -> tuple[pd.DataFrame, dict]:
    """Infer home team by independently scored team totals and the final scoreboard.

    Final NBA games cannot be tied, so the per-team total matching the final home
    score uniquely identifies the home team. Legacy made-shot home/visitor roles
    independently confirm identity and allow use of a completed cumulative score
    when event tallies have annotation errors (recorded in the audit).
    """
    events=events.drop_duplicates()
    if provider == 'cdnnba':
        e = events.sort_values(['gameId','orderNumber']).copy()
        e['points'] = e.actionType.map({'2pt':2,'3pt':3,'freethrow':1}).fillna(0).where(e.shotResult.eq('Made'),0)
        e = e.rename(columns={'gameId':'game_id','teamTricode':'team'})
        final = e.groupby('game_id')[['scoreHome','scoreAway']].last().rename(columns={'scoreHome':'home_score','scoreAway':'away_score'})
        ended = set(e.loc[e.actionType.eq('game') & e.subType.eq('end'),'game_id'])
        dates = e.groupby('game_id').timeActual.min().str[:10]
    else:
        e = events.sort_values(['GAME_ID','EVENTNUM']).copy()
        description = e.HOMEDESCRIPTION.fillna('') + e.VISITORDESCRIPTION.fillna('')
        e['points'] = np.select([e.EVENTMSGTYPE.eq(1),e.EVENTMSGTYPE.eq(3)&~description.str.contains('MISS')],
                              [2+description.str.contains('3PT').astype(int),1],default=0)
        e = e.rename(columns={'GAME_ID':'game_id','PLAYER1_TEAM_ABBREVIATION':'team'})
        scores = final_scores(events).str.split(' - ',expand=True)
        final = pd.DataFrame({'home_score':pd.to_numeric(scores[1]),'away_score':pd.to_numeric(scores[0])})
        ended = set(e.loc[e.EVENTMSGTYPE.eq(13)&e.PERIOD.ge(4)&e.PCTIMESTRING.eq('0:00'),'game_id'])
        dates = pd.Series(dtype=str)
        made=e[e.EVENTMSGTYPE.eq(1)]
        home_ids=made[made.HOMEDESCRIPTION.notna()].groupby('game_id').team.unique()
        away_ids=made[made.VISITORDESCRIPTION.notna()].groupby('game_id').team.unique()
    totals = e[e.points.gt(0)].groupby(['game_id','team']).points.sum()
    rows, excluded, scoring_flags = [], {}, {}
    prefix = f'{"004" if postseason else "002"}{str(season-1)[-2:]}'
    for game, score in final.iterrows():
        gid = str(int(game)).zfill(10)
        if not gid.startswith(prefix):
            continue  # play-in/NBA Cup final are separate competitions
        try:
            if game not in ended:
                raise ValueError('No completed-game marker')
            t = totals.loc[game]
            if len(t)!=2 or score.home_score==score.away_score:
                raise ValueError('Invalid teams or tied final score')
            home = t[t.eq(score.home_score)]
            away = t[t.eq(score.away_score)]
            if len(home)!=1 or len(away)!=1:
                if provider=='cdnnba' or len(home_ids.get(game,[]))!=1 or len(away_ids.get(game,[]))!=1:
                    raise ValueError('Ambiguous team attribution')
                home_team=home_ids[game][0];away_team=away_ids[game][0]
                if home_team==away_team or set(t.index)!={home_team,away_team}:
                    raise ValueError('Team attribution disagrees across sources within log')
                scoring_flags[gid]='Event scoring disagrees; use completed cumulative score with independently identified home/away teams'
            else:
                home_team=home.index[0];away_team=away.index[0]
                if provider!='cdnnba' and (len(home_ids.get(game,[]))!=1 or home_ids[game][0]!=home_team
                                         or len(away_ids.get(game,[]))!=1 or away_ids[game][0]!=away_team):
                    raise ValueError('Event roles disagree with score-derived home/away teams')
            rows.append(dict(season=season,game_id=gid,home=home_team,away=away_team,
                home_score=int(score.home_score),away_score=int(score.away_score),date=dates.get(game,None)))
        except (ValueError,KeyError) as exc:
            excluded[gid]=str(exc)
    present = {str(int(x)).zfill(10) for x in e.game_id.unique() if str(int(x)).zfill(10).startswith(prefix)}
    for gid in present - set(final.index.map(lambda x:str(int(x)).zfill(10))):
        excluded[gid]='Missing cumulative score'
    result = pd.DataFrame(rows)
    if result.empty:
        raise ValueError('No completed games passed score checks')
    return result, {'source_games':len(present),'accepted_games':len(result),'excluded':excluded,
                    'event_scoring_flags':scoring_flags,'provider':provider,
                    'score_validation':'completed cumulative scores and unambiguous home/away identity; event tally flags recorded'}


def load_games(season: int, data_dir: Path, postseason: bool = False) -> tuple[pd.DataFrame, dict]:
    kind = 'cdnnba' if season>=2026 else 'nbastats'
    source = data_dir/'external'/kind/f'{kind}_{"po_" if postseason else ""}{season-1}.csv'
    cache = data_dir/'processed'/'playoffs'/f'{"postseason" if postseason else "regular"}_{season}.parquet'
    metadata = cache.with_suffix('.json')
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    supplement = Path(__file__).with_name('metadata')/'playoffs_2026.json'
    supplement_digest = hashlib.sha256(supplement.read_bytes()).hexdigest() if postseason and season==2026 else None
    date_source=data_dir/'external'/'pbpstats'/f'pbpstats_{season-1}.csv' if not postseason and kind=='nbastats' else None
    date_digest=hashlib.sha256(date_source.read_bytes()).hexdigest() if date_source else None
    if metadata.exists() and cache.exists():
        audit=json.loads(metadata.read_text())
        if (audit.get('sha256')==digest and audit.get('game_version')==GAME_VERSION
            and audit.get('supplement_sha256')==supplement_digest and audit.get('date_source_sha256')==date_digest):
            return pd.read_parquet(cache),audit
    fields = (['gameId','orderNumber','actionType','subType','shotResult','teamTricode','scoreHome','scoreAway','timeActual']
              if kind=='cdnnba' else ['GAME_ID','EVENTNUM','EVENTMSGTYPE','PERIOD','PCTIMESTRING',
                  'HOMEDESCRIPTION','VISITORDESCRIPTION','SCORE','PLAYER1_TEAM_ABBREVIATION'])
    games,audit=game_scores(pd.read_csv(source,usecols=fields,low_memory=False),kind,season,postseason)
    if supplement_digest:
        supplement_data=json.loads(supplement.read_text());additions=[]
        existing=games.set_index('game_id')
        for series in supplement_data['series']:
            for number,(winner,win_score,loss_score) in enumerate(series['games'],1):
                home=series['home'] if number in [1,2,5,7] else series['away']
                away=series['away'] if home==series['home'] else series['home']
                row=dict(season=season,game_id=series['id']+str(number),home=home,away=away,
                    home_score=win_score if winner==home else loss_score,
                    away_score=win_score if winner==away else loss_score,date=None)
                if row['game_id'] in existing.index:
                    if any(existing.loc[row['game_id'],k]!=row[k] for k in ['home','away','home_score','away_score']):
                        raise ValueError('Official supplement disagrees with archived game')
                else:
                    additions.append(row)
        games=pd.concat([games,pd.DataFrame(additions)],ignore_index=True)
        audit.update(supplement_source=supplement_data['source'],supplement_games=[r['game_id'] for r in additions],
            supplement_sha256=supplement_digest,accepted_games=len(games))
    if not postseason and kind=='nbastats':
        path=date_source
        dates=pd.read_csv(path,usecols=['GAMEID','GAMEDATE']).drop_duplicates('GAMEID').set_index('GAMEID').GAMEDATE
        games['date']=pd.to_numeric(games.game_id).map(dates).astype('string').str[:10]
        audit['date_source_sha256']=date_digest
    audit.update(sha256=digest,source=str(source),game_version=GAME_VERSION)
    cache.parent.mkdir(parents=True,exist_ok=True)
    games.to_parquet(cache,index=False);metadata.write_text(json.dumps(audit,indent=2))
    return games,audit


def strength(games: pd.DataFrame) -> pd.DataFrame:
    """Wins, point differential and SRS from accepted completed regular games.

    Least squares solves margin = own rating - opponent rating, with mean rating
    constrained to zero. This implements a schedule adjustment, not playoff data.
    """
    if games.season.nunique()!=1:
        raise ValueError('Calculate strength one season at a time')
    teams=sorted(set(games.home)|set(games.away));positions={t:i for i,t in enumerate(teams)}
    x=np.zeros((len(games)+1,len(teams)))
    x[np.arange(len(games)),games.home.map(positions)]=1
    x[np.arange(len(games)),games.away.map(positions)]=-1
    x[-1]=1
    margin=(games.home_score-games.away_score).to_numpy()
    srs=np.linalg.lstsq(x,np.r_[margin,0],rcond=None)[0]
    home=games[['game_id','date','home','away']].rename(columns={'home':'team','away':'opponent'}).assign(margin=margin)
    away=games[['game_id','date','home','away']].rename(columns={'away':'team','home':'opponent'}).assign(margin=-margin)
    v=pd.concat([home,away]);v['win']=v.margin.gt(0)
    out=v.groupby('team').agg(games=('game_id','size'),wins=('win','sum'),point_diff=('margin','mean'))
    out['win_pct']=out.wins/out.games
    out['srs']=pd.Series(srs,index=teams)
    recent=v.dropna(subset=['date']).sort_values(['date','game_id']).groupby('team').tail(20)
    out['last20_point_diff']=recent.groupby('team').margin.mean()
    out['recent_games']=recent.groupby('team').size()
    out['season']=int(games.season.iloc[0]);out['conference']=['E' if t in EAST else 'W' for t in out.index]
    return out.reset_index()


def playoff_series(games: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Recover actual seeds from first-round bracket slots and Game 1 home court.

    NBA first-round series slots are 1/8, 2/7, 3/6, 4/5 in each conference.
    Validate the full 8-4-2-1 bracket and exactly four wins for every winner.
    """
    games=games.copy();games['series_id']=games.game_id.str[:9]
    rows,seeds=[],{}
    for sid,g in games.groupby('series_id',sort=True):
        first=g.sort_values('game_id').iloc[0]
        teams=sorted(set(g.home)|set(g.away))
        if len(teams)!=2:
            raise ValueError('Series has more than two teams')
        a,b=teams
        winner=np.where(g.home_score.gt(g.away_score),g.home,g.away)
        wins=pd.Series(winner).value_counts()
        if wins.max()!=4 or len(g) not in range(4,8) or int(g.game_id.str[-1:].astype(int).min())!=1:
            raise ValueError(f'Incomplete playoff series: {sid}')
        round_=int(sid[7]);slot=int(sid[8])
        if round_==1:
            seed=slot%4+1
            for team,position in [(first.home,seed),(first.away,9-seed)]:
                if team in seeds:
                    raise ValueError('Repeated first-round team')
                seeds[team]=position
        rows.append(dict(season=int(first.season),series_id=sid,round=round_,team_a=a,team_b=b,
            home_court=first.home,win_a=int(wins.get(a,0)),win_b=int(wins.get(b,0)),
            winner=wins.idxmax(),outcome=int(wins.idxmax()==a),games=len(g),
            point_diff_a=float(np.where(g.home.eq(a),g.home_score-g.away_score,g.away_score-g.home_score).mean())))
    series=pd.DataFrame(rows)
    if series.groupby('round').size().to_dict()!={1:8,2:4,3:2,4:1} or len(seeds)!=16:
        raise ValueError('Expected complete 15-series, 16-team playoff bracket')
    seed_frame=pd.DataFrame([dict(season=int(games.season.iloc[0]),team=t,seed=s,
        conference='E' if t in EAST else 'W') for t,s in seeds.items()])
    for conference,g in seed_frame.groupby('conference'):
        if sorted(g.seed)!=list(range(1,9)):
            raise ValueError(f'Invalid {conference} seed assignment')
    for _,row in series[series['round'].lt(4)].iterrows():
        if (row.team_a in EAST)!=(row.team_b in EAST):
            raise ValueError('Cross-conference series before Finals')
        if seeds[row.home_court]!=min(seeds[row.team_a],seeds[row.team_b]):
            raise ValueError('Seed-derived home court disagrees with actual schedule')
    return series,seed_frame
