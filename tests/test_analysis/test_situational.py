import numpy as np
import pandas as pd
import pytest
from backend.analysis.situational import team_perspectives, add_expectations, analyze
from backend.analysis.possessions import normalize_possessions
from backend.metrics.game_state import elapsed_seconds, season_from_game_id
from backend.pipeline.transformer import parse_score_margin, transform_pbp_season, classify_events
from backend.db.queries import season_filter_sql, LEGACY_SEASONS


def fixture_possessions():
    # Four teams, same game-state exposure. A scores two points per possession;
    # other teams one. Every game provides paired offensive and defensive rows.
    rows = []
    for game in range(24):
        for off,deff in [('A','B'),('B','A'),('C','D'),('D','C')]:
            for margin in [-15,0,15]:
                rows.append(dict(game_id=game,season=2025,possession_id=len(rows),offense=off,
                    defense=deff,period=4,seconds_remaining=180,regulation_remaining=180,
                    margin=margin,points=2 if off=='A' else 1,other_points=0))
    return pd.DataFrame(rows)


def test_signed_perspectives_and_counts():
    p=fixture_possessions();v=team_perspectives(p)
    assert v.off_n.sum()==len(p)==v.def_n.sum()
    assert v.off_pts.sum()==v.def_pts.sum()==p.points.sum()
    offense=v[v.off_n.eq(1)].reset_index(drop=True)
    defense=v[v.def_n.eq(1)].reset_index(drop=True)
    assert (offense.margin==-defense.margin).all()
    assert offense.loc[offense.margin.eq(15),'margin_bucket'].eq('Ahead 11–20').all()


def test_league_excludes_focal_team_and_weights_possessions():
    v=add_expectations(team_perspectives(fixture_possessions()))
    a=v[v.team.eq('A')&v.off_n.eq(1)]
    assert np.allclose(a.off_expected,1)
    assert np.allclose(a.off_resid,1)
    assert np.allclose(a.def_resid,0)


def test_bootstrap_constant_state_gap_zero_and_positive_defense():
    c,p,g=analyze(fixture_possessions(),draws=100)
    a=p[p.team.eq('A')&p.profile.eq('Overall')].iloc[0]
    assert a.relative_net==pytest.approx(100 + 100/3)
    assert a.off_rating==200 and a.def_rating==100
    gap=g[g.team.eq('A')&g.contrast.eq('Ahead vs behind')].iloc[0]
    assert gap.gap==pytest.approx(0)
    assert gap.ci_low==pytest.approx(0)


def test_zero_exposure_never_infinite_rating():
    p=fixture_possessions();p['margin']=15
    c,profiles,g=analyze(p,draws=100)
    assert not np.isinf(c.select_dtypes(include='number').to_numpy()).any()


def test_clock_and_season_conventions():
    assert elapsed_seconds(pd.Series([1,4,5,6]),pd.Series([720,0,300,0])).tolist()==[0,2880,2880,3480]
    assert season_from_game_id(pd.Series([22400001,'0022300001'])).tolist()==[2025,2024]
    with pytest.raises(ValueError):season_from_game_id(pd.Series(['0042400001']))


def test_game_score_fill_and_pre_event_state():
    df=pd.DataFrame({'GAME_ID':[1,1,1,2,2],'EVENTNUM':[1,2,3,1,2],
        'PERIOD':[1]*5,'PCTIMESTRING':['12:00','11:40','11:30','12:00','11:45'],
        'SCOREMARGIN':[None,'-3',None,None,'2'],'EVENTMSGTYPE':[12,1,2,12,1]})
    out=transform_pbp_season(df)
    assert out.HOME_SCORE_MARGIN_BEFORE.tolist()==[0,0,-3,0,0]
    assert parse_score_margin(df.SCOREMARGIN,df.GAME_ID,absolute=False).tolist()==[0,-3,-3,0,2]


def test_rebound_cumulative_text_does_not_mean_offensive():
    df=pd.DataFrame({'GAME_ID':[1]*4,'EVENTMSGTYPE':[2,4,2,4],
        'PLAYER1_TEAM_ID':[10,20,10,10],
        'HOMEDESCRIPTION':['MISS shot','REBOUND (Off:2 Def:5)','MISS shot','REBOUND (Off:3 Def:5)']})
    out=classify_events(df)
    assert out.is_oreb.tolist()==[0,0,0,1]
    assert out.is_dreb.tolist()==[0,1,0,0]


def test_sql_seasons_support_both_id_formats():
    import duckdb
    con=duckdb.connect()
    con.execute("CREATE TABLE legacy_shots AS SELECT * FROM (VALUES ('0022400001'),('22400002'),('0022300240')) t(GAME_ID)")
    assert con.execute('SELECT COUNT(*) FROM legacy_shots WHERE 1=1 '+season_filter_sql(2025)).fetchone()[0]==2
    assert con.execute(LEGACY_SEASONS).fetchall()==[(2024,1),(2025,2)]
    con.close()


def test_archive_event_repetition_and_opponent_technical_ft():
    records=[]
    for opp,margin,pts,desc in [('B',0,1,'Player Free Throw Technical (1 PTS)'),('A',-2,0,'Turnover')]:
        records.append(dict(GAMEID=22400001,PERIOD=1,OPPONENT=opp,STARTTIME='12:00',ENDTIME='11:40',
            STARTSCOREDIFFERENTIAL=margin,FG2M=pts,FG3M=0,EVENTS=desc,GAMEDATE='2024-10-22',DESCRIPTION=desc,URL=None))
    archive=pd.DataFrame(records+[records[0]])
    events=pd.DataFrame(dict(GAME_ID=[22400001]*3,PERIOD=[1]*3,EVENTNUM=[1,2,3],EVENTMSGTYPE=[1,3,13],
        PCTIMESTRING=['11:50','11:40','0:00'],HOMEDESCRIPTION=['Player Shot (2 PTS)',None,None],
        VISITORDESCRIPTION=[None,'Player Free Throw Technical (1 PTS)',None],
        PLAYER1_TEAM_ABBREVIATION=['A','B',None],SCORE=['0 - 2','1 - 2','1 - 2']))
    p,a=normalize_possessions(archive,events)
    assert len(p)==2 and a['games_analyzed']==1
    assert p.points.sum()==2 and p.other_points.sum()==1
    v=team_perspectives(p)
    assert v.groupby('team').off_pts.sum().to_dict()=={'A':2,'B':1}
