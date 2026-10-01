import numpy as np
import pandas as pd
import pytest
from backend.analysis.playoff_data import game_scores, strength, playoff_series
from backend.analysis.playoff_models import (Logistic,MODELS,SIGNALS,fit,chronological,
    bracket_expectations,join_series,team_outcomes)
from backend.analysis.possessions import normalize_possessions


def bracket(season=2020):
    east=['ATL','BOS','CHI','CLE','DET','IND','MIA','MIL']
    west=['DEN','GSW','HOU','LAC','LAL','MEM','MIN','NOP']
    rows=[]
    pairs=[(r,i+offset,a,b) for t,offset in [(east,0),(west,4)]
           for r,i,a,b in [(1,i,t[i],t[7-i]) for i in range(4)]]
    pairs += [(2,0,east[0],east[3]),(2,1,east[1],east[2]),
              (2,2,west[0],west[3]),(2,3,west[1],west[2]),
              (3,0,east[0],east[1]),(3,1,west[0],west[1]),(4,0,east[0],west[0])]
    for round_,slot,a,b in pairs:
        for game in range(1,5):
            home,away=(a,b) if game<=2 else (b,a)
            rows.append(dict(season=season,game_id=f'004{str(season-1)[-2:]}00{round_}{slot}{game}',
                home=home,away=away,home_score=110 if home==a else 100,away_score=110 if away==a else 100,date=None))
    return pd.DataFrame(rows)


def entrants(season=2020):
    _,seeds=playoff_series(bracket(season))
    for field in ['win_pct','point_diff','srs','last20_point_diff',*SIGNALS]:
        seeds[field]=.5 if field=='win_pct' else 0.0
    seeds['wins']=41;seeds['games']=82;seeds['possession_game_coverage']=1.0
    return seeds


def test_actual_seed_bracket_and_incomplete_series_rejection():
    series,seeds=playoff_series(bracket())
    assert len(series)==15 and seeds.set_index('team').loc['BOS','seed']==2
    assert seeds.set_index('team').loc['MIL','seed']==8
    assert series['round'].value_counts().to_dict()=={1:8,2:4,3:2,4:1}
    with pytest.raises(ValueError,match='Incomplete'):playoff_series(bracket().iloc[1:])


def test_equal_strength_bracket_conserves_wins_and_championship_probability():
    model=Logistic(['home_advantage'],np.ones(1),np.zeros(1))
    p=bracket_expectations(entrants(),model,2020)
    assert p.expected_series_wins.sum()==pytest.approx(15)
    assert p.title_probability.sum()==pytest.approx(1)
    assert np.allclose(p.expected_series_wins,.9375)
    assert np.allclose(p.title_probability,1/16)


def test_future_outcomes_do_not_change_earlier_predictions_and_symmetry():
    rng=np.random.default_rng(2);rows=[]
    fields=sorted(set(f for fields in MODELS.values() for f in fields))
    for season in [2018,2019,2020,2021]:
        for i in range(12):
            rows.append(dict(season=season,series_id=f'{season}-{i}',round=1,team_a='AAA',team_b='BBB',
                winner='AAA',outcome=i%2,**dict(zip(fields,rng.normal(size=len(fields))))))
    frame=pd.DataFrame(rows);before,coefficients=chronological(frame)
    frame.loc[frame.season.eq(2021),'outcome']=1-frame.loc[frame.season.eq(2021),'outcome']
    after,_=chronological(frame)
    assert np.allclose(before[before.season.eq(2020)].probability_a,after[after.season.eq(2020)].probability_a)
    assert (coefficients.training_last_season<coefficients.season).all()
    model=fit(frame,fields);swapped=frame.copy();swapped[fields]=-swapped[fields]
    assert np.allclose(model.predict(frame)+model.predict(swapped),1)


def test_actual_wins_include_series_excluded_from_predictive_comparison():
    all_series=pd.concat([playoff_series(bracket(s))[0] for s in [2019,2020]],ignore_index=True)
    teams=pd.concat([entrants(s) for s in [2019,2020]],ignore_index=True)
    teams.loc[teams.season.eq(2020)&teams.team.eq('ATL'),'possession_game_coverage']=.7
    paired,excluded=join_series(all_series,teams)
    assert len(excluded)==4
    outcomes=team_outcomes(paired,teams,all_series=all_series)
    assert outcomes.set_index('team').loc['ATL','actual_series_wins']==4
    assert outcomes.actual_series_wins.sum()==15
    assert outcomes.joint_expected_series_wins.isna().all()


def test_completed_scores_remove_duplicates_and_order_late_corrections_by_clock():
    rows=[dict(GAME_ID=22400001,EVENTNUM=1,EVENTMSGTYPE=1,PERIOD=4,PCTIMESTRING='1:00',
        HOMEDESCRIPTION='A 3PT Shot',VISITORDESCRIPTION=None,PLAYER1_TEAM_ABBREVIATION='AAA',SCORE='0 - 3'),
        dict(GAME_ID=22400001,EVENTNUM=2,EVENTMSGTYPE=1,PERIOD=4,PCTIMESTRING='0:30',
        HOMEDESCRIPTION=None,VISITORDESCRIPTION='B Layup',PLAYER1_TEAM_ABBREVIATION='BBB',SCORE='2 - 3'),
        dict(GAME_ID=22400001,EVENTNUM=3,EVENTMSGTYPE=13,PERIOD=4,PCTIMESTRING='0:00',
        HOMEDESCRIPTION=None,VISITORDESCRIPTION=None,PLAYER1_TEAM_ABBREVIATION=None,SCORE='2 - 3'),
        dict(GAME_ID=22400001,EVENTNUM=4,EVENTMSGTYPE=18,PERIOD=2,PCTIMESTRING='1:00',
        HOMEDESCRIPTION=None,VISITORDESCRIPTION=None,PLAYER1_TEAM_ABBREVIATION=None,SCORE='2 - 2')]
    scores,audit=game_scores(pd.DataFrame(rows+[rows[0]]),'nbastats',2025,False)
    assert len(scores)==1 and scores.home.iloc[0]=='AAA' and not audit['event_scoring_flags']
    assert scores.home_score.iloc[0]==3 and scores.away_score.iloc[0]==2


def test_schedule_adjustment_recovers_latent_strength():
    rows=[]
    ratings={'AAA':8,'BBB':3,'CCC':-11}
    for i,(a,b) in enumerate([('AAA','BBB')]*8+[('BBB','CCC')]*16+[('AAA','CCC')]):
        margin=ratings[a]-ratings[b]
        rows.append(dict(season=2025,game_id=str(i),date='2025-01-01',home=a,away=b,home_score=100+margin,away_score=100))
    result=strength(pd.DataFrame(rows)).set_index('team')
    assert result.loc['AAA','srs']==pytest.approx(8)
    assert result.loc['BBB','srs']==pytest.approx(3)
    assert result.loc['CCC','srs']==pytest.approx(-11)
    assert result.loc['AAA','point_diff'] < result.loc['BBB','point_diff']


def test_historical_ft_annotations_recover_only_reconciled_game():
    records=[]
    for opp,fg,desc in [('B',1,'Player Jr. Free Throw 1 of 2 (1 PTS)'),('A',0,'Turnover')]:
        records.append(dict(GAMEID=22400001,PERIOD=1,OPPONENT=opp,STARTTIME='12:00',ENDTIME='11:40',
            STARTSCOREDIFFERENTIAL=0,FG2M=fg,FG3M=0,EVENTS=desc,GAMEDATE='2024-10-22',DESCRIPTION=desc,URL=None))
    events=pd.DataFrame(dict(GAME_ID=[22400001]*3,PERIOD=[1]*3,EVENTNUM=[1,2,3],EVENTMSGTYPE=[1,3,13],
        PCTIMESTRING=['11:50','11:40','0:00'],HOMEDESCRIPTION=['Player Shot (2 PTS)','Player Free Throw 1 of 1 (3 PTS)',None],
        VISITORDESCRIPTION=[None,None,None],PLAYER1_TEAM_ABBREVIATION=['A','A',None],SCORE=['0 - 2','0 - 3','0 - 3']))
    p,audit=normalize_possessions(pd.DataFrame(records),events,robust_ft=True)
    assert audit['games_analyzed']==1 and p.points.sum()==3
    events.loc[2,'SCORE']='0 - 4'
    p,audit=normalize_possessions(pd.DataFrame(records),events,robust_ft=True)
    assert p.empty and audit['quarantined_games']==[22400001]
