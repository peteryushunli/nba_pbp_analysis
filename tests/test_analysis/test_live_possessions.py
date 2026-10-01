import pandas as pd
import pytest
from backend.analysis.live_possessions import normalize_live_game
from backend.analysis.download import source_kinds


def game():
    rows = []
    def event(kind, subtype, clock, possession, team=None, score=0, **extra):
        rows.append(dict(gameId=22500001, actionNumber=len(rows)+1, orderNumber=len(rows)+1,
            period=1, clock=clock, actionType=kind, subType=subtype, possession=possession,
            teamId=team, teamTricode={10:'AAA',20:'BBB'}.get(team), scoreHome=score,
            scoreAway=0, personId=1, description=kind, **extra))
    event('substitution','in','PT12M00.00S',20,20)
    event('period','start','PT12M00.00S',10)
    event('2pt','Jump Shot','PT11M40.00S',10,10,2,shotResult='Made')
    event('turnover','bad pass','PT11M20.00S',20,20,2)
    event('freethrow','1 of 1','PT11M00.00S',10,20,2,shotResult='Made',descriptor='technical')
    rows[-1]['scoreAway'] = 1
    event('2pt','Jump Shot','PT10M40.00S',10,10,4,shotResult='Made')
    rows[-1]['scoreAway'] = 1
    event('period','end','PT00M00.00S',20,score=4)
    rows[-1]['scoreAway'] = 1
    event('game','end','PT00M00.00S',0,score=4)
    rows[-1]['scoreAway'] = 1
    return pd.DataFrame(rows)


def test_period_administrative_rows_and_defensive_technical_ft():
    p=normalize_live_game(game(),2026)
    assert [r['offense'] for r in p] == ['AAA','BBB','AAA','BBB']
    assert [r['margin'] for r in p] == [0,-2,2,-3]
    assert [r['seconds_remaining'] for r in p] == [720,700,680,640]
    assert sum(r['points'] for r in p)==4
    assert sum(r['other_points'] for r in p)==1


def test_mismatched_score_and_unfinished_game_rejected():
    g=game();g.loc[2,'scoreHome']=3
    with pytest.raises(ValueError,match='scoreboard'):normalize_live_game(g,2026)
    with pytest.raises(ValueError,match='final marker'):normalize_live_game(game().iloc[:-1],2026)


def test_source_format_uses_ending_year():
    assert source_kinds(2025)==['pbpstats','nbastats']
    assert source_kinds(2026)==['cdnnba']
    with pytest.raises(ValueError,match='Game ID'):normalize_live_game(game(),2025)


def test_offensive_rebound_stays_in_same_possession():
    g=game()
    miss=g.iloc[2].to_dict()
    miss.update(actionNumber=100,orderNumber=2.1,clock='PT11M50.00S',shotResult='Missed',scoreHome=0)
    rebound=miss.copy()
    rebound.update(actionNumber=101,orderNumber=2.2,actionType='rebound',subType='offensive',shotActionNumber=100)
    rebound.pop('shotResult')
    p=normalize_live_game(pd.concat([g,pd.DataFrame([miss,rebound])],ignore_index=True),2026)
    assert len(p)==4
    assert p[0]['points']==2 and p[0]['seconds_remaining']==720


def test_overtime_uses_five_minute_period_and_zero_regulation_remaining():
    g=game();g['period']=5
    g['clock']=['PT05M00.00S','PT05M00.00S','PT04M40.00S','PT04M20.00S',
                'PT04M00.00S','PT03M40.00S','PT00M00.00S','PT00M00.00S']
    p=normalize_live_game(g,2026)
    assert p[0]['seconds_remaining']==300
    assert all(r['regulation_remaining']==0 for r in p)
