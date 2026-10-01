import json
import duckdb
from fastapi.testclient import TestClient
from backend.main import app
from backend.routers import efg, team_situations


def test_team_analysis_round_trip_and_missing_report(tmp_path, monkeypatch):
    monkeypatch.setattr(team_situations.settings,'PROJECT_ROOT',tmp_path)
    client=TestClient(app)
    assert client.get('/api/team-situations').status_code==404
    (tmp_path/'reports').mkdir()
    payload={'cells':[],'profiles':[],'contrasts':[]}
    (tmp_path/'reports'/'team_situations.json').write_text(json.dumps(payload))
    assert client.get('/api/team-situations').json()==payload


def test_efg_player_apostrophe_and_untrusted_name(monkeypatch):
    conn=duckdb.connect()
    conn.execute('''CREATE TABLE legacy_shots_bucketed (
        GAME_ID BIGINT, PLAYER_NAME VARCHAR, TEAM_NAME VARCHAR,
        TIME_BUCKET VARCHAR, SCORE_BUCKET VARCHAR, SHOT_MADE_FLAG INT,
        SHOT_ATTEMPTED_FLAG INT, "3PT_ATTEMPTED_FLAG" INT)''')
    conn.execute('INSERT INTO legacy_shots_bucketed VALUES (?,?,?,?,?,?,?,?)',
        [22400001,"D'Angelo Russell",'Los Angeles Lakers','4-0','0-5',1,1,1])
    monkeypatch.setattr(efg,'get_connection',lambda:conn)
    client=TestClient(app)
    response=client.get('/api/efg/heatmap',params={'season':2025,'player_name':"D'Angelo Russell"})
    assert response.status_code==200
    assert response.json()['cells'][0]['fga']==1
    response=client.get('/api/efg/heatmap',params={'season':2025,'player_name':"' OR 1=1 --"})
    assert response.status_code==200 and response.json()['cells']==[]
    conn.close()
