import pandas as pd
from nba_data_functions import shot_detail_time_elapsed
from nba_shot_viz import clean_pbp_data, create_buckets, aggregate_data


def test_legacy_notebook_helpers_copy_inputs_and_aggregate_only_counts():
    source=pd.DataFrame({'GAME_ID':[22400001],'GAME_DATE':[20241022],'PLAYER_ID':[1],
        'PLAYER_NAME':['Example'],'TEAM_NAME':['Example team'],'PERIOD':[5],
        'MINUTES_REMAINING':[5],'SECONDS_REMAINING':[0],'ABS_SCORE_DIFF':[3],
        'SHOT_ATTEMPTED_FLAG':[1],'SHOT_MADE_FLAG':[1],'3PT_ATTEMPTED_FLAG':[1]})
    timed=shot_detail_time_elapsed(source)
    assert timed.TIME_ELAPSED.iloc[0]==2880 and 'TIME_ELAPSED' not in source
    cleaned=clean_pbp_data(timed);bucketed=create_buckets(cleaned)
    assert 'FGM' not in timed and 'ABS_SCORE_DIFF_BUCKETS' not in cleaned
    aggregate=aggregate_data(bucketed)
    assert aggregate.FGA.sum()==1 and aggregate.EFG.max()==1.5
