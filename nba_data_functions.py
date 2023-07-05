import numpy as np
import pandas as pd
import os
import urllib.request
import tarfile
from pathlib import Path
from itertools import product


def pbp_processing(pbp):
    """
    Process the initial play by play data to format the time columns and output a cleaner dataframe
    """
    #Convert ENDTIME into seconds
    pbp['END_TIME2'] = pd.to_datetime(pbp['ENDTIME'], format = '%M:%S')
    pbp['SECONDS_REMAINING'] = pbp['END_TIME2'].dt.second
    pbp['MINUTES_REMAINING'] = pbp['END_TIME2'].dt.minute
    pbp.drop(columns = ['END_TIME2'], inplace = True)
    
    #Create new column for absolute score difference
    pbp['ABS_SCORE_DIFF'] = abs(pbp['STARTSCOREDIFFERENTIAL'])

    #Filter for relevant rows then drop redundant rows with the same time stamps
    output = pbp[['GAMEID', 'PERIOD', 'MINUTES_REMAINING', 'SECONDS_REMAINING', 'ABS_SCORE_DIFF']]
    output =  output.groupby(['GAMEID', 'PERIOD', 'MINUTES_REMAINING', 'SECONDS_REMAINING'], as_index = False).max()
    return output.sort_values(by = ['GAMEID', 'PERIOD', 'MINUTES_REMAINING', 'SECONDS_REMAINING'], ascending = [True, False, True, True])

def process_score_difference(game_df):
    """ 
    Output a new dataframe that contains the score difference at each second of the game.
    1. Calculate the absolute value of "STARTSCOREDIFFERENTIAL" as "ABS_SCORE_DIFF" in the input df
    2. Create a new df that contains the score difference at each second of the game based on the "TIME_REMAINING" column
    """
    if game_df.PERIOD.max() > 4:
        max_game_seconds = 48*60 + (game_df.PERIOD.max() - 4)*5*60
    else: 
        max_game_seconds = 48*60
    
    #Create new empty dataframee with a row for each second of game time
    score_diff = pd.DataFrame()
    score_diff['TIME_ELAPSED'] = range(0, max_game_seconds+1)
    score_diff['GAME_ID'] = game_df.GAMEID.unique()[0]

    #Create column for Time Elapsed in game_df as follows:
    # If PERIOD is betwee 1 - 4: time elapsed = (PERIOD-1)*12*60 + (12*60) - SECONDS_REMAINING - MINUTES_REMAINING*60
    # If PERIOD is greater than 4: time elapsed = 48*60 + (PERIOD-5)*5*60 + (5*60) - SECONDS_REMAINING - MINUTES_REMAINING*60
    game_df['TIME_ELAPSED'] = np.where(game_df['PERIOD'] <= 4, (game_df['PERIOD']-1)*12*60 + (12*60) - game_df['SECONDS_REMAINING'] - game_df['MINUTES_REMAINING']*60, 48*60 + (game_df['PERIOD']-5)*5*60 + (5*60) - game_df['SECONDS_REMAINING'] - game_df['MINUTES_REMAINING']*60)

    #Join the two dataframes on the TIME_ELAPSED column
    score_diff = score_diff.merge(game_df, how = 'left', on = 'TIME_ELAPSED')
    #Set the ABS_SCORE_DIFF at TIME_ELAPSED = 0 to be 0
    score_diff.loc[score_diff['TIME_ELAPSED'] == 0, 'ABS_SCORE_DIFF'] = 0
    #Fill rows with a missing ABS_SCORE_DIFF with the last known value backwards and then forwards
    score_diff['ABS_SCORE_DIFF'] = score_diff['ABS_SCORE_DIFF'].fillna(method = 'bfill')
    score_diff['ABS_SCORE_DIFF'] = score_diff['ABS_SCORE_DIFF'].fillna(method = 'ffill')

    return score_diff[['GAME_ID', 'TIME_ELAPSED', 'ABS_SCORE_DIFF']]

def shot_detail_time_elapsed(shot_detail):
    #Split data by regulation and overtime
    regulation = shot_detail.loc[shot_detail['PERIOD'] <= 4]
    overtime = shot_detail.loc[shot_detail['PERIOD'] > 4]

    #Calculate time elapsed for every row
    regulation['TIME_ELAPSED'] = (regulation['PERIOD'] - 1)*12*60 + (12*60 - regulation['MINUTES_REMAINING']*60 - regulation['SECONDS_REMAINING'])
    overtime['TIME_ELAPSED'] = (48*60 + (overtime['PERIOD'] - 5)*5*60 + (5*60 - overtime['MINUTES_REMAINING']*60 - overtime['SECONDS_REMAINING']))

    #Combine regulation and overtime
    shot_detail = pd.concat([regulation, overtime])
    return shot_detail

def pbp_game_processing(pbp):
    """
    This function will parse through every game in the pbp dataset (based on unique GAMEID) and then run the subsequent pbp_processing function on each game.
    """
    pbp_list = []
    #Parse through each game using a for loop
    for game in pbp.GAMEID.unique():
        pbp_game = pbp.loc[pbp['GAMEID'] == game]
        #Run the pbp_processing function on each game
        df1 = pbp_processing(pbp_game)
        pbp_scores = process_score_difference(df1)
        pbp_list.append(pbp_scores)
    pbp_processed = pd.concat(pbp_list)
    return pbp_processed[['GAME_ID','TIME_ELAPSED', 'ABS_SCORE_DIFF']]

def full_shot_detail_output(shot_detail, pbp):
    """ 
    This is a compliation function that takes the raw shot_detail and pbp dataframes and outputs a clean shot_detail dataframe
    """

    #Process the initial datasets
    shot_detail_df = shot_detail_time_elapsed(shot_detail)
    pbp_df = pbp_game_processing(pbp)
    pbp_df['ABS_SCORE_DIFF'] = pbp_df['ABS_SCORE_DIFF'].fillna(method = 'ffill')

    #Join the pbp data to have ABS SCORE DIFF
    shot_detail_full = shot_detail_df.merge(pbp_df, how = 'left', on = ['GAME_ID', 'TIME_ELAPSED'])
    #Mark if the shot attempt is a 3 pointer if the value of the column SHOT_ZONE_BASIC contains the character '3'
    shot_detail_full.loc[shot_detail_full['SHOT_ZONE_BASIC'].str.contains('3'), '3PT_ATTEMPTED_FLAG'] = 1
    shot_detail_full['3PT_ATTEMPTED_FLAG'] = shot_detail_full['3PT_ATTEMPTED_FLAG'].fillna(0)
    return shot_detail_full[['GAME_ID', 'GAME_DATE','PLAYER_ID', 'PLAYER_NAME', 'TEAM_NAME', 'PERIOD', 'MINUTES_REMAINING',
                             'SECONDS_REMAINING', 'TIME_ELAPSED', 'ABS_SCORE_DIFF','SHOT_ATTEMPTED_FLAG', 'SHOT_MADE_FLAG', '3PT_ATTEMPTED_FLAG']]


def get_nba_data(seasons=range(1996, 2022), 
                 data=("datanba", "nbastats", "pbpstats", "shotdetail"),
                 untar=False):
    if isinstance(seasons, int):
        seasons = (seasons,)
    need_data = tuple(["_".join([data, str(season)]) for (data, season) in product(data, seasons)])
    with urllib.request.urlopen("https://raw.githubusercontent.com/shufinskiy/nba_data/main/list_data.txt") as f:
        v = f.read().decode('utf-8').strip()
    
    name_v = [string.split("=")[0] for string in v.split("\n")]
    element_v = [string.split("=")[1] for string in v.split("\n")]
    
    need_name = [name for name in name_v if name in need_data]
    need_element = [element for (name, element) in zip(name_v, element_v) if name in need_data]
    
    for i in range(len(need_name)):
        t = urllib.request.urlopen(need_element[i])
        with open("".join([need_name[i], ".tar.xz"]), 'wb') as f:
            f.write(t.read())
        if untar:
            with tarfile.open("".join([need_name[i], ".tar.xz"])) as f:
                f.extract("".join([need_name[i], ".csv"]),'./')
            
            Path("".join([need_name[i], ".tar.xz"])).unlink()