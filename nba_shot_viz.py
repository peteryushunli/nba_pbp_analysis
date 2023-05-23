import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

def ingest_data(season):
    '''
    Ingest the data from the specified season as the trailing year of a season, i.e. 2021-2022 season is 2022
    '''
    df = pd.read_csv('processed_pbp/shot_detail_pbp_' + str(season) + '.csv')
    return df

def clean_pbp_data(df):
    '''Change Shot Values'''
    #Create shot value column
    df['SHOT_VALUE'] = df['3PT_ATTEMPTED_FLAG'].apply(lambda x: 3 if x == 1 else 2) 
    #Separate 3 point makes from 2 point makes
    df['3PM'] = np.where(df['SHOT_VALUE'] == 3, df['SHOT_MADE_FLAG'], 0)
    #Rename SHOT_ATTEMPTED_FLAG and 3PT_ATTEMPTED_FLAG columns
    df.rename(columns = {'SHOT_ATTEMPTED_FLAG':'FGA', '3PT_ATTEMPTED_FLAG':'3PA', 'SHOT_MADE_FLAG': 'FGM'}, inplace = True)

    '''Change Time Values'''
    #Change any PERIOD > 4 values to 4
    df['PERIOD'] = np.where(df['PERIOD'] > 4, 4, df['PERIOD'])
    #Calculate the raw minutes remaining
    df['RAW_MINUTES_REMAINING'] = df['MINUTES_REMAINING'] + (4-df['PERIOD'])*12

    #Drop unnecessary columns
    df.drop(columns = ['SHOT_VALUE', 'GAME_ID', 'GAME_DATE', 'PLAYER_ID', 'PERIOD', 'MINUTES_REMAINING', 'SECONDS_REMAINING', 'TIME_ELAPSED'], inplace = True)
    return df
    
def create_buckets(df):
    ''' This function will intake the pbp data and create bucketes from ABS_SCORE_DIFF and RAW_MINUTES_REMAINING'''
    #Create buckets for the column ABS_SCORE_DIFF: 0-5, 6-10, 11-15, 16-20, 21-25, 26+ 
    df['ABS_SCORE_DIFF_BUCKETS'] = pd.cut(df['ABS_SCORE_DIFF'], bins = [-1, 5, 10, 15, 20, 100],
                                        labels = ['0-5', '6-10', '11-15', '16-20', '21+'])
    #Create buckets for RAW_MINUTES_REMAINING in 4 minute intervals starting from 48 minutes
    df['RAW_MINUTES_REMAINING_BUCKETS'] = pd.cut(df['RAW_MINUTES_REMAINING'], bins = [-1, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48],
                                        labels = ['4-0', '8-5', '12-9', '16-13', '20-17', '24-21', '28-25', '32-29', '36-33', '40-37', '44-41', '48-45'])
    
    return df.drop(columns = ['ABS_SCORE_DIFF', 'RAW_MINUTES_REMAINING'])

def aggregate_data(df, player_name = None, team_name = None):
    ''' Aggregate the data based on the player or team name'''
    #Filter the dataframe for the player, if n
    if player_name is not None:
        df = df.loc[df['PLAYER_NAME'] == player_name]
    elif team_name is not None:
        df = df.loc[df['TEAM_NAME'] == team_name]
    else: df = df

    #Aggregate the data
    agg_df = df.groupby(['RAW_MINUTES_REMAINING_BUCKETS', 'ABS_SCORE_DIFF_BUCKETS'], as_index = False).sum()
    #Calculate the EFG
    agg_df['EFG'] = round((agg_df['FGM'] + 0.5*agg_df['3PM']) / agg_df['FGA'],3)
    agg_df.drop(columns = ['PLAYER_NAME', 'TEAM_NAME'], inplace = True)

    return agg_df

def pivot_efg(agg_df):
    ''' Pivot the aggregated dataframe to create a heatmap of EFG'''
    #Create a pivot table where the index is the absolute score difference, and the columns is the RAW_MINUTES_REMAINING_BUCKETS, and both are sorted in descending order
    agg_df_pivot = agg_df.pivot(index = 'ABS_SCORE_DIFF_BUCKETS', columns = 'RAW_MINUTES_REMAINING_BUCKETS', values = 'EFG')
    agg_df_pivot = agg_df_pivot.reindex(index = agg_df_pivot.index[::-1])
    agg_df_pivot = agg_df_pivot.reindex(columns = agg_df_pivot.columns[::-1])
    #Round the EFG values to 2 decimal places
    agg_df_pivot = agg_df_pivot.round(2)
    return agg_df_pivot

def pivot_attempt_fraction(agg_df):
    '''Create another pivot table to show the fraction of shot attempts at each time bucket that will be used for annotating the heatmap'''
    #Pivot attempts and makes
    pivot_attempts = pd.pivot_table(agg_df, values='FGA', index='ABS_SCORE_DIFF_BUCKETS', columns='RAW_MINUTES_REMAINING_BUCKETS', aggfunc=np.sum)
    pivot_makes = pd.pivot_table(agg_df, values='FGM', index='ABS_SCORE_DIFF_BUCKETS', columns='RAW_MINUTES_REMAINING_BUCKETS', aggfunc=np.sum)
    #calcualte the fraction of makes / attempts
    pivot_fraction = pivot_makes / pivot_attempts
    #Fill null values with 0
    pivot_fraction.fillna(0, inplace = True)
    # Convert the fraction pivot table to strings representing fractions
    pivot_fraction_str = pivot_makes.astype(str) + "/" + pivot_attempts.astype(str)
    #reverse the order of the rows and columns (to match the order of the EFG pivot table)
    pivot_fraction_str = pivot_fraction_str.reindex(index = pivot_fraction_str.index[::-1])
    pivot_fraction_str = pivot_fraction_str.reindex(columns = pivot_fraction_str.columns[::-1])
    return pivot_fraction_str

def create_heatmap(agg_df_pivot, pivot_fraction_str, team_name = None, player_name = None, season = None):
    #Create Heatmap
    fig, ax1 = plt.subplots(figsize=(10,6))
    #Create a color map where high values are red and low values are blue
    cmap = sns.diverging_palette(250, 10, as_cmap=True)
    #Outline the entirety of the heatmap so that null values are not confused with 0 values
    heat_map = sns.heatmap(agg_df_pivot, annot=pivot_fraction_str, fmt= '', linewidths=.5, cmap=cmap, vmin = 0.25, vmax = 0.7, ax=ax1)
    if type(season) == int:
        #Convert the season to a string
        season = str(season-1) + '-' + str(season)

    if player_name is not None:
        plt.title(player_name+': eFG% by Time Remaining and Abs. Score Diff in '+season)
    elif team_name is not None:
        plt.title(team_name+': eFG% by Time Remaining and Abs. Score Diff in '+season)
    else: plt.title('eFG% by Time Remaining and Abs. Score Diff in '+season)
    plt.xlabel('Minutes Remaining')
    plt.ylabel('Absolute Score Difference')

    # Format the color bar
    color_bar = heat_map.collections[0].colorbar
    color_bar.set_label('Effective Field Goal %')
    # Shift the color bar label slightly to the right
    color_bar.ax.yaxis.set_label_coords(4, 0.5)

    # Format the y-tick labels to be fractions with no decimals
    num_ticks = np.linspace(0.25, 0.7, len(color_bar.get_ticks()))
    new_labels = ['{:.0f}%'.format(i*100) for i in color_bar.get_ticks()]
    color_bar.set_ticks(num_ticks)
    color_bar.set_ticklabels(new_labels)

    #Add more space between the Ylabels and y-axis ticks
    ax1.tick_params(axis='y', which='minor', pad=15)

    # Create second x-axis
    ax2 = ax1.twiny()

    # Move twinned axis ticks and label from top to bottom
    ax2.xaxis.set_ticks_position("bottom")
    ax2.xaxis.set_label_position("bottom")
    
    # Offset the twin axis below the host
    ax2.spines["bottom"].set_position(("axes", -0.15))
    ax2.grid(False)
    ax1.grid(False)

    # Hide the spines (the line box)
    for sp in ax2.spines.values():
        sp.set_visible(False)

    # Decide the ticklabel to be put on the new x-axis (quarter labels)
    new_ticks = ['','1Q', '2Q', '3Q', '4Q or OT', '']

    # Decide the new ticks position
    new_position = [0,4, 16, 28, 40, 48]
    ax2.tick_params(axis='x', length=0)

    # Set the new ticks and labels
    ax2.set_xticks(new_position)
    ax2.set_xticklabels(new_ticks)

    plt.show()

def efg_scorediff_heatmap(season, team_name = None, player_name = None):
    '''
    This is the complete function that will intake the pbp data and create a heatmap of EFG by time remaining and absolute score difference
    Input variables are:
    - season: the season to be analyzed, input as the trailing year of a season, i.e. 2021-2022 season is 2022
    - team_name: the team to be analyzed, input as a string, i.e. 'Los Angeles Lakers'
    - player_name: the player to be analyzed, input as a string, i.e. 'LeBron James'
    '''
    df = ingest_data(season)
    #Run helper functions to manipulate the data
    clean_df = clean_pbp_data(df)
    clean_df = create_buckets(df)
    agg_df = aggregate_data(clean_df, player_name = player_name, team_name = team_name)
    agg_df_pivot = pivot_efg(agg_df)
    pivot_fraction_str = pivot_attempt_fraction(agg_df)
    #Create the heatmap
    create_heatmap(agg_df_pivot, pivot_fraction_str, team_name = team_name, player_name = player_name, season = season)

def create_selection_data(selected_season, selected_filter):
    ''' This function will create the data for the dropdown selection'''
    #Remove all characters after and including the dash
    season = selected_season.split('-')[0]
    #Ingest the dataset based on the selected season
    df = ingest_data(season)
    
    #Create the player and team lists
    player_list = df['PLAYER_NAME'].unique().tolist()
    team_list = df['TEAM_NAME'].unique().tolist()

    if selected_filter == 'Player':
        return player_list
    else:
        return team_list