import json
import sys, os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import unicodedata

from .wyscout_metadata_handler import get_match_mapper, get_metadata_dfs, get_metadata_maps
from .config import ESPN_DATA

import warnings
warnings.filterwarnings('ignore')

valid_competitions = {'England', 'France', 'Germany', 'Italy', 'Spain'}
original_labels = ['GK', 
                'LB', 'LCB', 'CB', 'RCB', 'RB', 
                'CDM', 
                'LM', 'LCM', 'CM', 'RCM', 'RM', 
                'LAM', 'CAM', 'RAM',
                'LW', 'LF', 'CF', 'RF', 'RW']

label_simplifier = {'LCB': 'CB',
                    'RCB': 'CB',
                    'CDM': 'CM',
                    'LCM': 'CM',
                    'RCM': 'CM',
                    #'LAM': 'LW',
                    #'RAM': 'RW',
                    'LAM': 'LW',
                    'RAM': 'RW',
                    'CAM': 'CM',
                    'LF': 'LW',
                    'RF': 'RW'}

final_espn_class_labels = ['GK', 
                           'LB', 'CB', 'RB', 
                           # 'CDM', 
                           'LM', 'CM', 'RM', 
                           # 'CAM', 
                           'LW', 'CF', 'RW']

espn_label_encoder = {pos: i for i, pos in enumerate(final_espn_class_labels)}
espn_label_decoder = {i: pos for i, pos in enumerate(final_espn_class_labels)}

class NameResolver:
    """
    Utility class used for matching espn match/team/player names with wyscout names,
    and other preprocessing tasks.
    """
    # match wyscout competition names (which are country names) to espn competition names
    country_to_espn_name = {'England': 'ENG-Premier League',
                            'France': 'FRA-Ligue 1',
                            'Germany': 'GER-Bundesliga',
                            'Italy': 'ITA-Serie A',
                            'Spain': 'ESP-La Liga'}

    # matching wyscout teams and espn teams which have different common names
    espn_to_wyscout_team_dict = {
        # england has no mismatches
        # spain
        'Alavés': 'Deportivo Alavés', 
        'Celta Vigo': 'Celta de Vigo',
        # france 
        'AS Monaco': 'Monaco', 
        'SC Amiens': 'Amiens SC', 
        'Angers': 'Angers SCO',
        'Paris Saint-Germain': 'PSG', 
        'Dijon FCO': 'Dijon', 
        'Lyon': 'Olympique Lyonnais', 
        'Stade Rennais': 'Rennes', 
        'Marseille': 'Olympique Marseille', 
        # germany
        'FC Augsburg': 'Augsburg',
        'SC Freiburg': 'Freiburg',
        'TSG Hoffenheim': 'Hoffenheim',
        'Mainz': 'Mainz 05',
        'VfB Stuttgart': 'Stuttgart',
        'Hertha Berlin': 'Hertha BSC',
        'Bayern Munich': 'Bayern München',
        'VfL Wolfsburg': 'Wolfsburg',
        'FC Cologne': 'Köln',
        'Hamburg SV': 'Hamburger SV', 
        'Borussia Mönchengladbach': 'Borussia M\'gladbach',
        # italy
        'AC Milan': 'Milan', 
        'AS Roma': 'Roma', 
        'Chievo Verona': 'Chievo'
    }

    wyscout_match_label_to_wyid = get_match_mapper(invert = True)    

    @staticmethod
    def pos_label_consolidater(original_label: str):
        """
        Maps original espn labels to final espn class labels.
        """
        try:
            return label_simplifier[original_label]
        except:
            return original_label

    @staticmethod
    def espn_to_wyscout_team_map(espn_name: str):
        """
        Maps espn team name to wyscout team name
        """
        try:    # if mismatched, then use dict
            return NameResolver.espn_to_wyscout_team_dict[espn_name]
        except: # if not mismatched, then return argument since name is same between espn and wyscout
            return espn_name

    @staticmethod
    def consolidate_match_wyid(label1: str, label2: str):
        """
        Maps wyscout match label to wyscout match id
        """
        try:
            return NameResolver.wyscout_match_label_to_wyid[label1]
        except KeyError:
            try:
                return NameResolver.wyscout_match_label_to_wyid[label2]
            except:
                return None

    @staticmethod
    def get_short_name(full_name: str):
        """
        Converts player's full  name to short name by abbreviating first name 
        (e.g. Jude Bellingham to J. Bellingham)
        params:
            full_name: a string representing a player's name.
        """
        names = full_name.split(' ')
        
        # if full_name is actually short_name, then return argument
        if len(names) == 1:
            return full_name
        
        # otherwise, abbreviate first name
        names[0] = names[0][0] + "."

        return ' '.join(names)

    @staticmethod
    def remove_diacritics(text: str):
        """
        Removes diacritical marks from player names. 
        """
        # Normalize to NFKD form which separates characters from diacritics
        normalized = unicodedata.normalize('NFKD', text)
        # Remove the diacritical marks (non-spacing marks)
        return ''.join(c for c in normalized if not unicodedata.combining(c))

    @staticmethod
    def standardize_text(text: str):
        """
        Standardizes player name by removing diacritical marks and converting to all lowercase
        """
        temp = NameResolver.remove_diacritics(text)
        temp = temp.lower()
        return temp

    @staticmethod
    def consolidate_player_wyid(id_from_short: int, id_from_full: int):
        """
        Given a pair of integers representing a player's possible wyid's, 
        the first of which may be <NA>, this returns the most likely wyid. 
        If id_from_full is not <NA>, then id_from_full is returned.
        If id_from_full is <NA>, then id_from_short is returned (even if id_from_short is also <NA>).
        """
        if pd.isna(id_from_full):
            return id_from_short
        else:
            return id_from_full

def get_raw_player_match_positions(countries: str | list = None):
    """
    Gets raw ESPN data, containing player positions for each individual match in the specified competitions.
    params:
        countries: a string, or list of strings in the set {'England', 'France', 'Germany', 'Italy', 'Spain'}. 
        If none, then the method defaults to all available countries.
    """
    if type(countries) == str:
        countries = [countries]
    if countries is None:
        countries = list(NameResolver.country_to_espn_name.keys())

    league_labels = [NameResolver.country_to_espn_name[country] for country in countries]
    all_pos = pd.read_csv(ESPN_DATA / 'all_positions.csv')
    relevant_pos = all_pos[all_pos['league'].isin(league_labels)].reset_index()
    return relevant_pos

def get_processed_player_match_positions(drop: bool = True):
    """
    Gets the raw ESPN data and performs preprocessing to match with Wyscout match/team/player id's.
    Note that not all data points are able to be matched, so some of the raw data is dropped when drop == True
    params:
        drop: a bool deciding whether or not to drop rows with any NA values
    """
    # import wyscout data needed to match with espn data
    wyscout_metadata_dfs = get_metadata_dfs(verbose = False)
    players_df = wyscout_metadata_dfs['players_df'] 

    wyscout_metadata_maps = get_metadata_maps(verbose = False)
    teams_map = wyscout_metadata_maps['teams_map']

    # get raw espn data
    raw_espn_df = get_raw_player_match_positions()

    # get rid of subs (not used in training)
    raw_espn_df = raw_espn_df[raw_espn_df['posLabel'] != 'SUB']

    # break down espn labels into date and team in order to facilitate name-resolving
    raw_espn_df['date'] = raw_espn_df['game'].apply(lambda x: x.split(' ')[0])
    raw_espn_df['teams'] = raw_espn_df['game'].apply(lambda x: ' '.join(x.split(' ')[1:]))
    raw_espn_df['espnTeam1'] = raw_espn_df['team']
    raw_espn_df['espnTeam2'] = raw_espn_df.apply(lambda row: row['teams'].replace(row['team'], '').strip('-').strip(' '), axis = 1)

    # resolve names by mapping espn team names to wyscout team names
    raw_espn_df['wyscoutTeam1'] = raw_espn_df['espnTeam1'].apply(NameResolver.espn_to_wyscout_team_map)
    raw_espn_df['wyscoutTeam2'] = raw_espn_df['espnTeam2'].apply(NameResolver.espn_to_wyscout_team_map)
    raw_espn_df['label1'] = raw_espn_df.apply(lambda row: row['date'] + ' ' + row['wyscoutTeam1'] + ' - ' + row['wyscoutTeam2'], axis = 1)
    raw_espn_df['label2'] = raw_espn_df.apply(lambda row: row['date'] + ' ' + row['wyscoutTeam2'] + ' - ' + row['wyscoutTeam1'], axis = 1)

    # drop intermediate cols
    raw_espn_df.drop(['index', 'season', 'game', 'team', 'date', 'teams', 'espnTeam1', 'espnTeam2', 'wyscoutTeam2'], axis = 1, inplace = True)
    raw_espn_df.rename(columns = {'wyscoutTeam1': 'team'}, inplace = True)

    # get team wyid's
    inverse_teams_map = {value: key for key, value in teams_map.items()}
    raw_espn_df['teamWyId'] = raw_espn_df['team'].map(inverse_teams_map).astype('Int64')

    # get match wyid's
    raw_espn_df['matchWyId'] = raw_espn_df.apply(lambda row: NameResolver.consolidate_match_wyid(row['label1'], row['label2']), axis = 1).astype('Int64')
    raw_espn_df.dropna(subset = 'matchWyId', inplace = True)

    # standardize player names for better matching
    raw_espn_df['standardizedShortName'] = raw_espn_df['player'].apply(NameResolver.standardize_text).apply(NameResolver.get_short_name)
    raw_espn_df['standardizedName'] = raw_espn_df['player'].apply(NameResolver.standardize_text)
    players_df['standardizedShortName'] = players_df['shortName'].apply(NameResolver.standardize_text)
    players_df['standardizedFullName'] = players_df['fullName'].apply(NameResolver.standardize_text)

    # use standardized names to get player wyid's
    standardized_short_name_to_wyid = dict(zip(players_df['standardizedShortName'], players_df['wyId']))
    standardized_full_name_to_wyid = dict(zip(players_df['standardizedFullName'], players_df['wyId']))

    # get wyid according to player's short name and full name
    raw_espn_df['playerWyIdFromShort'] = raw_espn_df['standardizedShortName'].map(standardized_short_name_to_wyid).astype('Int64')
    raw_espn_df['playerWyIdFromName'] = raw_espn_df['standardizedName'].map(standardized_full_name_to_wyid).astype('Int64')

    # wyid from full name is more reliable than short name because multiple players may have same short name
    # only keep id from short name if id from full name is <NA>
    raw_espn_df['playerWyId'] = raw_espn_df.apply(lambda row: 
                                                    NameResolver.consolidate_player_wyid(row['playerWyIdFromShort'], row['playerWyIdFromShort']),
                                                    axis = 1)

    final_espn_player_pos_df = raw_espn_df[['matchWyId', 'teamWyId', 'playerWyId', 'posLabel']]

    # drop unusable data
    if drop:
        final_espn_player_pos_df = final_espn_player_pos_df.dropna()
    
    # final_espn_player_pos_df = final_espn_player_pos_df[final_espn_player_pos_df['posLabel'] != 'SUB'].reset_index(drop = True)
    final_espn_player_pos_df['posLabel'] = final_espn_player_pos_df['posLabel'].apply(NameResolver.pos_label_consolidater)
    final_espn_player_pos_df.rename(columns = {'matchWyId': 'matchId', 
                                                'teamWyId': 'teamId', 
                                                'playerWyId': 'playerId'}, 
                                    inplace = True)
    
    return final_espn_player_pos_df.reset_index(drop = True)

def get_match_formations():
    espn_data = get_processed_player_match_positions(drop = False)
    espn_data = espn_data.set_index(['matchId', 'teamId'])

    label_to_category = {'GK': 'GK',
                        'LB': 'DF',
                        'CB': 'DF',
                        'RB': 'DF',
                        'LM': 'MD',
                        'CM': 'MD',
                        'RM': 'MD',
                        'LW': 'FW',
                        'CF': 'FW',
                        'RW': 'FW'}

    espn_data_simplified = espn_data.copy()
    espn_data_simplified['category'] = espn_data_simplified['posLabel'].map(label_to_category)

    def match_lineup_to_formation(df):
        pos_counts = df['category'].value_counts()
        num_defense = pos_counts['DF'].item()
        num_mid = pos_counts['MD'].item()
        num_forward = pos_counts['FW'].item()
        if (num_defense + num_mid + num_forward != 10):
            return ['Invalid']
        else:
            return [(num_defense, num_mid, num_forward)]

    formations = []
    all_multi_indices = espn_data_simplified.index.drop_duplicates()
    for multi_index in all_multi_indices:
        match_lineup = espn_data_simplified.loc[multi_index]
        formations.append(match_lineup_to_formation(match_lineup))

    match_formations = pd.DataFrame(formations, index = all_multi_indices, columns = ['formation'])
    match_formations = match_formations[match_formations['formation'] != 'Invalid']

    # treat 4-2-4 as 4-4-2 (since LW and RW are probably LM and RM)
    # treat 2-5-3 as 4-3-3 (since LM and RM are probably LB and RB)
    correct_espn_formations = [
        (5, 4, 1), 
        (5, 3, 2),
        (4, 5, 1),
        (4, 4, 2), 
        (4, 3, 3),
        (3, 5, 2), 
        (3, 4, 3), 
    ]

    formation_corrector = {formation: formation for formation in correct_espn_formations}
    formation_corrector.update({
        (4, 2, 4): (4, 4, 2),
        (2, 5, 3): (4, 3, 3)
    })

    match_formations['formation'] = match_formations['formation'].map(formation_corrector)

    return match_formations