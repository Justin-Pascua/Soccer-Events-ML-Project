import json
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from .config import WYSCOUT_DATA    # get path to wyscout dataset


valid_competitions = {'England', 'European_Championship', 'France',
                    'Germany', 'Italy', 'Spain', 'World_Cup'}

wyscout_class_labels = ['GK', 'DF', 'MD', 'FW']

wyscout_label_encoder = {'GK': 0, 'DF': 1, 'MD': 2, 'FW': 3}
wyscout_label_decoder = {value: key for key, value in wyscout_label_encoder.items()}

def get_match_mapper(invert: bool = False):
    """
    Returns dictionary used to map Wyscout match id's and Wyscout match labels.
    params:
        invert: a bool indicating whether or not to invert the dictionary. 
        If false, then returns a mapping from match id's to match labels.
        If true, then returns a mapping from match labels to match id's
    """
    mapper = dict()

    competitions = ['England', 'France', 'Germany', 'Italy', 'Spain']

    for competition in competitions:
        with open(WYSCOUT_DATA / f'matches/matches_{competition}.json') as json_data:
            matches_in_current_comp = json.load(json_data)

        ids = [match['wyId'] for match in matches_in_current_comp]
        dates = [match['dateutc'].split(' ')[0] for match in matches_in_current_comp]
        wyscout_match_teams = [match['label'].encode('latin-1').decode('unicode-escape').split(',')[0] for match in matches_in_current_comp]
        ids, dates, wyscout_match_teams

        processed_fisghare_match_labels = [dates[i] + ' ' + wyscout_match_teams[i] for i in range(len(dates))]

        wyid_to_match_label = dict(zip(ids, processed_fisghare_match_labels))

        mapper.update(wyid_to_match_label)

    if invert:
        inverse_map = {value: key for key, value in mapper.items()}
        return inverse_map

    return mapper


# Dataframe methods
def get_metadata_dfs(verbose: bool = True):
    """
    Gets all metadata in the form of dataframes. 
    params:
        verbose: a bool indicating whether or not to print after each dataframe has been loaded.
    """
    competitions_df = get_competitions_df(verbose)
    players_df = get_players_df(verbose)
    teams_df = get_teams_df(verbose)
    eventids_df = get_eventids_df(verbose)
    tags_df = get_tags_df(verbose)
    playeranks_df = get_ranks_df(verbose)
    return {'competitions_df': competitions_df, 'players_df': players_df,
            'teams_df': teams_df, 'eventids_df': eventids_df, 'tags_df': tags_df,
            'playeranks_df': playeranks_df}

def get_competitions_df(verbose: bool = True):
    """
    Gets competitions metadata as a dataframe.
    params:
        verbose: a bool indicating whether or not to print after the dataframe has been loaded.
    """
    competitions = {}
    with open(WYSCOUT_DATA / 'competitions.json') as json_data:
        competitions = json.load(json_data)
        if(verbose):
            print("Imported competitions data")
    competitions_df = pd.DataFrame([(e['name'],
                                e['wyId'], 
                                e['format'],
                                e['type'])
                                for e in competitions],
                                columns = ['competitionName', 'wyId', 'format', 'type'])
    return competitions_df

def get_players_df(verbose: bool = True):
    """
    Gets player metadata (i.e. names, club, country, wyId, and usual position) as a dataframe.
    params:
        verbose: a bool indicating whether or not to print after the dataframe has been loaded.
    """
    players = {}
    with open(WYSCOUT_DATA / 'players.json') as json_data:
        players = json.load(json_data)
        if(verbose):
            print("Imported players data")
    players_df = pd.DataFrame([(e['shortName'], 
                                e['firstName'],
                                e['middleName'],
                                e['lastName'],
                                e['wyId'], 
                                e['currentTeamId'], 
                                e['currentNationalTeamId'], 
                                e['role']['code2']) 
                                for e in players],
                            columns = ['shortName', 'firstName', 'middleName', 'lastName', 'wyId', 'clubWyId', 'nationalTeamWyId', 'position'])
    for col_name in ['shortName', 'firstName', 'middleName', 'lastName']:
        players_df[col_name] = players_df[col_name].str.encode('latin-1').str.decode('unicode-escape')
    
    players_df['fullName'] = players_df.apply(lambda row: 
                                        row['firstName'] + ' ' + row['lastName'], 
                                        axis = 1)

    return players_df

def get_teams_df(verbose: bool = True):
    """
    Gets team metadata (i.e. team name, city, and wyId) as a dataframe.
    params:
        verbose: a bool indicating whether or not to print after the dataframe has been loaded.
    """
    teams = {}
    with open(WYSCOUT_DATA / 'teams.json') as json_data:
        teams = json.load(json_data)
        if(verbose):
            print("Imported teams data")
    teams_df = pd.DataFrame([(e['name'],
                            e['wyId'],
                            e['city'],)
                            for e in teams],
                            columns = ['teamName', 'wyId', 'city'])
    teams_df['teamName'] = teams_df['teamName'].str.encode('latin-1').str.decode('unicode-escape')
    teams_df['city'] = teams_df['city'].str.encode('latin-1').str.decode('unicode-escape')
    return teams_df

def get_eventids_df(verbose: bool = True):
    """
    Gets event id details (i.e. event ids, subevent ids, and descriptions) as a dataframe.
    params:
        verbose: a bool indicating whether or not to print after the dataframe has been loaded.
    """
    eventids_df = pd.read_csv(WYSCOUT_DATA / 'eventid2name.csv')
    if(verbose):
        print("Imported events data")
    return eventids_df

def get_tags_df(verbose: bool = True):
    """
    Get tags details (i.e. tag numbers and descriptions) as a dataframe
    params:
        verbose: a bool indicating whether or not to print the dataframe has been loaded.
    """
    tags_df = pd.read_csv(WYSCOUT_DATA / 'tags2name.csv')
    if(verbose):
        print("Imported tags data")
    return tags_df

def get_ranks_df(verbose: bool = True):
    """
    Gets 'playerank' data as a dataframe.
    params:
        verbose: a bool indicating whether or not to print after the dataframe has been loaded.
    """
    with open(WYSCOUT_DATA / 'playerank.json') as json_data:
        ranks = json.load(json_data)
    if verbose:
        print("Imported playerank data")

    player_ranks = pd.DataFrame({
        'matchId': [e['matchId'] for e in ranks],
        'playerId': [e['playerId'] for e in ranks],
        'playerank': [e['playerankScore'] for e in ranks],
        'position': [e['roleCluster'] for e in ranks],
    }).set_index(['matchId', 'playerId'])

    return player_ranks

# Dictionary methods
def get_metadata_maps(competitions_df = None, players_df = None, teams_df = None, 
                    eventids_df = None, tags_df = None,
                    verbose: bool = True):
    """
    Gets all available maps for Wyscout metadata.
    params:
        competitions_df: competition metadata dataframe passed to get_competitions_map. 
        If None, then get_competitions_df is called to get substitute.
        players_df: player metadata dataframe passed to get_players_map. 
        If None, then get_players_df is called to get substitute.
        teams_df: team metadata dataframe passed to get_teams_map. 
        If None, then get_teams_df is called to get a substitute.
        eventids_df: event id's dataframe passed to get_eventids_map. 
        If None, then get_eventids_df is called to get a substitute.
        tags_df: event tags dataframe passed to get_tags_map.
        If None, then get_tags_df is called to get a substitute.
        verbose: a bool indicating whether or not to print after each dataframe has been loaded 
        if any of the dataframe getter functions are called.
    """
    competitions_map = get_competitions_map(competitions_df, verbose)
    player_to_short_name, player_to_full_name, player_to_pos, player_to_team = get_players_maps(players_df, verbose)
    teams_map = get_teams_map(teams_df, verbose)
    events_map, subevents_map = get_eventids_map(eventids_df, verbose)
    tags_map = get_tags_map(tags_df, verbose)

    return {'competitions_map': competitions_map, 'player_to_short_name': player_to_short_name, 'player_to_full_name': player_to_full_name, 
            'player_to_pos': player_to_pos, 'player_to_team': player_to_team, 'teams_map': teams_map, 
            'events_map': events_map, 'subevents_map': subevents_map, 'tags_map': tags_map}

def get_competitions_map(competitions_df = None, verbose: bool = True):
    """
    Returns a dictionary mapping competition wyId's to competition names.
    params:
        competitions_df: competition metadata dataframe used to create the dictionary. 
        If None, then get_competitions_df is called to get substitute.
        verbose: a bool indicating whether or not to print after the dataframe has been loaded 
        in the case where get_competitions_df is called.
    """
    if(competitions_df is None):
        competitions_df = get_competitions_df(verbose)
    competitions_map = dict(zip(competitions_df['wyId'], competitions_df['competitionName']))
    return competitions_map

def get_players_maps(players_df = None, verbose: bool = True):
    """
    Returns four dictionaries which map player wyId's to 
    i) short name, ii) full name, iii) usual position, iv) and team wyId respectively
    params:
        players_df: player metadata dataframe used to create the dictionaries. 
        If None, then get_players_df is called to get substitute.
        verbose: a bool indicating whether or not to print after the dataframe has been loaded 
        in the case where get_players_df is called.
    """
    if(players_df is None):
        players_df = get_players_df(verbose)
    player_to_short_name = dict(zip(players_df['wyId'], players_df['shortName']))
    player_to_full_name = dict(zip(players_df['wyId'], players_df['fullName']))
    player_to_pos = dict(zip(players_df['wyId'], players_df['position']))
    player_to_team = dict(zip(players_df['wyId'], players_df['clubWyId']))
    return player_to_short_name, player_to_full_name, player_to_pos, player_to_team

def get_teams_map(teams_df = None, verbose: bool = True):
    """
    Returns a dictionary mapping team wyId's to team name
    params:
        teams_df: teams metadata dataframe used to create the dictionary. 
        If None, then get_teams_df is called to get substitute.
        verbose: a bool indicating whether or not to print after the dataframe has been loaded 
        in the case where get_teams_df is called.
    """
    if(teams_df is None):
        teams_df = get_teams_df(verbose)
    teams_map = dict(zip(teams_df['wyId'], teams_df['teamName']))
    return teams_map

def get_eventids_map(eventids_df = None, verbose: bool = True):
    """
    Returns two dictionaries. 
    The first dict maps event id's to event labels. 
    The second dict maps subevent id's to subevent labels.
    params:
        eventids_df: event id's dataframe used to create the dictionaries. 
        If None, then get_eventids_df is called to get a substitute.
        verbose: a bool indicating whether or not to print after the dataframe has been loaded 
        in the case where get_eventids_df is called.
    """
    if(eventids_df is None):
        eventids_df = get_eventids_df(verbose)
    events_map = dict(zip(eventids_df['event'], eventids_df['event_label']))
    subevents_map = dict(zip(eventids_df['subevent'], eventids_df['subevent_label']))
    return events_map, subevents_map

def get_tags_map(tags_df = None, verbose: bool = True):
    """
    Returns a dictionary mapping event tags to tag descriptions.
    params:
        tags_df: event tags dataframe used to create the dictionaries.
        If None, then get_tags_df is called to get a substitute.
        verbose: a bool indicating whether or not to print after the dataframe has been loaded 
        in the case where get_tags_df is called.
    """
    if(tags_df is None):
        tags_df = get_tags_df(verbose)
    tags_map = dict(zip(tags_df['Tag'], tags_df['Description']))
    return tags_map
