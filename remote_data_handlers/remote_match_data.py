from pymongo import MongoClient
from pymongoarrow.monkey import patch_all
patch_all()

import ast

class RemoteMatchData:
    def __init__(self, competition_string: str, wyid: int, client: MongoClient):
        # get match wyId
        self.match_id = wyid

        # get events_df
        events = client['soccer_events'][f'events_{competition_string}'].find_pandas_all({'matchId': wyid})
        events.drop(columns = ['_id'], inplace = True)

        # convert to format compatible with local_data_handlers
        events['positions'] = events.apply(lambda row: [{'y': row['initY'], 'x': row['initX']}], axis = 1)
        events['tags'] = events['acc'].apply(lambda x: [1801] if x else [])
        events.drop(columns = ['initX', 'initY', 'acc'], inplace = True)
        self.events_df = events

        # get match details
        unprocessed_details = client['soccer_events'][f'matches_{competition_string}'].find_one({'wyId': wyid})
        self.details = {
            'date': unprocessed_details['date'],
            'dateutc': unprocessed_details['dateutc'],
            'label': unprocessed_details['label'].encode().decode('unicode_escape'),
            'winner': unprocessed_details['winner']
        }

        team1_id, team2_id = unprocessed_details['teamsData'].keys()
        self.details['team1_id'] = int(team1_id) 
        self.details['team2_id'] = int(team2_id) 
        self.details['team1_players'] = [e['playerId'] for e in unprocessed_details['teamsData'][team1_id]['formation']['lineup']]
        self.details['team2_players'] = [e['playerId'] for e in unprocessed_details['teamsData'][team2_id]['formation']['lineup']]

        # unpack match details
        self.winner = self.details['winner']
        self.team1 = self.details['team1_id']
        self.team2 = self.details['team2_id']
        self.team1_players = self.details['team1_players']
        self.team2_players = self.details['team2_players']
        
        # sort (used for indexing dataframes)
        self.team1_players.sort()  
        self.team2_players.sort()