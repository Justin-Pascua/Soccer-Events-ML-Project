from nn_models.utils import match_data_to_model_input, apply_model_to_match
from nn_models.heatmap_autoencoder import HeatmapAutoencoder
from nn_models.heatmap_classifier import HeatmapClassifier
from nn_models.player_classifier import PlayerClassifier
from remote_data_handlers.remote_match_data import RemoteMatchData
from remote_data_handlers.metadata_handler import get_eventids_map

import torch

events_map, _ = get_eventids_map(verbose = False)
final_espn_class_labels = ['GK', 
                           'LB', 'CB', 'RB', 
                           # 'CDM', 
                           'LM', 'CM', 'RM', 
                           # 'CAM', 
                           'LW', 'CF', 'RW']

class PlayerDetails():
    def __init__(self, current_match: RemoteMatchData, 
                 hm_model: HeatmapClassifier | PlayerClassifier, 
                 hm_ae: HeatmapAutoencoder):
        
        self.team1_players = current_match.team1_players
        self.team2_players = current_match.team2_players

        # player position predictions
        team1_prob, team2_prob = apply_model_to_match(hm_model, current_match)
        team1_prob_dict = dict(zip(current_match.team1_players, team1_prob))
        team2_prob_dict = dict(zip(current_match.team2_players, team2_prob))
        self.probs_dict = team1_prob_dict | team2_prob_dict

        # event dot maps
        team1_input, team2_input = match_data_to_model_input(current_match)
        team1_event_dots = team1_input[0]
        team2_event_dots = team2_input[0]
        team1_event_dots_dict = dict(zip(current_match.team1_players, team1_event_dots))
        team2_event_dots_dict = dict(zip(current_match.team2_players, team2_event_dots))
        self.event_dots_dict = team1_event_dots_dict | team2_event_dots_dict

        # inferred heatmaps
        team1_hm = hm_ae(team1_event_dots).detach()
        team2_hm = hm_ae(team2_event_dots).detach()
        team1_hm_dict = dict(zip(current_match.team1_players, team1_hm))
        team2_hm_dict = dict(zip(current_match.team2_players, team2_hm))
        self.hm_dict = team1_hm_dict | team2_hm_dict

        # event dfs
        events_df = current_match.events_df
        events_df = events_df[~events_df['eventId'].isin([5,6,7])]
        events_df['event'] = events_df['eventId'].map(events_map)
        self.events_df = events_df

    def get_position_probs(self, player_wyid: int):
        try:
            probabilities = self.probs_dict[player_wyid].tolist()
            return dict(zip(final_espn_class_labels, probabilities))
        except Exception as e:
            raise e
        
    def get_event_dots(self, player_wyid: int):
        try:
            player_event_dots = self.event_dots_dict[player_wyid].squeeze()
            if player_wyid in self.team2_players:
                player_event_dots = torch.flip(player_event_dots, dims = [0, 1])
            return player_event_dots
        except Exception as e:
            raise e
        
    def get_heatmap(self, player_wyid: int):
        try:
            player_hm = self.hm_dict[player_wyid].squeeze()
            if player_wyid in self.team2_players:
                player_hm = torch.flip(player_hm, dims = [0, 1])
            return player_hm
        except Exception as e:
            raise e
        
    def get_event_counts(self, player_wyid: int):
        selected_player_events = self.events_df[self.events_df['playerId'] == player_wyid]
        event_counts = selected_player_events['event'].value_counts()
        return event_counts
    
    