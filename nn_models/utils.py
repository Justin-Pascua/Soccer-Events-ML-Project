from remote_data_handlers.remote_match_data import RemoteMatchData

import nn_models.player_data as player_data
import nn_models.heatmap_data as heatmap_data
from nn_models.player_classifier import PlayerClassifier
from nn_models.heatmap_classifier import HeatmapClassifier


import torch
import torch.nn as nn  

# take data in RemoteMatchData instance and convert to model input
def match_data_to_model_input(current_match: RemoteMatchData):
    """
    Convert RemoteMatchData instance to model input for PlayerClassifier or HeatmapClassifier.
    params:
        current_match: a RemoteMatchData instance storing data for the desired match
    """
    # match details
    match_id = current_match.match_id

    team1_id = current_match.team1
    team2_id = current_match.team2

    team1_players = current_match.team1_players
    team2_players = current_match.team2_players

    # get features dataframe
    match_subevent_counts_df =  player_data.CustomData.get_subevent_counts(current_match.events_df)
    match_coords_df = heatmap_data.CustomData.get_coords_df(current_match.events_df)
    features_df = match_coords_df.merge(match_subevent_counts_df, how = 'inner', left_index = True, right_index = True)
    features_df['heatmap'] = features_df['coordsList'].apply(heatmap_data.CustomTransforms.coords_list_to_heatmap)

    # convert dataframe to tensors to be fed to model
    subevent_cols = features_df.columns[1:-1]
    heatmap_col = features_df.columns[-1]

    team1_features_df = features_df.loc[match_id, team1_id, team1_players]
    team1_subevent_counts_tensor = torch.tensor(team1_features_df[subevent_cols].values.astype(float), dtype = torch.float32)
    team1_heatmap_tensor = torch.stack(list(team1_features_df[heatmap_col].values))
    team1_model_input = (team1_heatmap_tensor, team1_subevent_counts_tensor, None)

    team2_features_df = features_df.loc[match_id, team2_id, team2_players]
    team2_subevent_counts_tensor = torch.tensor(team2_features_df[subevent_cols].values.astype(float), dtype = torch.float32)
    team2_heatmap_tensor = torch.stack(list(team2_features_df[heatmap_col].values))
    team2_model_input = (team2_heatmap_tensor, team2_subevent_counts_tensor, None)

    return team1_model_input, team2_model_input

# applying model to match data container
def apply_model_to_match(model: PlayerClassifier | HeatmapClassifier, current_match: RemoteMatchData, output_type: str = 'probabilities'):
    """
    Apply PlayerClassifier or HeatmapClassifier model to a RemoteMatchData instance to get position predictions for players on both teams.
    params:
        model: a PlayerClassifier or HeatmapClassifier model
        current_match: a RemoteMatchData instance storing data for the desired match
        output_type: a str indicating the type of output desired. Must be either 'probabilities', 'logits', or 'hidden'.
    """
    if output_type not in ['probabilities', 'logits', 'hidden']:
        raise ValueError("output_type must be either 'probabilities', 'logits', or 'hidden'")

    team1_model_input, team2_model_input = match_data_to_model_input(current_match)

    team1_output, team2_output = None, None
    with torch.no_grad():
        match output_type:
            case 'probabilities':
                team1_output = nn.Softmax(dim = 1)(model(team1_model_input))
                team2_output = nn.Softmax(dim = 1)(model(team2_model_input))
            case 'logits':
                team1_output = model(team1_model_input)
                team2_output = model(team2_model_input)
            case 'hidden':  
                team1_output = model(team1_model_input, get_hidden = True)
                team2_output = model(team2_model_input, get_hidden = True)

    return team1_output, team2_output