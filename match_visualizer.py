from local_data_handlers.wyscout_metadata_handler import get_players_maps, get_teams_map
from local_data_handlers.espn_data_handler import espn_label_decoder
from remote_data_handlers.remote_match_data import RemoteMatchData

from nn_models.player_classifier import PlayerClassifier
from nn_models.heatmap_classifier import HeatmapClassifier
from nn_models.heatmap_autoencoder import HeatmapAutoencoder
from nn_models.utils import apply_model_to_match
from formation_inference import get_best_formation
from passing_networks import NxPassingNetworks as NxUtils

import torch
import networkx as nx
import numpy as np
import pandas as pd
import ast

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import plotly.graph_objects as go

CATEGORIES = ['GK', 'DF', 'MD', 'FW']
POS_TO_CATEGORY = {
    'GK': 'GK',
    'LB': 'DF',
    'CB': 'DF',
    'RB': 'DF',
    'CM': 'MD',
    'LM': 'MD',
    'RM': 'MD',
    'LW': 'FW',
    'RW': 'FW',
    'CF': 'FW',
}
GK_VERTICAL = 4
DEF_VERTICAL = 15 
MIDFIELD_VERTICAL = 30
FORWARD_VERTICAL = 42

LEFT_HORIZONTAL = 10
MIDDLE_HORIZONTAL = 50
RIGHT_HORIZONTAL = 90

player_to_short_name, player_to_full_name, _ , _ = get_players_maps(verbose = False)
teams_map = get_teams_map(verbose = False)

#----------------GENERAL----------------
def name_standardizer(name: str):
    """
    This function standardizes player names.
    If the name is 1 word, then no change is applied.
    If the name is 2 words, then the first word is abbreviated.
    If the name is more than 2 words, then only the first and last word and kept, and the first word is abbreviated.
    params:
        name: a string representing a player's name
    """
    name_parts = name.split(' ')
    if len(name_parts) == 2:
        return f"{name_parts[0][0]}. {name_parts[1]}"
    elif len(name_parts) > 2:
        return f"{name_parts[0][0]}. {name_parts[-1]}"
    else:
        return name

def reflect_single_pos(pos: tuple):
    """
    Given a tuple, representing a coord in [0, 100] x [0, 100],
    this applies the map (x, y) -> (100 - x, 100 - y)
    params:
        pos: a tuple representing a 2d coord in [0, 100] x [0, 100]
    """
    return (100 - pos[0], 100 - pos[1])

def remove_overtime_suffix(input_str: str):
    """
    Removes the ' (E)' or ' (P)' suffix from a string if it exists
    params:
        input_str: a string representing an unprocessed match label
    """
    if input_str.endswith(' (E)') or input_str.endswith(' (P)'):
        return input_str[:-4]
    else:
        return input_str
    
def flip_sublabel(sublabel: str):
    """
    Helper function of process_match_label. Flips a string of the form 'x - y' to 'y - x'
    params:
        sublabel: a string of the form 'x - y'
    """
    x, y = sublabel.split(' - ')
    return f'{y} - {x}'

def process_match_label(current_match: RemoteMatchData):
    """
    Creates a match label such that current_match.team1 is the team first listed on the label
    params:
        current_match: a RemoteMatchData instance storing the data for the given match
    """
    raw_label: str = current_match.details['label']
    team1_str: str = teams_map[current_match.team1]
    if raw_label.startswith(team1_str):
        return raw_label
    else:
        teams_label, score_label = raw_label.split(', ')
        teams_label = flip_sublabel(teams_label)
        score_label = flip_sublabel(score_label)
        return f'{teams_label}, {score_label}'

def generate_match_title(current_match: RemoteMatchData):
    current_match.details['label'] = remove_overtime_suffix(current_match.details['label'])
    label = process_match_label(current_match)

    teams = label.split(', ')[0]
    first_team = teams.split(' - ')[0]
    second_team = teams.split(' - ')[1]

    scores = label.split(', ')[1]
    first_score = scores.split(' - ')[0]
    second_score = scores.split(' - ')[1]

    title = f"{first_team} ({first_score}) - ({second_score}) {second_team} ({current_match.details['dateutc'].split(' ')[0]})"
    return title

def get_roles_df(model_output: torch.Tensor, player_wyids: list, get_formation: bool = False):
    """
    Given model predictions on a specific team, and given the player's wyId's,
    this function returns a dataframe containing whose columns are ['wyId', 'predictedPosition', 'predictedCategory']
    params:
        model_output: an 11x10 torch.Tensor representing a model's position predictions on a team
        player_wyids: a list of player wyId's in the given team. The order of the id's should match 
        the ordering of the players in model_output (e.g. if player 229 is represented in row 3 of model_output,
        then 229 should be the 3rd entry of player_wyids)
        get_formation: a bool indicating whether to return the formation predicted by formation_inference.get_best_formation()
    """
    formation, choice_matrix = get_best_formation(model_output, get_choice_matrix = True)

    classes = choice_matrix.argmax(dim = 1).tolist()
    roles = {player_wyids[i]: espn_label_decoder[classes[i]] for i in range(11)}
    roles_df = pd.DataFrame(roles.items(), columns = ['wyId', 'predictedPosition'])
    roles_df['predictedCategory'] = roles_df['predictedPosition'].map(POS_TO_CATEGORY)
    if get_formation:
        return roles_df, formation
    else:
        return roles_df

def get_def_graph_pos(def_roles: pd.DataFrame, pos: dict, wide_exists: bool):
    """
    Helper function of assignments_to_graph_pos. Given a dataframe of defenders and a pre-existing 
    pos dict, this function assigns positions to the defenders.
    params:
        def_roles: a dataframe containing the wyId's and position of the defensive players in a given team
        pos: a dict whose keys are player wyId's, and whose values are tuples representing 2d coords in [0, 100] x [0, 100]
        wide_exists: a bool indicating whether to assign wide positions (i.e. LB and RB)
    """
    num_defenders = len(def_roles)
    if wide_exists:
        lb_player = def_roles[def_roles['predictedPosition'] == 'LB']['wyId'].iloc[0]
        rb_player = def_roles[def_roles['predictedPosition'] == 'RB']['wyId'].iloc[0]
        pos[lb_player] = (DEF_VERTICAL+2, LEFT_HORIZONTAL)
        pos[rb_player] = (DEF_VERTICAL+2, RIGHT_HORIZONTAL)
        num_defenders -= 2

    center_y_positions = np.linspace(MIDDLE_HORIZONTAL - 30, MIDDLE_HORIZONTAL + 30, num_defenders + 2)[1:-1]
    center_players = def_roles[def_roles['predictedPosition'] == 'CB']
    for i, player_id in enumerate(center_players['wyId']):
        pos[player_id] = (DEF_VERTICAL, center_y_positions[i])

def get_mid_graph_pos(mid_roles: pd.DataFrame, pos: dict, wide_exists: bool):
    """
    Helper function of assignments_to_graph_pos. Given a dataframe of midfielders and a pre-existing 
    pos dict, this function assigns positions to the defenders.
    params:
        mid_roles: a dataframe containing the wyId's and position of the midfield players in a given team
        pos: a dict whose keys are player wyId's, and whose values are tuples representing 2d coords in [0, 100] x [0, 100]
        wide_exists: a bool indicating whether to assign wide positions (i.e. LM and RM)
    """
    num_midfielders = len(mid_roles)
    if wide_exists:
        lm_player = mid_roles[mid_roles['predictedPosition'] == 'LM']['wyId'].iloc[0]
        rm_player = mid_roles[mid_roles['predictedPosition'] == 'RM']['wyId'].iloc[0]
        pos[lm_player] = (MIDFIELD_VERTICAL, LEFT_HORIZONTAL)
        pos[rm_player] = (MIDFIELD_VERTICAL, RIGHT_HORIZONTAL)
        num_midfielders -= 2
    
    center_y_positions = np.linspace(LEFT_HORIZONTAL + 10, RIGHT_HORIZONTAL - 10, num_midfielders + 2)[1:-1]
    center_players = mid_roles[mid_roles['predictedPosition'] == 'CM']
    for i, player_id in enumerate(center_players['wyId']):
        delta = 4*(-1)**i if num_midfielders == 3 else 0
        pos[player_id] = (MIDFIELD_VERTICAL + delta, center_y_positions[i])

def get_forward_graph_pos(forward_roles: pd.DataFrame, pos: dict, wide_exists: bool):
    """
    Helper function of assignments_to_graph_pos. Given a dataframe of forwards and a pre-existing 
    pos dict, this function assigns positions to the defenders.
    params:
        forward_roles: a dataframe containing the wyId's and position of the forward players in a given team
        pos: a dict whose keys are player wyId's, and whose values are tuples representing 2d coords in [0, 100] x [0, 100]
        wide_exists: a bool indicating whether to assign wide positions (i.e. LW and RW)
    """
    num_forwards = len(forward_roles)
    if wide_exists:
        lw_player = forward_roles[forward_roles['predictedPosition'] == 'LW']['wyId'].iloc[0]
        rw_player = forward_roles[forward_roles['predictedPosition'] == 'RW']['wyId'].iloc[0]
        pos[lw_player] = (FORWARD_VERTICAL, LEFT_HORIZONTAL-3)
        pos[rw_player] = (FORWARD_VERTICAL, RIGHT_HORIZONTAL+3)
        num_forwards -= 2
    
    center_x_positions = None
    if num_forwards == 1:
        center_x_positions = [MIDDLE_HORIZONTAL]
    else:
        center_x_positions = [MIDDLE_HORIZONTAL - 15, MIDDLE_HORIZONTAL + 15]
        
    center_players = forward_roles[forward_roles['predictedPosition'] == 'CF']
    for i, player_id in enumerate(center_players['wyId']):
        pos[player_id] = (FORWARD_VERTICAL, center_x_positions[i])

def assignments_to_graph_pos(team_roles: pd.DataFrame, formation: tuple):
    """
    Given a dataframe of team roles (as produced by get_roles_df) and a formation tuple,
    this function returns a dict whose keys are player wyIds, and whose values are tuples 
    representing 2d coords in [0, 100] x [0, 100]
    params:
        team_roles: a dataframe whose columns are ['wyId', 'predictedPosition', 'predictedCategory'], 
        as produced by get_roles_df
        formation: a triple representing a soccer formation (e.g. (4,3,3))
    """
    pos = dict()
    
    gk_player = team_roles[team_roles['predictedCategory'] == 'GK']['wyId'].iloc[0]
    pos[gk_player] = (GK_VERTICAL, MIDDLE_HORIZONTAL)

    wide_def_exists = formation[0] > 3
    wide_mid_exists = formation[1] > 3
    wide_fw_exists = formation[2] > 2

    get_def_graph_pos(team_roles[team_roles['predictedCategory'] == 'DF'], pos, wide_def_exists)
    get_mid_graph_pos(team_roles[team_roles['predictedCategory'] == 'MD'], pos, wide_mid_exists)
    get_forward_graph_pos(team_roles[team_roles['predictedCategory'] == 'FW'], pos, wide_fw_exists)
    
    return pos

#----------------PyPlot----------------
def plot_pitch_pyplot(ax):
    """
    Plot soccer pitch on a given ax using matplotlib.pyplot
    """
    ax.fill_between(range(-1, 102), -1, 101, facecolor = 'green', alpha = 1)

    #Pitch Outline & Centre Line
    ax.plot([0,0],[0,100], color="white")
    ax.plot([0,100],[100,100], color="white")
    ax.plot([100,100],[100,0], color="white")
    ax.plot([100,0],[0,0], color="white")
    ax.plot([50,50],[0,100], color="white")

    #Left Penalty Area
    ax.plot([16,16],[81,19],color="white")
    ax.plot([0,16],[81,81],color="white")
    ax.plot([16,0],[19,19],color="white")

    #Right Penalty Area
    ax.plot([84,100],[81,81],color="white")
    ax.plot([84,84],[81,19],color="white")
    ax.plot([84,100],[19,19],color="white")

    #Left 6-yard Box
    ax.plot([0, 6],[63,63],color="white")
    ax.plot([6, 6],[63,37],color="white")
    ax.plot([6, 0],[37,37],color="white")

    #Right 6-yard Box
    ax.plot([100, 94],[63, 63],color="white")
    ax.plot([94, 94],[63, 37],color="white")
    ax.plot([94, 100],[37, 37],color="white")

    #Prepare Circles
    centreCircle = Ellipse((50, 50), width=30, height=39, edgecolor="white", facecolor="None", lw=1.8)
    centreSpot = Ellipse((50, 50), width=1, height=1.5, edgecolor="white", facecolor="white", lw=1.8)
    leftPenSpot = Ellipse((10, 50), width=1, height=1.5, edgecolor="white", facecolor="white", lw=1.8)
    rightPenSpot = Ellipse((90, 50), width=1, height=1.5, edgecolor="white", facecolor="white", lw=1.8)

    #Draw Circles
    ax.add_patch(centreCircle)
    ax.add_patch(centreSpot)
    ax.add_patch(leftPenSpot)
    ax.add_patch(rightPenSpot)

    #limit axis
    ax.set_xlim(-1,101)
    ax.set_ylim(-1,101)
    # Erase axes. Note that this approach allows us to keep axis labels
    ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    ax.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
    for e in ['top', 'right', 'bottom', 'left']:
        ax.spines[e].set_visible(False)
    ax.invert_yaxis()

def draw_passing_network(G: nx.DiGraph, pos: dict):
    """
    Draws an nx graph with edge opacities dependent on edge weight.
    params:
        G: an nx.DiGraph representing a team's passing network
        pos: a dict of node positions
    """
    nx.draw_networkx_nodes(G, pos)
    labels = {node: f'{name_standardizer(player_to_short_name[node])}' for node in G.nodes()}
    labels = nx.draw_networkx_labels(G, pos, labels = labels, verticalalignment = 'top', 
                                     font_color = 'white', font_size = 10)

    edge_weights = np.array([G[u][v]['weight'] for u, v in G.edges()])
    nx.draw_networkx_edges(G, pos, alpha = edge_weights/edge_weights.max(), )

def plot_match_pyplot(current_match: RemoteMatchData, position_model: PlayerClassifier | HeatmapClassifier):
    """
    Given a RemoteMatchData instance, this plots (using matplotlib.pyplot) a soccer pitch and the passing networks of
    the two teams in the given match. This first applies a given model to the current match to produce 
    predictions which are then used to produce the plots.
    params:
        current_match: a RemoteMatchData (or compatible type like MatchData) storing data for the desired match
        position_model: a PlayerClassifier or HeatmapClassifier model which classifies player positions given match data
    """
    team1_prob, team2_prob = apply_model_to_match(position_model, current_match, output_type = 'probabilities')
    team1_roles, team1_formation = get_roles_df(team1_prob, current_match.team1_players, get_formation = True)
    team2_roles, team2_formation = get_roles_df(team2_prob, current_match.team2_players, get_formation = True)

    pos_team1 = assignments_to_graph_pos(team1_roles, team1_formation)
    pos_team2 = assignments_to_graph_pos(team2_roles, team2_formation)
    pos_team2 = {player: reflect_single_pos(coords) for player, coords in pos_team2.items()}

    fig, ax = plt.subplots(figsize = (12, 7))
    plot_pitch_pyplot(ax)

    team1_g, team2_g = NxUtils.generate_nx_graph_from_match(current_match)

    draw_passing_network(team1_g, pos_team1)
    draw_passing_network(team2_g, pos_team2)

    title = f'{teams_map[current_match.team1]} vs {teams_map[current_match.team2]} - {current_match.details["dateutc"].split(" ")[0]}'

    plt.title(title)
    return fig

#----------------Plotly----------------
def process_hover_text(d: dict):
    """
    Given a dict of position prediction confidence values, 
    this produces a single string used as hover text of the form
    'Prediction confidence: POS1: VAL1 | POS2: VAL2 | POS3: VAL3 | Other: VAL4'
    params:
        d: a dict whose values are probabilities such that the sum of the dict values does not exceed 1
    """
    temp_copy = d.copy()
    prob_sum = sum(temp_copy.values())
    temp_copy['Other'] = np.round(1 - prob_sum, 2)
    output_str = 'Prediction confidence: <br>'
    for key, value in temp_copy.items():
        if value > 0:
            output_str += f'{key}: {value} <br>'
    return output_str [:-4]

def add_soccer_pitch_plotly(fig: go.Figure):
    """
    Adds a soccer pitch plot to a go.Figure object:
    params:
        fig: a go.Figure object
    """
    # Set background color (entire figure area)
    fig.update_layout(plot_bgcolor = 'green')
    
    # Fill the pitch area with green
    fig.add_shape(
        type = "rect",
        x0 = -1, y0 = -1,
        x1 = 101, y1 = 101,
        line = dict(color = 'green', width = 0),
        fillcolor = 'green',
        layer = 'below'
    )
    
    # Pitch Outline
    fig.add_shape(
        type = "rect",
        x0 = 0, y0 = 0,
        x1 = 100, y1 = 100,
        line = dict(color = 'white', width = 2),
        fillcolor = 'rgba(0,0,0,0)',
        layer = 'below'
    )
    
    # Centre Line
    fig.add_shape(
        type = "line",
        x0 = 50, y0 = 0,
        x1 = 50, y1 = 100,
        line=dict(color = 'white', width = 2),
        layer = 'below'
    )
    
    # Left Penalty Area
    fig.add_shape(
        type = "rect",
        x0 = 0, y0 = 19,  # Note: y-values reversed since you invert y-axis
        x1 = 16, y1 = 81,
        line = dict(color = 'white', width = 2),
        fillcolor = 'rgba(0,0,0,0)',
        layer = 'below'
    )
    
    # Right Penalty Area
    fig.add_shape(
        type = "rect",
        x0 = 84, y0 = 19,
        x1 = 100, y1 = 81,
        line = dict(color = 'white', width = 2),
        fillcolor = 'rgba(0,0,0,0)',
        layer = 'below'
    )
    
    # Left 6-yard Box
    fig.add_shape(
        type = "rect",
        x0 = 0, y0 = 37,
        x1 = 6, y1 = 63,
        line = dict(color = 'white', width = 2),
        fillcolor = 'rgba(0,0,0,0)',
        layer = 'below'
    )
    
    # Right 6-yard Box
    fig.add_shape(
        type = "rect",
        x0 = 94, y0 = 37,
        x1 = 100, y1 = 63,
        line = dict(color = 'white', width = 2),
        fillcolor = 'rgba(0,0,0,0)',
        layer = 'below'
    )
    
    # Centre Circle
    fig.add_shape(
        type = "circle",
        xref = "x", yref = "y",
        x0 = 35, y0 = 30.5,  # Centered at (50,50) with width=30, height=39
        x1 = 65, y1 = 69.5,
        line = dict(color = 'white', width = 2),
        fillcolor = 'rgba(0,0,0,0)',
        layer = 'below'
    )
    
    # Centre Spot
    fig.add_trace(go.Scatter(
        x = [50], y = [50],
        mode = 'markers',
        marker = dict(size = 6, color = 'white'),
        showlegend = False
    ))
    
    # Left Penalty Spot
    fig.add_trace(go.Scatter(
        x = [10], y = [50],
        mode = 'markers',
        marker = dict(size = 6, color = 'white'),
        showlegend = False
    ))
    
    # Right Penalty Spot
    fig.add_trace(go.Scatter(
        x = [90], y = [50],
        mode = 'markers',
        marker = dict(size = 6, color = 'white'),
        showlegend = False
    ))
    
    return fig

def add_edge_trace(fig: go.Figure, G: nx.DiGraph):
    """
    Adds the edges of an nx.DiGraph to a go.Figure figure.
    params:
        fig: a go.Figure object
        G: an nx.DiGraph representing a team's passing network. The graph's edges must be equipped 
        with attributes 'normalized_weight' which are values in [0, 1], 'weight' which are 
        positive integers, and whose nodes are equipped with a 'pos' attribute which are float doubles
        representing 2d coords in [0, 100] x [0, 100]
    """
    for edge in G.edges(data = True):
        source, target, data = edge
        source_name = player_to_short_name[source]
        target_name = player_to_short_name[target]
        weight = data['normalized_weight']
        num_passes = data['weight']
        
        x0, y0 = G.nodes[source]['pos']
        x1, y1 = G.nodes[target]['pos']

        x_mid = x0 + 0.7*(x1 - x0)
        y_mid = y0 + 0.7*(y1 - y0)

        # each line is split near middle, 
        # so that hover text is triggered in middle of line, and not hidden by node
        edge_trace_first = go.Scatter(
            x = [x0, x_mid, x_mid, x1], y = [y0, y_mid, y_mid, y1],
            hoverinfo = 'text',
            hovertext = f'Passes from {source_name} to {target_name}: {num_passes}',
            mode = 'lines',
            line = dict(color = 'yellow'), 
            opacity = weight,
        )
        
        fig.add_traces([edge_trace_first])

def add_node_trace(fig: go.Figure, G: nx.DiGraph, team_top_labels: dict, team_top_probs: dict):
    """
    Adds the nodes of an nx.DiGraph to a go.Figure figure.
    params:
        fig: a go.Figure object
        G: an nx.DiGraph representing a team's passing network. The graph's edges must be equipped 
        with attributes 'normalized_weight' which are values in [0, 1], 'weight' which are 
        positive integers, and whose nodes are equipped with a 'pos' attribute which are float doubles
        representing 2d coords in [0, 100] x [0, 100]
        team_top_labels: a dict whose keys are player wyId's, and whose values are 1d tensors of 3 elements, 
        each representing the top 3 labels predicted by the model for a specific player
        team_top_probs: a dict whose keys are player wyId's, and whose values are 1d tensors of 3 elements, 
        each representing the probabilities for the top 3 labels predicted by the model for a specific player
    """
    node_x = [G.nodes[node]['pos'][0] for node in G.nodes()]
    node_y = [G.nodes[node]['pos'][1] for node in G.nodes()] 
    node_text = [name_standardizer(player_to_short_name[node]) for node in G.nodes()]
    node_hover_text_dicts = []
    for node in G.nodes():
        labels = [espn_label_decoder[e] for e in team_top_labels[node].tolist()]
        probs = np.round(team_top_probs[node].tolist(), 2).tolist()
        node_hover_text_dicts.append(dict(zip(labels, probs)))
    node_hover_text = [process_hover_text(d) for d in node_hover_text_dicts]

    node_trace = go.Scatter(
        x = node_x, y = node_y,
        mode = 'markers+text',
        hoverinfo = 'text',
        hovertext = node_hover_text,
        text = node_text,
        textposition = 'top center',
        textfont = dict(
            size = 12,
            color = 'black'
        ),
        marker = dict(
            size = 10,
            color = 'lightblue',
            line = dict(width = 2, color = 'darkblue')
        )
    )
    fig.add_trace(node_trace)

def plot_match_plotly(current_match: RemoteMatchData, position_model: PlayerClassifier | HeatmapClassifier):
    """
    Given a RemoteMatchData instance, this plots (using plotly) a soccer pitch and the passing networks of
    the two teams in the given match. This first applies a given model to the current match to produce 
    predictions which are then used to produce the plots.
    params:
        current_match: a RemoteMatchData (or compatible type like MatchData) storing data for the desired match
        position_model: a PlayerClassifier or HeatmapClassifier model which classifies player positions given match data
    """
    # apply model to match, and process predictions
    team1_prob, team2_prob = apply_model_to_match(position_model, current_match, output_type = 'probabilities')
    team1_roles, team1_formation = get_roles_df(team1_prob, current_match.team1_players, get_formation = True)
    team2_roles, team2_formation = get_roles_df(team2_prob, current_match.team2_players, get_formation = True)

    # assign node coords based on model predictions 
    pos_team1 = assignments_to_graph_pos(team1_roles, team1_formation)
    pos_team2 = assignments_to_graph_pos(team2_roles, team2_formation)
    pos_team2 = {player: reflect_single_pos(coords) for player, coords in pos_team2.items()}

    # create nx graphs
    team1_g, team2_g = NxUtils.generate_nx_graph_from_match(current_match)
    nx.set_node_attributes(team1_g, pos_team1, 'pos')
    nx.set_node_attributes(team2_g, pos_team2, 'pos')
    NxUtils.add_normalized_edge_weights(team1_g)
    NxUtils.add_normalized_edge_weights(team2_g)

    # get topk predictions for node hover text
    team1_top_labels = dict(zip(current_match.team1_players, 
                                torch.topk(team1_prob, k = 3, dim = 1).indices))
    team1_top_probs = dict(zip(current_match.team1_players, 
                            torch.topk(team1_prob, k = 3, dim = 1).values))
    team2_top_labels = dict(zip(current_match.team2_players, 
                                torch.topk(team2_prob, k = 3, dim = 1).indices))
    team2_top_probs = dict(zip(current_match.team2_players, 
                            torch.topk(team2_prob, k = 3, dim = 1).values))

    
    # generate plot title
    title = generate_match_title(current_match)
    
    # initialize graph
    fig = go.Figure(layout = go.Layout(
                        hovermode = 'closest',
                        margin = dict(b = 20, l = 20, r = 20, t = 40),
                        xaxis = dict(showgrid = False, zeroline = False, showticklabels = False,
                                    range = [-1, 101]),
                        yaxis = dict(showgrid = False, zeroline = False, showticklabels = False, 
                                    range = [-1, 101], autorange = 'reversed'),
                        height = 550,
                        width = 800,
                        showlegend = False,
                        title = dict(
                            text = title
                        )
                    ))

    # add passing network elements
    add_edge_trace(fig, team1_g)
    add_edge_trace(fig, team2_g)
    add_node_trace(fig, team1_g, team1_top_labels, team1_top_probs)
    add_node_trace(fig, team2_g, team2_top_labels, team2_top_probs)
    add_soccer_pitch_plotly(fig)
    
    return fig

