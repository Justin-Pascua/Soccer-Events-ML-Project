from remote_data_handlers.remote_match_data import RemoteMatchData
from remote_data_handlers.player_details import PlayerDetails
from remote_data_handlers.metadata_handler import get_players_maps, get_teams_map, get_eventids_map
from nn_models.player_classifier import PlayerClassifier
from nn_models.heatmap_classifier import HeatmapClassifier
from nn_models.heatmap_autoencoder import HeatmapAutoencoder
from nn_models.utils import match_data_to_model_input, apply_model_to_match
from match_visualizer import plot_match_plotly, plot_match_pyplot, get_heatmap_reconstructions


import streamlit as st
import pandas as pd
from pymongo import MongoClient
import matplotlib.pyplot as plt
import plotly.express as px
import torch
import time

#-------------------DATA CACHING-------------------
# connect to MongoDB
@st.cache_resource(show_spinner = 'Connecting to database...', show_time = True)
def init_connection():
    client = None
    while True:
        try:
            client = MongoClient(st.secrets['mongo']['connection_url'])
            break
        except:
            continue
    return client

# get match data
@st.cache_data(ttl = 120, show_spinner = 'Retrieving match data...')
def get_match_data(league_string, match_wyid, _client):
    match_data = RemoteMatchData(league_string, match_wyid, _client)
    st.session_state['match_data'] = match_data

# get metadata
@st.cache_data(ttl = 60, show_spinner = 'Importing league, team, player, and event names...')
def get_metadata():
    teams_by_league = pd.read_csv('match_metadata/teams_by_league.csv')
    
    all_matches = pd.read_csv('match_metadata/all_matches.csv', index_col = 0)
    all_matches['label'] = all_matches['label'].apply(lambda x: x.encode().decode('unicode_escape'))
    all_matches['externalName'] = all_matches['externalName'].apply(lambda x: x.encode().decode('unicode_escape'))

    events_map, subevents_map = get_eventids_map(verbose = False)
    player_to_short_name, player_to_full_name, _ , _ = get_players_maps(verbose = False)
    teams_map = get_teams_map(verbose = False)
    teams_map_inverse = {value: key for key, value in teams_map.items()}

    return {'teams_by_league': teams_by_league, 'all_matches': all_matches,
            'events_map': events_map, 'subevents_map': subevents_map, 
            'player_to_short_name': player_to_short_name, 
            'player_to_full_name': player_to_full_name,
            'teams_map': teams_map, 
            'teams_map_inverse': teams_map_inverse}

# load models
@st.cache_resource(show_spinner = 'Loading neural network models...')
def get_models():
    player_model = torch.load('model_weights/player_model.pth', weights_only = False)
    player_model.eval()
    hm_model = torch.load('model_weights/heatmap_model.pth', weights_only = False)
    hm_model.eval()
    hm_ae = torch.load('model_weights/heatmap_autoencoder.pth', weights_only = False)
    hm_ae.eval()
    return player_model, hm_model, hm_ae

#-------------------CALLBACKS-------------------
def reset_user_input():
    st.session_state['selected_team1'] = None
    st.session_state['selected_team2'] = None
    st.session_state['selected_match'] = None
    st.session_state['match_data'] = None
    st.session_state['selected_player'] = None
    st.session_state['player_details'] = None

#-------------------UTILS-------------------
def generate_abbreviation(full_team_name: str):
    # replace dash with space to prepare for .split(' ')
    full_team_name = full_team_name.replace('-', ' ')
    words = full_team_name.split(' ')
    if len(words) > 1:
        abbr = ''.join([word[0] for word in words])
        return abbr
    else:
        return full_team_name[:3].upper()

def plot_arr(arr):
    fig = px.imshow(
        arr,
        zmax = torch.quantile(arr, 0.95).item()*2.75,
        color_continuous_scale = 'thermal',
        height = 250,
        width = 250,
    )
    fig.update_traces(hoverinfo = 'skip', hovertemplate = None)
    fig.update_layout(
        coloraxis_showscale = False,
        margin = dict(l = 0, r = 0, t = 0, b = 0)
    )
    fig.update_xaxes(showticklabels = False, ticks = '')
    fig.update_yaxes(showticklabels = False, ticks = '')

    return fig

#-------------------MAIN PAGE ELEMENTS-------------------
# connect to database and load models
def data_initialization():
    st.session_state.client = init_connection()
    st.session_state.update(get_metadata())
    player_model, hm_model, hm_ae = get_models()
    st.session_state.models = dict(
        player_model = player_model,
        hm_model = hm_model,
        hm_ae = hm_ae
    )

    state_vars = ['selected_comp',
                  'selected_team1',
                  'selected_team2',
                  'selected_match',
                  'selected_player',
                  'match_data',
                  'player_details']
    for var in state_vars:
        if var not in st.session_state:
            st.session_state[var] = None 
    
def league_prompt():
    # resolve internal and external competition names
    internal_comp_names = ['England', 'European_Championship', 'France', 'Germany', 'Italy', 'Spain', 'World_Cup']
    external_comp_names = ['Premier League (ENG) 2017/18', 'European Championsip 2016', 
                            'Ligue 1 (FRA) 2017/18', 'Bundesliga (GER) 2017/18', 
                            'Serie A (ITA) 2017/18', 'La Liga (SPA) 2017/18', 'World Cup 2018']
    external_to_internal_comp = dict(zip(external_comp_names, internal_comp_names))

    # prompt user for competition
    selected_comp = st.selectbox('Select competition', external_comp_names,
                                 on_change = reset_user_input,)
    selected_comp = external_to_internal_comp[selected_comp]
    
    st.session_state.selected_comp = selected_comp

def teams_prompt():
    # identify which teams are in competition selected by user
    teams_by_league = st.session_state.teams_by_league
    available_teams = sorted(teams_by_league[teams_by_league['league'] == st.session_state['selected_comp']]
                            ['teamId']
                            .map(st.session_state.teams_map)
                            .to_list())
    
    # prompt user to select 2 teams
    selected_teams = st.multiselect('Select two teams', available_teams, 
                                    help = 'Must select exactly 2 teams',
                                    on_change = reset_user_input)
    
    # if too many teams, warn user
    if len(selected_teams) > 2:
        st.error('Please select exactly 2 teams')
        st.stop()
    elif len(selected_teams) == 2:
        st.session_state['selected_team1'] = selected_teams[0]
        st.session_state['selected_team2'] = selected_teams[1]

def match_prompt(mode = 'domestic'):
    if mode == 'domestic':
        match_prompt_domestic()
    elif mode == 'international':
        match_prompt_international()

def match_prompt_domestic():
    # convert team name to wyid (int), then to str
    team1_wyid = str(st.session_state.teams_map_inverse[st.session_state['selected_team1']])
    team2_wyid = str(st.session_state.teams_map_inverse[st.session_state['selected_team2']])

    # get available matches
    all_matches: pd.DataFrame = st.session_state['all_matches']
    mask1 = all_matches['teamIds'].apply(lambda x: team1_wyid in x and team2_wyid in x)
    available_matches = all_matches[mask1]['externalName'].to_list()

    # prompt user to select a specific match
    selected_match = st.selectbox('Select specific match',
                                  available_matches, 
                                  disabled = (st.session_state['selected_team1'] is None or 
                                              st.session_state['selected_team2'] is None),
                                  help = 'Select from one of two matches played between the two teams')
    
    # get wyid of match
    if selected_match:
        mask2 = all_matches['externalName'] == selected_match
        match_wyid = all_matches[mask2]['wyId'].item()
        st.session_state['selected_match'] = match_wyid

def match_prompt_international():
    all_matches: pd.DataFrame = st.session_state['all_matches']
    selected_comp = st.session_state.selected_comp
    mask1 = all_matches['league'] == selected_comp
    available_matches = all_matches[mask1]['externalName'].to_list()

    selected_match = st.selectbox('Select specific match',
                                  available_matches,
                                  disabled = (st.session_state['selected_comp'] is None),
                                  help = 'Select a specific match from the tournament')
    if selected_match:
        mask2 = all_matches['externalName'] == selected_match
        match_wyid = all_matches[mask2]['wyId'].item()
        st.session_state['selected_match'] = match_wyid

def match_output():
    if st.session_state['match_data'] is None:
        st.write('Awaiting match selection')
        return
    
    st.header("Passing Networks")
    current_match = st.session_state['match_data']
    hm_model = st.session_state.models['hm_model']
    st.plotly_chart(plot_match_plotly(current_match, hm_model),
                    width = 'content',
                    theme = None)    
    st.caption('Hover over players/connections for more info! Double click on graph to reset view.')

def player_prompt():
    if st.session_state['match_data'] is None:
        st.write('Please select a match before selecting a player')
    else:
        current_match: RemoteMatchData = st.session_state['match_data']
        player_to_short_name = st.session_state['player_to_short_name']
        teams_map = st.session_state['teams_map']

        team1_name = generate_abbreviation(teams_map[current_match.team1])
        team2_name = generate_abbreviation(teams_map[current_match.team2])

        # get all player names from match
        team1_player_names = [f'({team1_name}) {player_to_short_name[wyid]}' for wyid in current_match.team1_players]
        team2_player_names = [f'({team2_name}) {player_to_short_name[wyid]}' for wyid in current_match.team2_players]
        all_player_names = team1_player_names + team2_player_names

        all_player_wyids = current_match.team1_players + current_match.team2_players
        current_name_to_wyid_map = dict(zip(all_player_names, all_player_wyids))
        all_player_names = sorted(all_player_names)

        # prompt user to select player
        selected_player_name = st.selectbox('Select player', 
                                            all_player_names,
                                            disabled = (st.session_state['match_data'] is None))
        selected_player_wyid = current_name_to_wyid_map[selected_player_name]
        st.session_state['selected_player'] = selected_player_wyid

def player_output():
    if st.session_state['selected_player'] is None:
        st.write('Awaiting player selection')
        return    
    if st.session_state['player_details'] is None:
        current_match = st.session_state['match_data']
        hm_model = st.session_state['models']['hm_model']
        hm_ae = st.session_state['models']['hm_ae']
        st.session_state['player_details'] = PlayerDetails(current_match, hm_model, hm_ae)
    
    player_wyid = st.session_state['selected_player']
    player_details: PlayerDetails = st.session_state['player_details']

    st.markdown(f'## **Player Details**: {st.session_state['player_to_short_name'][player_wyid]}')
    
    tab_names = ['Position', 'Events', 'Events Map', 'Heatmap Inference']
    predictions_tab, event_counts_tab, event_dots_tab, heatmap_tab = st.tabs(tab_names)
    with predictions_tab:
        position_probabilities = player_details.get_position_probs(player_wyid)
        probs_df = pd.DataFrame(data = [position_probabilities.keys(), 
                                        position_probabilities.values()]
                                ).transpose()
        fig = px.bar(probs_df, x = 1, y  = 0, 
                     orientation = 'h', 
                     height = 250, 
                     width = 100,
                     color_discrete_sequence = ['#8A2BE2']
                     )
        fig.update_layout(
            plot_bgcolor = 'rgba(0,0,0,0)',
            paper_bgcolor = 'rgba(0,0,0,0)',
            yaxis_title = 'Position',
            xaxis_title = 'Probability',
            margin = dict(l = 0, r = 0, t = 0, b = 0),
        )
        fig.update_xaxes(color = 'white')
        fig.update_yaxes(color = 'white')
        st.plotly_chart(fig)
        st.caption("""The player's postion is predicted by feeding a map of their 
                   in-match actions through a neural network. The input fed into the
                   neural network can be found in the "Events Map" tab.""")
        st.caption("""Note that the position occupied by the player within the plot in 
                   the center of your screen may differ from the position with the 
                   highest probability. For more details about how positions are decided, 
                   please refer to the "Methodology" page. """)
    with event_counts_tab:
        event_counts = player_details.get_event_counts(player_wyid)
        st.dataframe(event_counts)
        st.caption("""This table lists the number of times the given player 
                   commits an "event" (e.g. a pass, a shot, etc.)""")
    with event_dots_tab:
        event_dots = player_details.get_event_dots(player_wyid)
        fig = plot_arr(event_dots)
        st.plotly_chart(fig)
        st.caption("""Above is a map showing where the given player committed events on 
                   the pitch.""")
        st.caption("""Here, we've adjusted the orientation of the map to match the 
                   orientation of the pitch in the center of your screen. In other 
                   words, players for the team on the right-hand side have had their maps 
                   rotated by 180 degrees about the center of the pitch. Internally, when 
                   maps are fed into the model, maps are oriented such that the player's 
                   own goal is on the left-hand side of the map.""")
    with heatmap_tab:
        player_hm = player_details.get_heatmap(player_wyid)
        fig = plot_arr(player_hm)
        st.plotly_chart(fig)
        st.caption("""This is a reconstruction of the player's heatmap generated
                   by feeding the events map in the previous tab into an autoencoder. """)
        st.caption("""As noted in the previous tab, we've oriented this heatmap in order to
                   match the orientation of the pitch in the center of the screen.""")
           
def main_display():
    input_col, match_output_col, player_output_col = st.columns([1, 2.5, 1])
    match_input_cell = input_col.container(
        border = True, 
        height = 'content', 
        vertical_alignment = 'top'
    )
    player_input_cell = input_col.container(
        border = True, 
        height = 'content', 
        vertical_alignment = 'top'
    )
    match_output_cell = match_output_col.container(
        border = False, 
        height = 'content', 
        width = 'stretch', 
        vertical_alignment = 'center', 
        horizontal_alignment = 'center'
    )
    player_output_cell = player_output_col.container(
        border = True, 
        height = 'content', 
        width = 'stretch', 
        vertical_alignment = 'top', 
        horizontal_alignment = 'center'
    )

    with match_input_cell:
        st.subheader('**Match Selection**')

        league_prompt()
        
        if st.session_state['selected_comp'] is None:
            pass
        elif st.session_state['selected_comp'] in ['European_Championship', 'World_Cup']:
            match_prompt(mode = 'international')
        else:
            teams_prompt()

        if st.session_state['selected_comp'] not in ['European_Championship', 'World_Cup']:
            if st.session_state['selected_team1'] is not None and st.session_state['selected_team2'] is not None:
                match_prompt(mode = 'domestic')

        apply = st.button('Apply Selection', 
                      on_click = get_match_data,
                      args = (st.session_state['selected_comp'],
                              st.session_state['selected_match'],
                              st.session_state['client']),
                      disabled = st.session_state['selected_match'] is None,
                      help = 'Please fill the above fields to select a match')

    with player_input_cell:
        st.subheader('**Player Selection**')
        player_prompt()

    with match_output_cell:
        match_output()

    with player_output_cell:
        player_output()

#-------------------EXECUTION-------------------
st.set_page_config(
    layout = 'wide',
    initial_sidebar_state = 'collapsed',
)
with st.sidebar:
    st.markdown(""":material/attribution: :grey[Justin Pascua - 2025]  
                :material/work: :grey[[LinkedIn](https://www.linkedin.com/in/justin-pascua-673686187/)]  
                :material/code: :grey[[GitHub](https://github.com/Justin-Pascua)]""")
st.title("Soccer Match Visualizer")
data_initialization()
main_display()