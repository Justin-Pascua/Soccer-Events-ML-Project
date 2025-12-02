from remote_data_handlers.remote_match_data import RemoteMatchData
from local_data_handlers.wyscout_metadata_handler import get_players_maps, get_teams_map
from nn_models.player_classifier import PlayerClassifier
from nn_models.heatmap_classifier import HeatmapClassifier
from nn_models.heatmap_autoencoder import HeatmapAutoencoder
from match_visualizer import plot_match_plotly, plot_match_pyplot
from nn_models.utils import match_data_to_model_input, apply_model_to_match

import streamlit as st
import pandas as pd
from pymongo import MongoClient
import matplotlib.pyplot as plt
import torch
import time

#-------------------DATA CACHING-------------------
# connect to MongoDB
@st.cache_resource
def init_connection():
    placeholder = st.empty()
    placeholder.write('Connecting to database')
    client = None
    while True:
        try:
            client = MongoClient(st.secrets['mongo']['connection_url'])
            placeholder.write('Successfully connected to database!')
            time.sleep(2)
            placeholder.empty()
            break
        except:
            placeholder.write('Failed to connect to database, attempting again')
    return client

# get match data
@st.cache_data(ttl = 120)
def get_match_data(league_string, match_wyid, _client):
    match_data = RemoteMatchData(league_string, match_wyid, _client)
    st.session_state.match_data = match_data

# get metadata
@st.cache_data(ttl = 60)
def get_metadata():
    teams_by_league = pd.read_csv('match_metadata/teams_by_league.csv')
    all_matches = pd.read_csv('match_metadata/all_matches.csv')
    player_to_short_name, player_to_full_name, _ , _ = get_players_maps(verbose = False)
    teams_map = get_teams_map(verbose = False)
    teams_map_inverse = {value: key for key, value in teams_map.items()}

    return {'teams_by_league': teams_by_league, 'all_matches': all_matches, 
            'player_to_short_name': player_to_short_name, 
            'player_to_full_name': player_to_full_name,
            'teams_map': teams_map, 
            'teams_map_inverse': teams_map_inverse}

# load models
@st.cache_resource
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
    st.session_state.team1 = None
    st.session_state.team2 = None
    st.session_state.match_wyid = None

#-------------------FRAGMENTS-------------------
# connect to database and load models
@st.fragment()
def data_initialization():
    st.session_state.client = init_connection()
    st.session_state.update(get_metadata())
    player_model, hm_model, hm_ae = get_models()
    st.session_state.models = dict(
        player_model = player_model,
        hm_model = hm_model,
        hm_ae = hm_ae
    )
    
# prompt user for match, display passing networks
@st.fragment()
def display_passing_networks():
    # initialize session variables for user input
    for x in ['league', 'team1', 'team2', 'matchday', 'match_wyid']:
        if x not in st.session_state:
            st.session_state[x] = None
    cols = st.columns(3)
    # select competition
    with cols[0]:
        # resolve internal 
        internal_comp_names = ['England', 'European_Championship', 'France', 'Germany', 'Italy', 'Spain', 'World_Cup']
        external_comp_names = ['Premier League (ENG) 2017/18', 'European Championsip 2016', 
                               'Ligue 1 (FRA) 2017/18', 'Bundesliga (GER) 2017/18', 
                               'Serie A (ITA) 2017/18', 'La Liga (SPA) 2017/18', 'World Cup 2018']
        external_to_internal_comp = dict(zip(external_comp_names, internal_comp_names))

        # prompt user for competition
        selected_comp = st.selectbox('Select competition', external_comp_names,
                                     on_change = reset_user_input)
        selected_comp = external_to_internal_comp[selected_comp]
        
        st.session_state['league'] = selected_comp
    # select teams
    with cols[1]:
        if st.session_state['league'] is not None:
            # identify which teams are in competition selected by user
            teams_by_league = st.session_state.teams_by_league
            available_teams = sorted(teams_by_league[teams_by_league['league'] == st.session_state['league']]
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
                st.session_state.team1 = selected_teams[0]
                st.session_state.team2 = selected_teams[1]
    # select match
    with cols[2]:
        if st.session_state['team1'] is not None and st.session_state['team2'] is not None:
            # convert team name to wyid (int), then to str
            team1_wyid = str(st.session_state.teams_map_inverse[st.session_state['team1']])
            team2_wyid = str(st.session_state.teams_map_inverse[st.session_state['team2']])

            # get available matches
            mask1 = st.session_state.all_matches['teamIds'].apply(lambda x: team1_wyid in x and team2_wyid in x)
            available_matches = st.session_state.all_matches[mask1]['externalName'].to_list()
            available_matches = [string.encode('latin-1').decode('latin-1') for string in available_matches]

            # prompt user to select a specific match
            selected_matchday = st.selectbox('Select specific match',
                                             available_matches, 
                                             disabled = (st.session_state.team1 is None or st.session_state.team2 is None))
            
            # get wyid of match
            if selected_matchday:
                mask2 = st.session_state.all_matches['externalName'] == selected_matchday
                match_wyid = st.session_state.all_matches[mask2]['wyId'].item()
                st.session_state.match_wyid = match_wyid
    
    st.button('Apply', 
              on_click = get_match_data,
              args = (st.session_state.league,
                      st.session_state.match_wyid,
                      st.session_state.client),
              disabled = st.session_state.match_wyid is None)

    if 'match_data' not in st.session_state:
        st.session_state.match_data = None
    elif st.session_state.match_data is not None:
        match_data = st.session_state.match_data
        model = st.session_state.models['hm_model']
        st.plotly_chart(plot_match_plotly(match_data, model),
                        width = 'content',
                        theme = None)
        

# prompt user for specific player
@st.fragment()
def user_prompt_for_player():
    # get necessary items from session state
    current_match: RemoteMatchData = st.session_state.match_data
    player_to_short_name = st.session_state.player_to_short_name
    teams_map = st.session_state.teams_map

    # get all player names from match
    all_player_wyids = current_match.team1_players + current_match.team2_players
    all_player_names = [player_to_short_name[wyid] for wyid in all_player_wyids]
    current_name_to_wyid_map = dict(zip(all_player_names, all_player_wyids))
    all_player_names = sorted(all_player_names)

    # prompt user to select player
    selected_player_name = st.selectbox('Select player', all_player_names)
    selected_player_wyid = current_name_to_wyid_map[selected_player_name]

    relative_index = all_player_wyids.index(selected_player_wyid)
    print(relative_index)

    st.write(selected_player_name)
    st.write(selected_player_wyid)
    st.session_state.selected_player_relative_index = relative_index  
    display_player_stats()  

# display player's stats and heatmap
@st.fragment()
def display_player_stats():
    # get current match and heatmap autoencoder
    current_match: RemoteMatchData = st.session_state.match_data
    hm_ae = st.session_state.models['hm_ae']

    # get events map
    team1_input, team2_input = match_data_to_model_input(current_match)
    team1_hm = team1_input[0]
    team2_hm = team2_input[0]

    # get heatmaps
    team1_reconstructed_hm = hm_ae(team1_hm).detach()
    team2_reconstructed_hm = hm_ae(team2_hm).detach()

    if 'selected_player_relative_index' not in st.session_state:
        st.session_state.selected_player_relative_index = None
    if st.session_state.selected_player_relative_index is not None:
        relative_index = st.session_state.selected_player_relative_index
        fig, ax = plt.subplots()
        player_hm = None
        if relative_index < 11:
            player_hm = team1_reconstructed_hm[relative_index].squeeze()
        else:
            player_hm = team2_reconstructed_hm[relative_index - 11].squeeze()
        ax.imshow(player_hm, vmax = torch.quantile(player_hm, 0.95).item()*2.75)
        ax.axis('off')
        st.pyplot(fig)

data_initialization()
display_passing_networks()
print(st.session_state.get('match_wyid', 'Not found'))
if 'match_wyid' not in st.session_state:
    st.session_state.match_wyid = None
if st.session_state.match_wyid is not None:
    user_prompt_for_player()