from remote_data_handlers.remote_match_data import RemoteMatchData
from local_data_handlers.wyscout_metadata_handler import get_players_maps, get_teams_map
from nn_models.player_classifier import PlayerClassifier
from nn_models.heatmap_classifier import HeatmapClassifier
from nn_models.heatmap_autoencoder import HeatmapAutoencoder
from match_visualizer import plot_match
from nn_models.utils import match_data_to_model_input

import streamlit as st
import pandas as pd
from pymongo import MongoClient
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
    return match_data

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
def apply_model():
    st.session_state.match_data = get_match_data(
        st.session_state.league,
        st.session_state.match_wyid,
        st.session_state.client
    )

#-------------------FRAGMENTS-------------------
# connect to database and load models
@st.fragment()
def data_initialization():
    st.session_state.client = init_connection()
    st.session_state.update(get_metadata())
    st.session_state.models = get_models()
    

# prompt user for league, teams, and matchday
@st.fragment()
def user_prompt():
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
        selected_comp = st.selectbox('Select competition', external_comp_names)
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
            selected_teams = st.multiselect('Select two teams', available_teams, help = 'Must select exactly 2 teams')
            
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
            available_matches = [string.encode('utf-8').decode('latin-1') for string in available_matches]

            # prompt user to select a specific match
            selected_matchday = st.selectbox('Select specific match',
                                             available_matches)

            # get wyid of match
            mask2 = st.session_state.all_matches['externalName'] == selected_matchday
            match_wyid = st.session_state.all_matches[mask2]['wyId'].item()
            st.session_state.match_wyid = match_wyid
    
# run models and vis result
def display_model_outputs():
    if 'match_data' not in st.session_state:
        st.session_state.match_data = None
    
    apply = st.button('Apply', on_click = apply_model)
    if apply:
        st.pyplot(plot_match(st.session_state.match_data, st.session_state.models[1]))
    



data_initialization()
user_prompt()
if st.session_state['match_wyid'] is not None:
    display_model_outputs()