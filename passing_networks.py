from local_data_handlers.wyscout_metadata_handler import get_metadata_maps 
from local_data_handlers.wyscout_data_handler import CompetitionClient, MatchData
import networkx as nx
import numpy as np
import pandas as pd
import time
import torch
from torch_geometric.data import Data
import math

from classifiers.heatmap_classifier import HeatmapMLP, CustomTransforms, CustomData

wyscout_metadata_maps = get_metadata_maps(verbose = False)
player_to_short_name = wyscout_metadata_maps['player_to_short_name']
player_to_pos = wyscout_metadata_maps['player_to_pos']
events_map = wyscout_metadata_maps['events_map']
subevents_map = wyscout_metadata_maps['subevents_map']

def draw_passing_network(g: nx.DiGraph, with_names: bool = True):
    """
    Draws the given passing network.
    params:
        g: an nx.DiGraph with nodes labeled by player wyIds and edges weighted by pass frequency
        with_names: a bool determining whether or not to label nodes with player names and positions
    """
    pos = nx.spring_layout(g)
    nx.draw_networkx_nodes(g, pos)

    if(with_names):
        labels = {node: f'{player_to_short_name[node]}\n{player_to_pos[node]}' for node in g.nodes()}
        nx.draw_networkx_labels(g, pos, labels = labels)
    else:
        nx.draw_networkx_labels(g, pos)

    edge_weights = np.array([g[u][v]['weight'] for u, v in g.edges()])
    nx.draw_networkx_edges(g, pos, alpha = edge_weights/edge_weights.max())

# NetworkX functions
class NxPassingNetworks:
    @staticmethod
    def generate_nx_graph_from_match(match_data: MatchData):
        """
        Generates 2 passing networks, which are weighted directed graphs, from the current match.
        Returns the two graphs as g1, g2
        params:
            match_data: a MatchData instance
        """
        team1_id = match_data.team1
        team1_nodes = match_data.team1_players
        team1_edges = dict()

        team2_id = match_data.team2
        team2_nodes = match_data.team2_players
        team2_edges = dict()

        team_to_nodes = {team1_id: team1_nodes, team2_id: team2_nodes}
        team_to_edges = {team1_id: team1_edges, team2_id: team2_edges}

        # below, we extract all rows corresponding to accurate passes and the rows immediately proceeding an accurate pass 
        temp = match_data.events_df[['eventId', 'tags', 'playerId', 'teamId']].copy()
        temp['isAccPass'] = temp['tags'].apply(lambda x: 1801 in x) & (temp['eventId'] == 8)    # flag if accurate pass
        temp['prevWasAccPass'] = temp['isAccPass'].shift(1)                                     # propagate flag to next row down
        temp = temp[temp['isAccPass'] | temp['prevWasAccPass']].reset_index(drop = True)        # filter by flags

        # get passing edges
        for i in range(len(temp) - 1):
            window = temp.iloc[i:i+2]
            # check if accurate pass
            if(window.iloc[0]['isAccPass'] != True):
                continue

            # check if on same team
            if(window.iloc[0]['teamId'] != window.iloc[1]['teamId']):
                continue

            sender = window.iloc[0]['playerId']
            receiver = window.iloc[1]['playerId']    
            
            # check if players are in starting 11
            team_num = window.iloc[0]['teamId']
            if((sender in team_to_nodes[team_num]) and (receiver in team_to_nodes[team_num])):
                try:
                    team_to_edges[team_num][(sender, receiver)] += 1
                except:
                    team_to_edges[team_num][(sender, receiver)] = 1

        graph_team_1 = nx.DiGraph()
        for key, value in team1_edges.items():
            graph_team_1.add_edge(key[0], key[1], weight = value)

        graph_team_2 = nx.DiGraph()
        for key, value in team2_edges.items():
            graph_team_2.add_edge(key[0], key[1], weight = value)

        return graph_team_1, graph_team_2

    @staticmethod
    def generate_nx_graphs_from_competition(current_competition: CompetitionClient, verbose: bool = True):
        """
        Generates passing networks as nx.DiGraph objects for all matches in the current competition. 
        Returns the a dictionary whose keys are the matchIds, and whose values are a list containing the graphs for the 2 teams in a given match
        params:
            current_competition: a CompetitionClient instance
            verbose: a bool determining whether or not to print progress
        """
        start_time = time.perf_counter()
        
        # initialize dict to store result
        graphs = dict()

        events_df = current_competition.events_df

        # tag by accuracy
        events_df['isAcc'] = events_df['tags'].apply(lambda x: 1801 in x)

        # add fields used to get edge weights
        events_df['sender'] = events_df['playerId']
        events_df['senderTeamId'] = events_df['teamId']
        events_df['receiver'] = events_df['playerId'].shift(-1).astype('Int64')
        events_df['receiverTeamId'] = events_df['teamId'].shift(-1).astype('Int64')

        # get passing events used to create passing netwrosk
        passes_df = events_df[                  
            events_df['isAcc'] &                                        # filter only accurate events
            (events_df['eventId'] == 8) &                               # filter only pass events
            (events_df['senderTeamId'] == events_df['receiverTeamId'])  # filter only passes sent and received by players on same team
        ][['matchId', 'teamId', 'sender', 'receiver']]

        # get edge weights by computing number of passes between distinct pairs of players
        pass_counts = passes_df.groupby(['matchId', 'teamId'])[['sender', 'receiver']].value_counts()


        # MATCH SPECIFIC OPERATIONS
        match_ids = current_competition.match_details.keys()
        for i, match_id in enumerate(list(match_ids)):
            intermediate_time = time.perf_counter()
            if(verbose and (i+1)%20 == 0):
                print(f"\tProcessed {i+1} matches ({intermediate_time - start_time:.3f} secs)")
            
            current_match = MatchData(current_competition, match_id)
            
            # quick aliases for match-specific information 
            match_id = current_match.match_id

            team1_id = current_match.team1
            team2_id = current_match.team2

            team1_players = current_match.team1_players
            team2_players = current_match.team2_players

            # get info about passes in match
            # Note: the last two components of the multi-index filter out any players not on the starting 11
            team1_pass_info = pass_counts.loc[match_id, team1_id, team1_players, team1_players]
            team2_pass_info = pass_counts.loc[match_id, team2_id, team2_players, team2_players]

            # get edge weights
            team1_edge_attr = np.array(team1_pass_info.values).reshape(-1,1)
            team2_edge_attr = np.array(team2_pass_info.values).reshape(-1,1)    

            # get edge indices
            team1_edge_index = np.array(list(team1_pass_info.droplevel(level = [0, 1]).index))
            team2_edge_index = np.array(list(team2_pass_info.droplevel(level = [0, 1]).index))

            team1_weighted_edges = np.concat([team1_edge_index, team1_edge_attr], axis = 1)
            team2_weighted_edges = np.concat([team2_edge_index, team2_edge_attr], axis = 1)

            team1_graph = nx.DiGraph()
            team2_graph = nx.DiGraph()

            team1_graph.add_weighted_edges_from(team1_weighted_edges)
            team2_graph.add_weighted_edges_from(team2_weighted_edges)

            graphs[match_id] = [team1_graph, team2_graph]

        return graphs

    @staticmethod
    def get_all_graphs_as_nx(verbose: bool = True, connection_string: str = "mongodb://localhost:27017/"):
        """
        Generates NetworkX graphs for all matches and all competitions.  
        Returns a dictionary called "graphs," which is a nested dictionary which
        can be indexed as follows:
        graphs[competition_string][match_id][index_in_01]
        params:
            verbose: a bool indicating whether or not to print progress
            connection_string: connection string used to initialize CompetitionClient (which is a wrapper for MongoClient).
                            Default value is the connection string used to connect to locally hosted server 
        """
        competitions = ['England', 'European_Championship', 'France', 'Germany', 'Italy', 'Spain', 'World_Cup']

        start_time = time.perf_counter()
        graphs = dict()
        for competition in competitions:
            if(verbose):
                print(f"Processing {competition}")
            
            current_client = CompetitionClient(connection_string, competition)
            
            graphs[competition] = NxPassingNetworks.generate_nx_graphs_from_competition(current_client, verbose = verbose)
            
            intermediate_time = time.perf_counter()
            if(verbose):
                print(f"Finished graphs for {competition} ({intermediate_time - start_time:.3f} secs)")
                print("="*50) 
        end_time = time.perf_counter()
        if(verbose):
            print(f"Execution time: {end_time - start_time:.3f} seconds")
        
        return graphs


# PyG functions
X_DTYPE = torch.float32
EDGE_WEIGHT_DTYPE = torch.float32
TARGET_DTYPE = torch.long

class PygPassingNetworks:
    @staticmethod
    def get_node_features_df(events_df):
        temp_df = events_df.copy()

        # filter out interruptions and offsides
        temp_df = temp_df[~ temp_df['eventId'].isin([5,6])]

        # get coords
        temp_df['initX'] = temp_df['positions'].apply(lambda x: x[0]['x'])
        temp_df['initY'] = temp_df['positions'].apply(lambda x: x[0]['y'])

        # drop rows with misinputted coords (i.e. those at the corners (0,0), (0, 100), (100, 0), (100, 100))
        temp_df = temp_df[~((temp_df['initX'] == 0) & (temp_df['initX'] == 0))]     # get rid of points recorded at (0, 0)
        temp_df = temp_df[~((temp_df['initX'] == 0) & (temp_df['initX'] == 100))]   # get rid of points recorded at (0, 100)
        temp_df = temp_df[~((temp_df['initX'] == 100) & (temp_df['initX'] == 0))]   # get rid of points recorded at (100, 0)
        temp_df = temp_df[~((temp_df['initX'] == 100) & (temp_df['initX'] == 100))] # get rid of points recorded at (100, 100)

        # get subevent count features
        temp_df['subEventId'] = temp_df['subEventId'].astype('Int64')
        subevent_counts = temp_df.groupby(['matchId', 'teamId', 'playerId'])['subEventId'].value_counts().unstack(fill_value = 0)

        # get position summary features
        position_summary = temp_df.groupby(['matchId', 'teamId', 'playerId'])[['initX', 'initY']].agg({
            'initX': ['min', 'max', 'mean', 'median', 'std', 'skew'],            
            'initY': ['min', 'max', 'mean', 'median', 'std', 'skew'],
        })

        # compute means for columns that likely have missing values
        values = {(upper, lower): position_summary[upper][lower].mean() 
                for upper in ['initX', 'initY'] 
                for lower in ['std', 'skew']}
        # fill na values with means of respective column
        position_summary.fillna(value = values, inplace = True)

        # collapse multileveled columns
        position_summary.columns = [' '.join(col).strip() for col in position_summary.columns.values]

        # merge dataframes by multindices
        node_feature_df = pd.merge(subevent_counts, position_summary, left_index = True, right_index = True)

        return node_feature_df

    @staticmethod
    def get_edge_features_df(events_df):
        temp_df = events_df.copy()

        # add fields used to determine edge weights
        temp_df['sender'] = temp_df['playerId']
        temp_df['senderTeamId'] = temp_df['teamId']
        temp_df['receiver'] = temp_df['playerId'].shift(-1).astype('Int64')
        temp_df['receiverTeamId'] = temp_df['teamId'].shift(-1).astype('Int64')

        # mark accurate events
        temp_df['isAcc'] = temp_df['tags'].apply(lambda x: 1801 in x)

        # get passing events used to create passing networks
        passes_df = temp_df[                  
            temp_df['isAcc'] &                                        # filter only accurate events
            (temp_df['eventId'] == 8) &                               # filter only pass events
            (temp_df['senderTeamId'] == temp_df['receiverTeamId'])  # filter only passes sent and received by players on same team
        ][['matchId', 'teamId', 'sender', 'receiver', 'positions']]

        # compute pass distances and inverse
        passes_df['dist'] = passes_df['positions'].apply(
            lambda points: math.sqrt((points[0]['x'] - points[1]['x'])**2 + (points[0]['y'] - points[1]['y'])**2)
        )
        passes_df['inverseDist'] = 1/(passes_df['dist'] + 1)    # add 1 to denom to avoid div by zero

        # get edge features
        pass_strengths = passes_df.groupby(['matchId', 'teamId', 'sender', 'receiver'])['inverseDist'].mean()  # mean of inverseDists across pairs of players
        pass_counts = passes_df.groupby(['matchId', 'teamId'])[['sender', 'receiver']].value_counts()          # number of passes between pairs of players

        # merge pass strength and pass count info by matching multi-index
        edge_features_df = pd.merge(pass_strengths, pass_counts, left_index = True, right_index = True)

        # for each pass sender, keep only the top 3 pass receivers (i.e. the 3 players that the sender passes to most often)
        edge_features_df = (edge_features_df
                            .groupby(['matchId', 'teamId', 'sender'], as_index = False)
                            .apply(lambda x: x.nlargest(3, columns = ['count']))
                            .reset_index(level = 0, drop = True))

        return edge_features_df

    @staticmethod
    def generate_pyg_graph_from_match(current_match: MatchData):
        """
        Generates 2 passing networks, which are weighted directed graphs, from the current match.
        Returns the two graphs as g1, g2
        params:
            current_match: a MatchData instance
        """
        events_df = current_match.events_df

        # tag by accuracy. Note: this is used for both node and edge features
        events_df['isAcc'] = events_df['tags'].apply(lambda x: 1801 in x)

        # add fields used to determine edge weights
        events_df['sender'] = events_df['playerId']
        events_df['senderTeamId'] = events_df['teamId']
        events_df['receiver'] = events_df['playerId'].shift(-1).astype('Int64')
        events_df['receiverTeamId'] = events_df['teamId'].shift(-1).astype('Int64')

        # get passing events used to create passing netwrosk
        passes_df = events_df[                  
            events_df['isAcc'] &                                        # filter only accurate events
            (events_df['eventId'] == 8) &                               # filter only pass events
            (events_df['senderTeamId'] == events_df['receiverTeamId'])  # filter only passes sent and received by players on same team
        ][['matchId', 'teamId', 'sender', 'receiver']]

        # get edge weights by computing number of passes between distinct pairs of players
        pass_counts = passes_df.groupby(['matchId', 'teamId'])[['sender', 'receiver']].value_counts()

        # get events used for computing node features
        events_for_nodes = events_df[events_df['eventId'].isin([1,8,10])]

        # add fields for position features
        # unpack first coord
        events_for_nodes['initialX'] = events_for_nodes['positions'].apply(lambda x: x[0]['x'])
        events_for_nodes['initialY'] = events_for_nodes['positions'].apply(lambda x: x[0]['y'])
        # unpack second coord. If none given, use first coord 
        # Note: for some reason, there are 2 anomalous pass/duel/shots events in this dataset do not have a 2nd coord
        events_for_nodes['finalX'] = events_for_nodes['positions'].apply(lambda x: x[1]['x'] if (len(x) == 2) else x[0]['x'])
        events_for_nodes['finalY'] = events_for_nodes['positions'].apply(lambda x: x[1]['y'] if (len(x) == 2) else x[0]['y'])
        

        # add fields used for accuracy features
        events_for_nodes['eventAndAcc'] = events_for_nodes.apply(
            lambda row: f"{'acc' if row['isAcc'] else 'inacc'}{events_map[row['eventId']]}",
            axis = 1
        )

        # compute node features
        # summary statistics for coordinates
        position_summary = events_for_nodes.groupby(['matchId', 'teamId', 'playerId']).agg({
            'initialX': ['mean', 'std', 'skew'],            
            'initialY': ['mean', 'std', 'skew'],
            'finalX': ['mean', 'std', 'skew'],
            'finalY': ['mean', 'std', 'skew']
        })
        # counts for accurate/inaccurate duels/passes/shots
        accuracy_summary = (events_for_nodes
                            .groupby(['matchId', 'teamId', 'playerId'])['eventAndAcc']
                            .value_counts()
                            .unstack(fill_value = 0))

        # MATCH SPECIFIC OPERATIONS
        # quick aliases for match-specific information 
        match_id = current_match.match_id

        team1_id = current_match.team1
        team2_id = current_match.team2

        team1_players = current_match.team1_players
        team2_players = current_match.team2_players

        team1_encoder = current_match.team1_encoder
        team2_encoder = current_match.team2_encoder

        # get info about passes in match
        # Note: the last two components of the multi-index filter out any players not on the starting 11
        team1_pass_info = pass_counts.loc[match_id, team1_id, team1_players, team1_players]
        team2_pass_info = pass_counts.loc[match_id, team2_id, team2_players, team2_players]

        # get edge weights
        team1_edge_attr = torch.tensor(team1_pass_info.values).reshape(-1,1)
        team2_edge_attr = torch.tensor(team2_pass_info.values).reshape(-1,1)    

        # get edge indices and map playerId's to node indices in {0, ..., 10}
        team1_edge_index = torch.tensor(
            team1_encoder.transform(
                team1_pass_info.droplevel(level = [0, 1]).reset_index()[['sender', 'receiver']].values
                .flatten(order = 'F')   # flatten so that labelEncoder can be applied. 
                ).reshape(2, -1))       # used Fortran order above so that original order can be recovered after transforming
        team2_edge_index = torch.tensor(
            team2_encoder.transform(
                team2_pass_info.droplevel(level = [0, 1]).reset_index()[['sender', 'receiver']].values
                .flatten(order = 'F')    
                ).reshape(2, -1)) 

        # get accuracy features
        team1_accuracy_summary = accuracy_summary.loc[match_id, team1_id, team1_players]
        team2_accuracy_summary = accuracy_summary.loc[match_id, team2_id, team2_players]

        # get position summary
        team1_position_summary = position_summary.loc[match_id, team1_id, team1_players]
        team2_position_summary = position_summary.loc[match_id, team2_id, team2_players]

        # get all node features
        team1_x = torch.cat([torch.tensor(team1_accuracy_summary.values), 
                            torch.tensor(team1_position_summary.values)], dim = 1)
        team2_x = torch.cat([torch.tensor(team2_accuracy_summary.values), 
                            torch.tensor(team2_position_summary.values)], dim = 1)

        # get targets
        team1_y = torch.tensor(current_match.team1_positions)#.reshape(-1,1)
        team2_y = torch.tensor(current_match.team2_positions)#.reshape(-1,1)

        # create graphs
        team1_data = Data(x = team1_x, edge_index = team1_edge_index, edge_attr = team1_edge_attr, y = team1_y)
        team2_data = Data(x = team2_x, edge_index = team2_edge_index, edge_attr = team2_edge_attr, y = team2_y)

        return team1_data, team2_data

    @staticmethod
    def generate_pyg_graphs_from_competition_v1(current_competition: CompetitionClient, 
                                                verbose: bool = True):
        """
        Generates passing networks as torch_geometric.data.Data objects for all matches in the current competition. 
        Returns the a dictionary whose keys are the matchIds, and whose values are a list containing the graphs for the 2 teams in a given match
        This version generates graphs with 18 features (6 for pass/shot/duel * acc/inacc, and 12 for summary statistics for initialX, initialY, finalX, finalY)
        params:
            current_competition: a CompetitionClient instance
            verbose: a bool determining whether or not to print progress
        """
        start_time = time.perf_counter()
        
        graphs = dict()

        events_df = current_competition.events_df

        # tag by accuracy. Note: this is used for both node and edge features
        events_df['isAcc'] = events_df['tags'].apply(lambda x: 1801 in x)

        # add fields used to determine edge weights
        events_df['sender'] = events_df['playerId']
        events_df['senderTeamId'] = events_df['teamId']
        events_df['receiver'] = events_df['playerId'].shift(-1).astype('Int64')
        events_df['receiverTeamId'] = events_df['teamId'].shift(-1).astype('Int64')

        # get passing events used to create passing networks
        passes_df = events_df[                  
            events_df['isAcc'] &                                        # filter only accurate events
            (events_df['eventId'] == 8) &                               # filter only pass events
            (events_df['senderTeamId'] == events_df['receiverTeamId'])  # filter only passes sent and received by players on same team
        ][['matchId', 'teamId', 'sender', 'receiver', 'positions']]

        # compute pass distances and inverse
        passes_df['dist'] = passes_df['positions'].apply(
            lambda points: math.sqrt((points[0]['x'] - points[1]['x'])**2 + (points[0]['y'] - points[1]['y'])**2)
        )
        passes_df['inverseDist'] = 1/(passes_df['dist'] + 1)    # add 1 to denom to avoid div by zero

        # get edge features
        pass_strengths = passes_df.groupby(['matchId', 'teamId', 'sender', 'receiver'])['inverseDist'].mean() # mean of inverseDists across pairs of players
        pass_counts = passes_df.groupby(['matchId', 'teamId'])[['sender', 'receiver']].value_counts()           # number of passes between pairs of players

        # get events used for computing node features
        events_for_nodes = events_df[events_df['eventId'].isin([1,8,10])]

        # add fields for position features
        # unpack first coord
        events_for_nodes['initialX'] = events_for_nodes['positions'].apply(lambda x: x[0]['x'])
        events_for_nodes['initialY'] = events_for_nodes['positions'].apply(lambda x: x[0]['y'])
        # unpack second coord. If none given, use first coord 
        # Note: for some reason, there are 2 anomalous pass/duel/shots events in this dataset do not have a 2nd coord
        events_for_nodes['finalX'] = events_for_nodes['positions'].apply(lambda x: x[1]['x'] if (len(x) == 2) else x[0]['x'])
        events_for_nodes['finalY'] = events_for_nodes['positions'].apply(lambda x: x[1]['y'] if (len(x) == 2) else x[0]['y'])
        

        # add fields used for accuracy features
        events_for_nodes['eventAndAcc'] = events_for_nodes.apply(
            lambda row: f"{'acc' if row['isAcc'] else 'inacc'}{events_map[row['eventId']]}",
            axis = 1
        )

        # compute node features
        # counts for accurate/inaccurate duels/passes/shots
        accuracy_summary = (events_for_nodes
                            .groupby(['matchId', 'teamId', 'playerId'])['eventAndAcc']
                            .value_counts()
                            .unstack(fill_value = 0))
        # summary statistics for coordinates
        position_summary = events_for_nodes.groupby(['matchId', 'teamId', 'playerId']).agg({
            'initialX': ['mean', 'std', 'skew'],            
            'initialY': ['mean', 'std', 'skew'],
            'finalX': ['mean', 'std', 'skew'],
            'finalY': ['mean', 'std', 'skew']
        })
        # compute means for columns that likely have missing values
        values = {(upper, lower): position_summary[upper][lower].mean() 
                for upper in ['initialX', 'initialY', 'finalX', 'finalY'] 
                for lower in ['std', 'skew']}
        # fill na values with means of respective column
        position_summary.fillna(value = values, inplace = True)

        # MATCH SPECIFIC OPERATIONS
        match_ids = current_competition.match_details.keys()
        for i, match_id in enumerate(list(match_ids)):
            intermediate_time = time.perf_counter()
            if(verbose and (i+1)%20 == 0):
                print(f"\tProcessed {i+1} matches \t({intermediate_time - start_time:.3f} secs)")
            
            current_match = MatchData(current_competition, match_id)
            
            # quick aliases for match-specific information 
            match_id = current_match.match_id

            team1_id = current_match.team1
            team2_id = current_match.team2

            team1_players = current_match.team1_players
            team2_players = current_match.team2_players

            team1_encoder = current_match.team1_encoder
            team2_encoder = current_match.team2_encoder

            # get info about passes in match
            # pass counts divided by max
            team1_pass_freqs = pass_counts.loc[match_id, team1_id, team1_players, team1_players]    # Note: the last two components of the multi-index gets only the starting 11
            team2_pass_freqs = pass_counts.loc[match_id, team2_id, team2_players, team2_players]
            team1_pass_freqs = team1_pass_freqs/(team1_pass_freqs.max())
            team2_pass_freqs = team2_pass_freqs/(team2_pass_freqs.max())

            # pass strengths
            team1_pass_strengths = pass_strengths.loc[match_id, team1_id, team1_players, team1_players]
            team2_pass_strengths = pass_strengths.loc[match_id, team2_id, team2_players, team2_players]

            # get edge features
            team1_edge_attr = torch.stack([torch.tensor(team1_pass_freqs.values, dtype = EDGE_WEIGHT_DTYPE),
                                        torch.tensor(team1_pass_strengths.values, dtype = EDGE_WEIGHT_DTYPE)]).T
            team2_edge_attr = torch.stack([torch.tensor(team2_pass_freqs.values, dtype = EDGE_WEIGHT_DTYPE),
                                        torch.tensor(team2_pass_strengths.values, dtype = EDGE_WEIGHT_DTYPE)]).T

            # get edge indices and map playerId's to node indices in {0, ..., 10}
            team1_edge_index = torch.tensor(
                team1_encoder.transform(
                    team1_pass_freqs.droplevel(level = [0, 1]).reset_index()[['sender', 'receiver']].values
                    .flatten(order = 'F')   # flatten so that labelEncoder can be applied. 
                    ).reshape(2, -1))       # used Fortran order above so that original order can be recovered after transforming    
            team2_edge_index = torch.tensor(
                team2_encoder.transform(
                    team2_pass_freqs.droplevel(level = [0, 1]).reset_index()[['sender', 'receiver']].values
                    .flatten(order = 'F')    
                    ).reshape(2, -1))

            # get accuracy features
            team1_accuracy_summary = accuracy_summary.loc[match_id, team1_id, team1_players]
            team2_accuracy_summary = accuracy_summary.loc[match_id, team2_id, team2_players]

            # get position summary
            team1_position_summary = position_summary.loc[match_id, team1_id, team1_players]
            team2_position_summary = position_summary.loc[match_id, team2_id, team2_players]

            # get all node features
            team1_x = torch.cat([torch.tensor(team1_accuracy_summary.values, dtype = X_DTYPE), 
                                torch.tensor(team1_position_summary.values, dtype = X_DTYPE)], dim = 1)
            team2_x = torch.cat([torch.tensor(team2_accuracy_summary.values, dtype = X_DTYPE), 
                                torch.tensor(team2_position_summary.values, dtype = X_DTYPE)], dim = 1)

            # get targets (player positions)
            team1_y = torch.tensor(current_match.team1_positions, dtype = TARGET_DTYPE)
            team2_y = torch.tensor(current_match.team2_positions, dtype = TARGET_DTYPE)

            # create graphs
            team1_data = Data(x = team1_x, edge_index = team1_edge_index, edge_attr = team1_edge_attr, y = team1_y)
            team2_data = Data(x = team2_x, edge_index = team2_edge_index, edge_attr = team2_edge_attr, y = team2_y)

            graphs[match_id] = [team1_data, team2_data]

        return graphs

    @staticmethod
    def generate_pyg_graphs_from_competition_v2(current_competition: CompetitionClient, 
                                                verbose: bool = True):
        """
        Generates passing networks as torch_geometric.data.Data objects for all matches in the current competition. 
        Returns the a dictionary whose keys are the matchIds, and whose values are a list containing the graphs for the 2 teams in a given match
        This version generates graphs with 41 features (29 for subevents counts, 12 position summary stats)
        params:
            current_competition: a CompetitionClient instance
            verbose: a bool determining whether or not to print progress
        """
        start_time = time.perf_counter()
        
        graphs = dict()

        events_df = current_competition.events_df

        # get information node/edge info for entire competition
        node_features_df = PygPassingNetworks.get_node_features_df(events_df)
        edge_features_df = PygPassingNetworks.get_edge_features_df(events_df)
        
        # MATCH SPECIFIC OPERATIONS
        match_ids = current_competition.match_details.keys()
        for i, match_id in enumerate(list(match_ids)):
            intermediate_time = time.perf_counter()
            if(verbose and (i+1)%20 == 0):
                print(f"\tProcessed {i+1} matches \t({intermediate_time - start_time:.3f} secs)")
            
            current_match = MatchData(current_competition, match_id)
            
            # quick aliases for match-specific information 
            match_id = current_match.match_id

            team1_id = current_match.team1
            team2_id = current_match.team2

            team1_players = current_match.team1_players
            team2_players = current_match.team2_players

            team1_encoder = current_match.team1_encoder
            team2_encoder = current_match.team2_encoder

            # get edge info specific to match
            team1_edge_info = edge_features_df.reset_index(level = [2, 3]).loc[match_id, team1_id]
            team2_edge_info = edge_features_df.reset_index(level = [2, 3]).loc[match_id, team2_id]
            
            # get only starting 11. Here, the columns are [sender, receiver, inverseDist, count]
            team1_edge_info = team1_edge_info[team1_edge_info['sender'].isin(team1_players) & team1_edge_info['receiver'].isin(team1_players)]
            team2_edge_info = team2_edge_info[team2_edge_info['sender'].isin(team2_players) & team2_edge_info['receiver'].isin(team2_players)]

            # get edge features
            # need to use .astype(float) first because .values returns an array of type 'object'
            team1_edge_attr = torch.tensor(team1_edge_info.values[:, -2:].astype(float), dtype = EDGE_WEIGHT_DTYPE)
            team2_edge_attr = torch.tensor(team2_edge_info.values[:, -2:].astype(float), dtype = EDGE_WEIGHT_DTYPE)

            # get edge indices and map playerId's to node indices in {0, ..., 10}
            team1_edge_index = torch.tensor(
                team1_encoder.transform(
                    team1_edge_info.values[:, :2]   # get sender, receiver indices
                    .flatten()                      # flatten in order to apply encoder
                    ).reshape(2, -1, order = 'F')   # reshape with Fortran order to recover original structure
                    )
            team2_edge_index = torch.tensor(
                team2_encoder.transform(
                    team2_edge_info.values[:, :2]   # get sender, receiver indices
                    .flatten()                      # flatten in order to apply encoder
                    ).reshape(2, -1, order = 'F')   # reshape with Fortran order to recover original structure
                    )

            # get node info specific to match
            team1_node_info = node_features_df.loc[match_id, team1_id, team1_players]
            team2_node_info = node_features_df.loc[match_id, team2_id, team2_players]

            # get node features
            # need to use .astype(float) first because .values returns an array of type 'object'
            team1_x = torch.tensor(team1_node_info.values.astype(float), dtype = X_DTYPE)
            team2_x = torch.tensor(team2_node_info.values.astype(float), dtype = X_DTYPE)

            # get targets (player positions)
            team1_y = torch.tensor(current_match.team1_positions, dtype = TARGET_DTYPE)
            team2_y = torch.tensor(current_match.team2_positions, dtype = TARGET_DTYPE)

            # create graphs
            team1_data = Data(x = team1_x, edge_index = team1_edge_index, edge_attr = team1_edge_attr, y = team1_y)
            team2_data = Data(x = team2_x, edge_index = team2_edge_index, edge_attr = team2_edge_attr, y = team2_y)

            graphs[match_id] = [team1_data, team2_data]

        return graphs

    @staticmethod
    def generate_pyg_graphs_from_competition_v3(current_competition: CompetitionClient,
                                                heatmap_model: HeatmapMLP, 
                                                verbose: bool = True):
        """
        Generates passing networks as torch_geometric.data.Data objects for all matches in the current competition. 
        Returns the a dictionary whose keys are the matchIds, and whose values are a list containing the graphs for the 2 teams in a given match
        This version generates graphs with 4 features generated by taking player heatmaps and feeding into HeatmapMLP model.
        params:
            current_competition: a CompetitionClient instance,
            verbose: a bool determining whether or not to print progress,
            heatmap_model: a HeatmapMLP model trained to identify player position using heatmaps
        """
        heatmap_model.eval()
        start_time = time.perf_counter()

        graphs = dict()

        events_df = current_competition.events_df

        coords_df = CustomData.get_coords_df(events_df, append_labels = False)
        coords_df['heatmap'] = coords_df['coordsList'].apply(CustomTransforms.coords_list_to_heatmap)

        edge_features_df = PygPassingNetworks.get_edge_features_df(events_df)

        # MATCH SPECIFIC OPERATIONS
        match_ids = current_competition.match_details.keys()
        for i, match_id in enumerate(list(match_ids)):
            intermediate_time = time.perf_counter()
            if(verbose and ((i+1)%20 == 0)):
                print(f"\tProcessed {i+1} matches \t({intermediate_time - start_time:.3f} secs)")
            
            current_match = MatchData(current_competition, match_id)
            
            # quick aliases for match-specific information 
            match_id = current_match.match_id
            
            team1_id = current_match.team1
            team1_players = current_match.team1_players
            team1_encoder = current_match.team1_encoder
            
            team2_id = current_match.team2
            team2_players = current_match.team2_players
            team2_encoder = current_match.team2_encoder

            # get edge info specific to match
            team1_edge_info = edge_features_df.reset_index(level = [2, 3]).loc[match_id, team1_id]
            team2_edge_info = edge_features_df.reset_index(level = [2, 3]).loc[match_id, team2_id]
            
            # get only starting 11. Here, the columns are [sender, receiver, inverseDist, count]
            team1_edge_info = team1_edge_info[team1_edge_info['sender'].isin(team1_players) & team1_edge_info['receiver'].isin(team1_players)]
            team2_edge_info = team2_edge_info[team2_edge_info['sender'].isin(team2_players) & team2_edge_info['receiver'].isin(team2_players)]

            # get edge features
            # need to use .astype(float) first because .values returns an array of type 'object'
            team1_edge_attr = torch.tensor(team1_edge_info.values[:, -2:].astype(float), dtype = torch.float32)
            team2_edge_attr = torch.tensor(team2_edge_info.values[:, -2:].astype(float), dtype = torch.float32)

            # get edge indices and map playerId's to node indices in {0, ..., 10}
            team1_edge_index = torch.tensor(
                team1_encoder.transform(
                    team1_edge_info.values[:, :2]   # get sender, receiver indices
                    .flatten()                      # flatten in order to apply encoder
                    ).reshape(2, -1, order = 'F')   # reshape with Fortran order to recover original structure
                    )
            team2_edge_index = torch.tensor(
                team2_encoder.transform(
                    team2_edge_info.values[:, :2]   # get sender, receiver indices
                    .flatten()                      # flatten in order to apply encoder
                    ).reshape(2, -1, order = 'F')   # reshape with Fortran order to recover original structure
                    )

            # get coordinates specific to match
            team1_coords_df = coords_df.loc[match_id, team1_id, team1_players]
            team2_coords_df = coords_df.loc[match_id, team1_id, team1_players]

            # flatten heatmaps
            team1_flat_heatmaps = torch.stack(list(team1_coords_df['heatmap'].values)).flatten(start_dim = -3)
            team2_flat_heatmaps = torch.stack(list(team2_coords_df['heatmap'].values)).flatten(start_dim = -3)

            # feed flat heatmaps to MLP
            # The 0 here is meaningless and is just included since the heatmap_model expects a tuple (X, y) 
            # .detach_() removes these tensors from the computation graph. We do this since these will  be inputs to our new model
            team1_x = heatmap_model((team1_flat_heatmaps, 0), is_flattened = True).detach_()
            team2_x = heatmap_model((team2_flat_heatmaps, 0), is_flattened = True).detach_()

            # get targets (player positions)
            team1_y = torch.tensor(current_match.team1_positions, dtype = torch.long)
            team2_y = torch.tensor(current_match.team2_positions, dtype = torch.long)

            # create graphs
            team1_data = Data(x = team1_x, edge_index = team1_edge_index, edge_attr = team1_edge_attr, y = team1_y)
            team2_data = Data(x = team2_x, edge_index = team2_edge_index, edge_attr = team2_edge_attr, y = team2_y)

            graphs[match_id] = [team1_data, team2_data]

        return graphs

    @staticmethod
    def generate_pyg_graphs_from_competition_v4(current_competition: CompetitionClient,
                                                verbose: bool = True):
        """
        Generates passing networks as torch_geometric.data.Data objects for all matches in the current competition. 
        Returns the a dictionary whose keys are the matchIds, and whose values are a list containing the graphs for the 2 teams in a given match
        This version generates graphs whose node features are a (2500,) tensor which is the player's flattened heatmap.
        params:
            current_competition: a CompetitionClient instance,
            verbose: a bool determining whether or not to print progress,
        """
        start_time = time.perf_counter()

        graphs = dict()

        events_df = current_competition.events_df

        coords_df = CustomData.get_coords_df(events_df, append_labels = False)
        coords_df['heatmap'] = coords_df['coordsList'].apply(CustomTransforms.coords_list_to_heatmap)

        edge_features_df = PygPassingNetworks.get_edge_features_df(events_df)

        # MATCH SPECIFIC OPERATIONS
        match_ids = current_competition.match_details.keys()
        for i, match_id in enumerate(list(match_ids)):
            intermediate_time = time.perf_counter()
            if(verbose and ((i+1)%20 == 0)):
                print(f"\tProcessed {i+1} matches \t({intermediate_time - start_time:.3f} secs)")
            
            current_match = MatchData(current_competition, match_id)
            
            # quick aliases for match-specific information 
            match_id = current_match.match_id
            
            team1_id = current_match.team1
            team1_players = current_match.team1_players
            team1_encoder = current_match.team1_encoder
            
            team2_id = current_match.team2
            team2_players = current_match.team2_players
            team2_encoder = current_match.team2_encoder

            # get edge info specific to match
            team1_edge_info = edge_features_df.reset_index(level = [2, 3]).loc[match_id, team1_id]
            team2_edge_info = edge_features_df.reset_index(level = [2, 3]).loc[match_id, team2_id]
            
            # get only starting 11. Here, the columns are [sender, receiver, inverseDist, count]
            team1_edge_info = team1_edge_info[team1_edge_info['sender'].isin(team1_players) & team1_edge_info['receiver'].isin(team1_players)]
            team2_edge_info = team2_edge_info[team2_edge_info['sender'].isin(team2_players) & team2_edge_info['receiver'].isin(team2_players)]

            # get edge features
            # need to use .astype(float) first because .values returns an array of type 'object'
            team1_edge_attr = torch.tensor(team1_edge_info.values[:, -2:].astype(float), dtype = torch.float32)
            team2_edge_attr = torch.tensor(team2_edge_info.values[:, -2:].astype(float), dtype = torch.float32)

            # get edge indices and map playerId's to node indices in {0, ..., 10}
            team1_edge_index = torch.tensor(
                team1_encoder.transform(
                    team1_edge_info.values[:, :2]   # get sender, receiver indices
                    .flatten()                      # flatten in order to apply encoder
                    ).reshape(2, -1, order = 'F')   # reshape with Fortran order to recover original structure
                    )
            team2_edge_index = torch.tensor(
                team2_encoder.transform(
                    team2_edge_info.values[:, :2]   # get sender, receiver indices
                    .flatten()                      # flatten in order to apply encoder
                    ).reshape(2, -1, order = 'F')   # reshape with Fortran order to recover original structure
                    )

            # get coordinates specific to match
            team1_coords_df = coords_df.loc[match_id, team1_id, team1_players]
            team2_coords_df = coords_df.loc[match_id, team1_id, team1_players]

            # flatten heatmaps
            team1_x = torch.stack(list(team1_coords_df['heatmap'].values)).flatten(start_dim = -3)
            team2_x = torch.stack(list(team2_coords_df['heatmap'].values)).flatten(start_dim = -3)

            # get targets (player positions)
            team1_y = torch.tensor(current_match.team1_positions, dtype = torch.long)
            team2_y = torch.tensor(current_match.team2_positions, dtype = torch.long)

            # create graphs
            team1_data = Data(x = team1_x, edge_index = team1_edge_index, edge_attr = team1_edge_attr, y = team1_y)
            team2_data = Data(x = team2_x, edge_index = team2_edge_index, edge_attr = team2_edge_attr, y = team2_y)

            graphs[match_id] = [team1_data, team2_data]

        return graphs
    
    @staticmethod
    def get_all_graphs_as_pyg(version: int = 2, model: HeatmapMLP = None, 
                              verbose: bool = True, 
                              connection_string: str = "mongodb://localhost:27017/"):
        """
        Generates PyG graphs for all matches and all competitions.  
        Returns a dictionary called "graphs," which is a nested dictionary which
        can be indexed as follows:
        graphs[competition_string][match_id][index_in_01]
        params:
            version: an int indicating which version of generate_pyg_graphs_from_competition to use
            model: a HeatmapMLP model. Is only used if version == 3
            verbose: a bool indicating whether or not to print progress
            connection_string: connection string used to initialize CompetitionClient (which is a wrapper for MongoClient).
                            Default value is the connection string used to connect to locally hosted server 
        """
        competitions = ['England', 'European_Championship', 'France', 'Germany', 'Italy', 'Spain', 'World_Cup']

        start_time = time.perf_counter()
        graphs = dict()
        for competition in competitions:
            if(verbose):
                print(f"Processing {competition}")
            
            current_client = CompetitionClient(connection_string, competition)

            if version == 1:
                graphs[competition] = PygPassingNetworks.generate_pyg_graphs_from_competition_v1(current_client, verbose = verbose)
            elif version == 2:
                graphs[competition] = PygPassingNetworks.generate_pyg_graphs_from_competition_v2(current_client, verbose = verbose)
            elif version == 3:
                graphs[competition] = PygPassingNetworks.generate_pyg_graphs_from_competition_v3(current_client, heatmap_model = model, verbose = verbose)
            elif version == 4:
                graphs[competition] = PygPassingNetworks.generate_pyg_graphs_from_competition_v4(current_client, verbose = verbose)

            intermediate_time = time.perf_counter()
            if(verbose):
                print(f"Finished graphs for {competition} ({intermediate_time - start_time:.3f} secs)")
                print("="*50) 
        end_time = time.perf_counter()
        if(verbose):
            print(f"Execution time: {end_time - start_time:.3f} seconds")
        
        return graphs

    @staticmethod
    def validate_data(graphs: dict, mutate: bool = False):
        """
        Validates a collection of PyG graphs by calling the .validate() method on each graph. 
        Params:
            graphs: a nested dictionary whose structure is as the output of get_all_graphs_as_pyg
            mutate: a bool indicating whether or not to delete items which raise an error
        """
        bad_indices = []
        for comp_name, comp_dict in graphs.items():
            for match_id, match_data in comp_dict.items():
                for i, team_data in enumerate(match_data):
                    try:
                        team_data.validate(raise_on_error = True) 
                    except ValueError as e:
                        print(f'Found error in {comp_name, match_id, i}:')
                        print(f'\tError message: {e}')
                        if(mutate):
                            bad_indices.append((comp_name, match_id, i))

        if(mutate):
            if(len(bad_indices) > 0):
                for triple in bad_indices:
                    graphs[triple[0]][triple[1]].pop(triple[2])
                print("Removed malformed data")
            else:
                print("No malformed data found")

# PyG to NetworkX
def pyg_graph_to_nx_graph(data: Data):
    """
    Converts PyGeometric graph into a NetworkX graph.
    Returns an nx.DiGraph()
    params:
        data: a torch_geometric.data.Data object
    """
    g = nx.DiGraph()
    weighted_edge_list = np.concat([data.edge_index.T, data.edge_attr[:, 1].reshape(-1,1)], axis = 1)
    g.add_weighted_edges_from(weighted_edge_list)
    return g

if __name__ == "__main__": 
    pass



    