from remote_data_handlers.metadata_handler import get_metadata_maps 
from remote_data_handlers.remote_match_data import RemoteMatchData
import networkx as nx
import numpy as np

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


def generate_nx_graph_from_match(match_data: RemoteMatchData):
    """
    Generates 2 passing networks, which are weighted directed graphs, from the current match.
    Returns the two graphs as g1, g2
    params:
        match_data: a RemoteMatchData instance
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

    graph_team_1.remove_edges_from(list(nx.selfloop_edges(graph_team_1)))
    graph_team_2.remove_edges_from(list(nx.selfloop_edges(graph_team_2)))

    return graph_team_1, graph_team_2


def add_normalized_edge_weights(G: nx.DiGraph):
    """
    Given an nx graph with edges have a 'weight' attribute (which are positive values),
    this adds an edge attribute named 'normalized_weight' which is the edge weight 
    normalized by the max edge weight of the graph
    params:
        G: an nx.DiGraph whose edges have a 'weight' attribute, which are positive values
    """
    weights = [e[-1] for e in G.edges.data('weight')]
    max_weight = max(weights)

    for edge in G.edges:
        G.edges[edge]['normalized_weight'] = G.edges[edge]['weight']/max_weight

