import itertools
import torch

GK_COL = 0
LB_COL, CB_COL, RB_COL = 1, 2, 3
LM_COL, CM_COL, RM_COL = 4, 5, 6
LW_COL, CF_COL, RW_COL = 7, 8, 9

GK_COLS = [GK_COL]
DF_COLS = [LB_COL, CB_COL, RB_COL]
MD_COLS = [LM_COL, CM_COL, RM_COL]
FW_COLS = [LW_COL, CF_COL, RW_COL]

# helper function for segment_players
def create_position_mask(row_indices: int | list, col_indices: int | list, 
                         height: int = 11, width: int = 10):
    """
    Creates a tensor of 0's and 1's where the tensor is 0 at the specified rows and columns,
    and 1 where the rows and cols intersect and elsewhere.  
    params:
        row_indices: int or list of ints specifying which row(s) to zero-out
        col_indices: int or list of ints specifying which column(s) to zero-out
        height: int specifying height of desired tensor
        width: int specifying height of desired tensor
    """
    result = torch.ones(height, width)
    
    # Create masks
    row_mask = torch.zeros(height, dtype=torch.bool)
    col_mask = torch.zeros(width, dtype=torch.bool)
    row_mask[row_indices] = True
    col_mask[col_indices] = True
    
    # Set zeros in specified rows and columns
    result[row_mask, :] = 0
    result[:, col_mask] = 0
    
    # Set 1's at intersections
    result[row_mask[:, None] & col_mask[None, :]] = 1
    
    return result

# segments players into categories based on formation
def segment_players(original_team_probabilities, formation: tuple = (4, 3, 3)):
    """
    Adjusts the original team probabilities by segmenting the players into categories (i.e. GK, DF, MD, FW).
    params:
        original_team_probabilities: an 11x10 tensor representing model predictions on a full starting-11.
        formation: a tuple (x, y, z) where x is the number of defenders, y is the number of midfielders, and z is the number of forwards.
        Note that the tuple entries must sum to 10.
    """
    num_df, num_md, num_fw = formation
    num_players = num_df + num_md + num_fw 
    assert num_players == 10, f"Invalid formation. Expected 10 outfield players but got {num_players}"
    
    # prob_adjusted is a clone of the model output. 
    # This will get modified in order to assign players to the correct categories
    # We create a clone as to not tamper with the original probabilities, which will be used later to compute evaluation metric 
    prob_adjusted = original_team_probabilities.clone()

    # identify player with highest probability of being GK
    gk_player_index = prob_adjusted.argmax(dim = 0)[0]
    # zero-out the GK probability for all other players, and the outfield probability of chosen GK
    gk_mask = create_position_mask(gk_player_index, 0)
    prob_adjusted = prob_adjusted*gk_mask

    # identify players with highest probability of being DF
    df_col = torch.sum(prob_adjusted[:, 1:4], dim = 1, keepdim = True) # {LB, CB, RB} -> DF
    df_player_indices = torch.topk(df_col, k = num_df, dim = 0)[1].flatten().sort()[0]
    # zero-out DF probability for all other players, and the non-DF probability of chosen DF players
    df_mask = create_position_mask(df_player_indices, [1, 2, 3])
    prob_adjusted = prob_adjusted*df_mask 

    # identify players with highest probability of being MD
    md_col = torch.sum(prob_adjusted[:, 4:7], dim = 1, keepdim = True) # {LM, CM, RM} -> MD
    md_player_indices = torch.topk(md_col, k = num_md, dim = 0)[1].flatten().sort()[0]
    # zero-out MD probability for all other players, and the non-MD probability of chosen MD players
    md_mask = create_position_mask(md_player_indices, [4, 5, 6])
    prob_adjusted = prob_adjusted*md_mask 

    # identify players with highest probability of being FW
    fw_col = torch.sum(prob_adjusted[:, 7:], dim = 1, keepdim = True)  # {LW, CF, RW} -> FW
    fw_player_indices = torch.topk(fw_col, k = num_fw, dim = 0)[1].flatten().sort()[0]
    # zero-out MD probability for all other players, and the non-MD probability of chosen MD players
    fw_mask = create_position_mask(fw_player_indices, [7, 8, 9])
    prob_adjusted = prob_adjusted*fw_mask 

    # normalize within segmentation
    prob_adjusted = torch.nn.functional.normalize(prob_adjusted, p = 1, dim = 1)

    return prob_adjusted, {'GK': gk_player_index.tolist(),
                           'DF': df_player_indices.tolist(),
                           'MD': md_player_indices.tolist(),
                           'FW': fw_player_indices.tolist()}


# given a miniheatmap for 2 wide positions and some central positions, find optimal assignment
def find_miniheatmap_optimizer(miniheatmap):
    max_score = 0
    final_left_player = None
    final_right_player = None

    k = len(miniheatmap)
    for left_candidate, right_candidate in itertools.product(range(k), repeat = 2):
        if left_candidate == right_candidate:
            continue
        # initialize choice as all central
        mini_choice_matrix = torch.zeros(k, 3)
        mini_choice_matrix[:, 1] = 1
        
        # change left candidate to left position
        mini_choice_matrix[left_candidate, 0] = 1
        mini_choice_matrix[left_candidate, 1] = 0

        # change right candidate to right position
        mini_choice_matrix[right_candidate, 2] = 1
        mini_choice_matrix[right_candidate, 1] = 0

        # evaluate current choice
        current_score = (mini_choice_matrix * miniheatmap).sum().item()
        if current_score > max_score:
            final_left_player = left_candidate
            final_right_player = right_candidate
            max_score = current_score

    return final_left_player, final_right_player

# helper function for assign_team_by_formation
def assign_gk_subroles(prob_adjusted: torch.Tensor, player_categories: dict, choice_matrix: torch.Tensor):
    """
    Helper function of assign_team_by_formation used to assign GK player to GK role
    params:
        prob_adjusted: an 11x10 torch.Tensor representing team position predictions adjusted by 
        segmenting into categories GK, DF, MD, FW (typically done using the segment_players function) 
        player_categories: a dictionary whose keys are the categories 'GK', 'DF', 'MD', 'FW', 
        and whose values are an int or list of ints, which are the row indices of the players in a given category.
        These indices are with respect to prob_adjusted.
        choice_matrix: a {0, 1}-valued 11x10 matrix representing the current class assignments of all players in the squad.
    """
    # assign GK
    gk_player_index = player_categories['GK']
    choice_matrix[gk_player_index, 0] = 1

# helper function for assign_team_by_formation
def assign_df_subroles(prob_adjusted: torch.Tensor, player_categories: dict, choice_matrix: torch.Tensor, 
                       wide_df_exists: bool):
    """
    Helper function of assign_team_by_formation used to assign players in the 'DF' category 
    to the classes LB, CB, RB.
    params:
        prob_adjusted: an 11x10 torch.Tensor representing team position predictions adjusted by 
        segmenting into categories GK, DF, MD, FW (typically done using the segment_players function) 
        player_categories: a dictionary whose keys are the categories 'GK', 'DF', 'MD', 'FW', 
        and whose values are an int or list of ints, which are the row indices of the players in a given category.
        These indices are with respect to prob_adjusted.
        choice_matrix: a {0, 1}-valued 11x10 matrix representing the current class assignments of all players in the squad.
        wide_df_exists: a bool indicating whether or not to assign players to LW and RW. If false, then all players are assigned CB
    """
    df_rows = player_categories['DF']

    # isolate heatmap to df candidates and df columns
    df_heatmap = prob_adjusted[df_rows][:, DF_COLS]
    
    # begin by assigning all candidates to CB
    choice_matrix[df_rows, CB_COL] = 1

    # if wide players needed
    if wide_df_exists:
        # identify optimal choice of LB and RB
        lb_player_relative_row, rb_player_relative_row = find_miniheatmap_optimizer(df_heatmap)
        lb_player_row = df_rows[lb_player_relative_row]
        rb_player_row = df_rows[rb_player_relative_row]

        # change candidates from CB to LB and RB respectively 
        choice_matrix[[lb_player_row, rb_player_row], CB_COL] = 0   
        choice_matrix[lb_player_row, LB_COL] = 1
        choice_matrix[rb_player_row, RB_COL] = 1

# helper function for assign_team_by_formation
def assign_md_subroles(prob_adjusted: torch.Tensor, player_categories: dict, choice_matrix: torch.Tensor, 
                       wide_md_exists: bool):
    """
    Helper function of assign_team_by_formation used to assign players in the 'MD' category 
    to the classes LM, CM, RM.
    params:
        prob_adjusted: an 11x10 torch.Tensor representing team position predictions adjusted by 
        segmenting into categories GK, DF, MD, FW (typically done using the segment_players function) 
        player_categories: a dictionary whose keys are the categories 'GK', 'DF', 'MD', 'FW', 
        and whose values are an int or list of ints, which are the row indices of the players in a given category.
        These indices are with respect to prob_adjusted.
        choice_matrix: a {0, 1}-valued 11x10 matrix representing the current class assignments of all players in the squad.
        wide_md_exists: a bool indicating whether or not to assign players to LM and RM. If false, then all players are assigned CM
    """
    md_rows = player_categories['MD']

    # isolate heatmap to md candidates and md columns
    md_heatmap = prob_adjusted[md_rows][:, MD_COLS]
    
    # begin by assigning all candidates to CM
    choice_matrix[md_rows, CM_COL] = 1

    # if wide players needed
    if wide_md_exists:
        # identify optimal choice of LM and RM
        lm_player_relative_row, rm_player_relative_row = find_miniheatmap_optimizer(md_heatmap)
        lm_player_row = md_rows[lm_player_relative_row]
        rm_player_row = md_rows[rm_player_relative_row]

        # change candidates from CB to LM and RM respectively 
        choice_matrix[[lm_player_row, rm_player_row], CM_COL] = 0   
        choice_matrix[lm_player_row, LM_COL] = 1
        choice_matrix[rm_player_row, RM_COL] = 1

# helper function for assign_team_by_formation
def assign_fw_subroles(prob_adjusted: torch.Tensor, player_categories: dict, choice_matrix: torch.Tensor, 
                       wide_fw_exists: bool):
    """
    Helper function of assign_team_by_formation used to assign players in the 'FW' category 
    to the classes LW, CF, RW.
    params:
        prob_adjusted: an 11x10 torch.Tensor representing team position predictions adjusted by 
        segmenting into categories GK, DF, MD, FW (typically done using the segment_players function) 
        player_categories: a dictionary whose keys are the categories 'GK', 'DF', 'MD', 'FW', 
        and whose values are an int or list of ints, which are the row indices of the players in a given category.
        These indices are with respect to prob_adjusted.
        choice_matrix: a {0, 1}-valued 11x10 matrix representing the current class assignments of all players in the squad.
        wide_fw_exists: a bool indicating whether or not to assign players to LW and RW. If false, then all players are assigned CF
    """
    fw_rows = player_categories['FW']

    # isolate heatmap to md candidates and md columns
    fw_heatmap = prob_adjusted[fw_rows][:, FW_COLS]
    
    # begin by assigning all candidates to CF
    choice_matrix[fw_rows, CF_COL] = 1

    # if wide players needed
    if wide_fw_exists:
        # identify optimal choice of LW and RW
        lw_player_relative_row, rw_player_relative_row = find_miniheatmap_optimizer(fw_heatmap)
        lw_player_row = fw_rows[lw_player_relative_row]
        rw_player_row = fw_rows[rw_player_relative_row]

        # change candidates from CF to LW and RW respectively 
        choice_matrix[[lw_player_row, rw_player_row], CF_COL] = 0   
        choice_matrix[lw_player_row, LW_COL] = 1
        choice_matrix[rw_player_row, RW_COL] = 1

# given segmented team probabilities and formation, assign detailed class labels to each player
def assign_team_by_formation(prob_adjusted: torch.Tensor, player_categories: dict, formation: tuple):
    """
    Given team position predictions segmented into categories (GK, DF, MD, FW), 
    this function assigns detailed class labels (GK, LB, CB, RB, LM, CM, RM, LW, CF, RW) to each player.
    This returns a {0, 1}-valued 11x10 matrix $C$ where $C_{ij} = 1$ if player i is assigned class j, and 0 otherwise.
    params:
        prob_adjusted: an 11x10 torch.Tensor representing team position predictions adjusted by 
        segmenting into categories GK, DF, MD, FW (typically done using the segment_players function) 
        player_categories: a dictionary whose keys are the categories 'GK', 'DF', 'MD', 'FW', 
        and whose values are an int or list of ints, which are the row indices of the players in a given category.
        These indices are with respect to prob_adjusted.
        formation: a tuple (x, y, z) where x is the number of defenders, y is the number of midfielders, and z is the number of forwards.
        Note that the tuple entries must sum to 10.
    """
    # initialize choice matrix
    choice_matrix = torch.zeros(11, 10)

    # determine if wide players exist based on given formation
    num_df, num_md, num_fw = formation
    wide_df_exists = (num_df > 3)
    wide_md_exists = (num_md > 3)
    wide_fw_exists = (num_fw > 2)

    # perform assignments
    assign_gk_subroles(prob_adjusted, player_categories, choice_matrix)
    assign_df_subroles(prob_adjusted, player_categories, choice_matrix, wide_df_exists)
    assign_md_subroles(prob_adjusted, player_categories, choice_matrix, wide_md_exists)
    assign_fw_subroles(prob_adjusted, player_categories, choice_matrix, wide_fw_exists)

    return choice_matrix

# evaluation metric. We choose a formation by minimizing this over all formations
ONES = torch.ones((11,10))
def eval_formation_score(original_probabilities: torch.Tensor, choice_matrix: torch.Tensor, formation: tuple):
    """
    Computes evaluation metric for class assignments given original probabilities and choice matrix.
    This metric measures how far the choice matrix deviates from the original probabilities.
    params:
        original_team_probabilities: an 11x10 tensor representing model predictions on a full starting-11.
        choice_matrix: a {0, 1}-valued 11x10 matrix $C$ where $C_{ij} = 1$ if player i is assigned class j, and 0 otherwise.
    """
    weights = -(2*choice_matrix - ONES)
    prod = original_probabilities*weights

    # # get score for each category
    # gk_score = prod[:, GK_COLS].sum()
    # df_score = prod[:, DF_COLS].sum()
    # md_score = prod[:, MD_COLS].sum()
    # fw_score = prod[:, FW_COLS].sum()

    # normalize by number of players in category
    # df_score /= math.sqrt(formation[0])
    # md_score /= math.sqrt(formation[1])
    # fw_score /= math.sqrt(formation[2])

    # unweighted sum
    score = prod.sum()

    return score

# idea: come up with list of admissable formations, e.g. 4-3-3, 4-4-2, 5-3-2, etc. 
# For each admissable lineup, identify which players occupy a given position by taking players who have the highest probability of being in that pos. 
# Score each admissable lineup based on the chosen players (e.g. maybe summing the probabilities of the chosen players, or something else)
# Chose the lineup which maximizes or minimizes whatever metric we choose

# obtained from espn data, excluding 4-2-4, 4-5-1, and 2-5-3
admissable_formations = [
    (5, 4, 1), 
    (5, 3, 2),
    (4, 4, 2), 
    (4, 3, 3),
    (3, 5, 2), 
    (3, 4, 3), 
]

# test all admissable formations and return the one which minimizes the evaluation metric
def get_best_formation(original_probabilities, get_choice_matrix: bool = False):
    
    best_score = float('inf')
    best_formation = None
    best_choice_matrix = None
    for formation in admissable_formations:
        try:
            prob_adjusted, player_categories = segment_players(original_probabilities, formation)
            current_choice_matrix = assign_team_by_formation(prob_adjusted, player_categories, formation)
            current_score = eval_formation_score(original_probabilities, current_choice_matrix, formation)            
            if current_score < best_score:
                best_score = current_score
                best_formation = formation
                if get_choice_matrix:
                    best_choice_matrix = current_choice_matrix
        except:
            print(f'Error while trying {formation}')
            raise
        
    return (best_formation, best_choice_matrix) if get_choice_matrix else best_formation



