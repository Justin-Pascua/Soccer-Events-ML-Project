import numpy as np
import pandas as pd

import torch
import torchvision
from torch.utils.data import Dataset

import time

from sklearn.model_selection import train_test_split
from scipy import sparse

import os
os.environ['TORCH'] = torch.__version__

from local_data_handlers.wyscout_metadata_handler import get_players_maps, wyscout_label_encoder
from local_data_handlers.wyscout_data_handler import get_full_events_df

from local_data_handlers.espn_data_handler import get_processed_player_match_positions, espn_label_encoder

# X and y transforms
class CustomTransforms:
    """
    Collection of transformations used to prepare events data to be fed into heatmap classifier    
    """
    @staticmethod
    def coords_list_to_tensor_image(coord_list: list):
        """
        Converts list of coordinates representing indices into a torch tensor by 
        converting into a scipy sparse array, then converting to torch tensor.
        params:
            coord_list: a list of integer ordered pairs with coordinates in {0, ..., 100}
        """
        # check if empty
        if not coord_list:
            return torch.zeros((1, 101, 101))
        coords_arr = np.array(coord_list)

        # we take transpose in the end so that the image obtained aligns with intuition,
        # i.e. x-axis is horizontal, y-axis is vertical
        sparse_arr = sparse.coo_matrix((np.ones(len(coords_arr)),
                                        (coords_arr[:, 0], coords_arr[:, 1])),
                                        shape = (101, 101)).T
        
        # reshape to (1, 101, 101) so that pytorch recognizes it as an image
        image_arr = sparse_arr.toarray().reshape(1, 101, 101)
        torch_image = torch.tensor(image_arr, dtype = torch.float32)

        return torch_image

    @staticmethod
    def multi_coords_list_to_tensor_image(multichannel_list: list):
        return torch.cat([CustomTransforms.coords_list_to_tensor_image(channel) for channel in multichannel_list])
    
    @staticmethod
    def gaussian_blur(img: torch.Tensor):
        """
        Applies gaussian blur to 3d torch tensor.
        params:
            img: a PyTorch tensor of shape (num_channels, width, height)
        """
        return torchvision.transforms.GaussianBlur(kernel_size = 3, sigma = (5, 5))(img)

    @staticmethod
    def resizer(img: torch.Tensor):
        """
        Resizes 3d torch tensor into (*, 50, 50)
        params:
            img: a PyTorch tensor of shape (num_channels, width, height)
        """
        return torchvision.transforms.Resize(size = 50)(img)

    @staticmethod
    def coords_list_to_heatmap(coord_list: list):
        """
        Applies coords_list_to_tensor, resizer, and gaussian_blur in sequence to convert
        a list of coordinates into a (1, 50, 50) tensor.
        params:
            coord_list: a list of integer ordered pairs with coordinates in {0, ..., 100}
        """
        heatmap = CustomTransforms.gaussian_blur(
                  CustomTransforms.resizer(
                  CustomTransforms.coords_list_to_tensor_image(coord_list)))
        return heatmap

    @staticmethod
    def multi_coords_list_to_heatmap(multichannel_list: list):
        heatmap = CustomTransforms.gaussian_blur(
                  CustomTransforms.resizer(
                  CustomTransforms.multi_coords_list_to_tensor_image(multichannel_list)))
        return heatmap

# custom dataset class
class HeatmapDataset(Dataset):
    def __init__(self, X_arr, y_arr):
        self.data = X_arr
        self.targets = y_arr

    def __len__(self):
        return len(self.targets)
    
    def __getitem__(self, i):
        x = self.data[i]
        y = self.targets[i]
        return x, y
    
    def __str__(self):
        output = f"""HeatmapDataset:
        - length: {len(self)}
        - sample shapes: {self.data[0].shape}
        """
        return output

# functions for getting data
class CustomData:
    """
    Collections of functions used to manipulate events dataframes into suitable formats for heatmap classifier
    """
    @staticmethod
    def drop_out_of_bounds_events(events_df: pd.DataFrame):
        if ('initX' not in events_df.columns) or ('initY' not in events_df.columns):
            # get first coordinate of event
            events_df['initialPos'] = events_df['positions'].apply(lambda x: [(x[0]['x'], x[0]['y'])])

            # get coords
            events_df['initX'] = events_df['positions'].apply(lambda x: x[0]['x'])
            events_df['initY'] = events_df['positions'].apply(lambda x: x[0]['y'])

        # drop rows with misinputted coords (i.e. those at the corners (0,0), (0, 100), (100, 0), (100, 100) and everything out of bounds)
        events_df = events_df[~((events_df['initX'] == 0) & (events_df['initX'] == 0))]     # (0, 0) 
        events_df = events_df[~((events_df['initX'] == 0) & (events_df['initX'] == 100))]   # (0, 100)
        events_df = events_df[~((events_df['initX'] == 100) & (events_df['initX'] == 0))]   # (100, 0)
        events_df = events_df[~((events_df['initX'] == 100) & (events_df['initX'] == 100))] # (100, 100)
        events_df = events_df[(events_df['initX'] >= 0) & (events_df['initX'] <= 100)]      # keep only coords in [0, 100]
        events_df = events_df[(events_df['initY'] >= 0) & (events_df['initY'] <= 100)]

        return events_df

    @staticmethod
    def get_coords_df(events_df: pd.DataFrame, separate_by_events: bool = False, 
                      labels_source: str = None):
        """
        Groups events dataframe by match, team, and player. Then, combines all initial coordinates into 
        a list stored in a column 'coordsList.'
        params:
            events_df: a pandas Dataframe containing match events.
            separate_by_events: a bool indicating whether or not to group coords based on event id's. 
            If false, all event coords are placed into one list.
            If true, event coords are partitioned into separate lists.
            labels_source: a string, either 'wyscout' or 'espn', indicating which dataset to use for labels
        """
        # remove events associated with no player
        events_df = events_df[events_df['playerId'] != 0]

        # filter out interruptions and offsides
        events_df = events_df[~ events_df['eventId'].isin([5,6])] 

        # get first coordinate of event
        events_df['initialPos'] = events_df['positions'].apply(lambda x: [(x[0]['x'], x[0]['y'])])
        
        # get rid of events whose coords are not in [0, 100] x [0, 100]
        events_df = CustomData.drop_out_of_bounds_events(events_df)       
        
        # get all coords of player in a specific match
        grouping_cols = ['matchId', 'teamId', 'playerId']
        if separate_by_events:
            grouping_cols.append('eventId')
        coords_df = events_df.groupby(grouping_cols)[['initialPos']].agg({
            'initialPos': ['sum'],
        })

        # rename columns
        coords_df.columns = coords_df.columns.droplevel(1)
        coords_df.rename(columns = {'initialPos': 'coordsList'}, inplace = True)

        # if separate_by_events, need to fill empty events with empty list and combine into list of lists
        if separate_by_events:
            # fill empty events with empty list
            complete_index = pd.MultiIndex.from_tuples(
                [(match, team, player, event) 
                for (match, team, player) in coords_df.index.droplevel(3).drop_duplicates()
                for event in {1, 2, 3, 4, 7, 8, 9, 10}],
                names = ['matchId', 'teamId', 'playerId', 'eventId'])
            coords_df = coords_df.reindex(complete_index, fill_value = [])

            # move separate events into columns
            coords_df = coords_df.unstack(level = 'eventId')

            # aggregate into list of lists
            coords_df['coordsList'] = coords_df.apply(lambda row: [row[col] for col in coords_df.columns], axis = 1)
            
            # drop unused columns 
            coords_df.drop(columns = coords_df.columns[:-1], inplace = True)
            coords_df.columns = ['coordsList']

        # get labels
        if labels_source == 'wyscout':
            # turn playerId index into data column
            coords_df.reset_index(level = [2], inplace = True)
            
            # get labels using playerId
            _, _, player_to_pos, _ = get_players_maps(verbose = False)
            coords_df['numericalLabel'] = coords_df['playerId'].apply(lambda x: wyscout_label_encoder[player_to_pos[x]])
        elif labels_source == 'espn':
            # get espn labels data
            processed_espn_data = get_processed_player_match_positions()
            processed_espn_data['numericalLabel'] = processed_espn_data['posLabel'].map(espn_label_encoder)
            processed_espn_data.set_index(['matchId', 'teamId', 'playerId'], inplace = True)

            # append espn labels to coords df
            coords_df = coords_df.join(processed_espn_data[['numericalLabel']], how = 'left')
            coords_df['numericalLabel'] = coords_df['numericalLabel'].astype('Int64')
            coords_df.dropna(subset = 'numericalLabel', inplace = True)

        return coords_df

    @staticmethod
    def _coords_df_to_tensors(coords_df: pd.DataFrame, separate_by_events: bool = False):
        """
        This method applies the CustomTransforms.coords_list_to_heatmap transformation to a dataframe of coordinate lists and player labels.
        Then, the heatmaps are stacked into a single tensor, and the player labels column is converted into a tensor.
        params:
            coords_df: a pandas Dataframe containing heatmaps in a column called 'heatmap' and player labels in a column called 'numericalLabel'.
            separated_by_events: a bool indicating whether or not to the coordinate lists of coords_df are partitioned by eventId.
        """
        # if separated, then need to apply coords_list_to_heatmap to each separate list
        if separate_by_events:
            coords_df['heatmap'] = coords_df['coordsList'].apply(
                lambda x: torch.cat([CustomTransforms.coords_list_to_heatmap(channel) for channel in x])
            )
        # otherwise, we can apply coords_list_to_heatmap to the given list
        else:
            coords_df['heatmap'] = coords_df['coordsList'].apply(
                lambda x: CustomTransforms.coords_list_to_heatmap(x)
            )
        
        X_tensor = torch.stack(list(coords_df['heatmap'].values))
        y_tensor = torch.tensor(coords_df['numericalLabel'].values)
        
        return X_tensor, y_tensor

    @staticmethod
    def _get_train_val_test_tensors(coords_df: pd.DataFrame, separate_by_events: bool = False,
                                    train_test_frac: float = 0.2, train_val_frac: float = 0.2):
        """
        Given a dataframe of coordinate lists and player labels, this method splits the dataframe into train/val/test sets in the form of tensors.
        params:
            coords_df: a pandas Dataframe containing heatmaps in a column called 'heatmap' and player labels in a column called 'numericalLabel'.
            train_test_frac: a float indicating what portion of the dataset is to be used for the test set.
            train_val_frac: a float indicating what portion of the non-test data is to be used for the validation set.
            separate_by_events: a bool indicating whether or not to the coordinate lists of coords_df are partitioned by eventId.
        """
        train_df, test_df = train_test_split(coords_df, test_size = train_test_frac)
        train_df, val_df = train_test_split(train_df, test_size = train_val_frac)

        X_train_images, y_train = CustomData._coords_df_to_tensors(train_df, separate_by_events)
        X_val_images, y_val = CustomData._coords_df_to_tensors(val_df, separate_by_events)
        X_test_images, y_test = CustomData._coords_df_to_tensors(test_df, separate_by_events)

        return X_train_images, y_train, X_val_images, y_val, X_test_images, y_test

    @staticmethod
    def get_datasets(labels_source: str = None, separate_by_events: bool = False,
                     train_test_frac: float = 0.2, train_val_frac: float = 0.2, 
                     verbose: bool = True):
        """
        Gets the training/validation/test sets for training the HeatmapMLP model.
        params:
            labels_source: a string, either 'wyscout' or 'espn', indicating which dataset to use for labels
            separate_by_events: a bool indicating whether or not to group coords based on event id's. 
            If false, all event coords are placed into one list, and resulting heatmaps are of shape (1, 50, 50).
            If true, event coords are partitioned into separate lists, and resulting heatmaps are of shape (8, 50, 50). 
            train_test_frac: a float indicating what portion of the dataset is to be used for the test set.
            train_val_frac: a float indicating what portion of the non-test data is to be used for the validation set.
            verbose: a bool indicating whether or not print progress
        """
        
        start_time = time.perf_counter()

        # identify valid competitions based on labels source
        competitions = []
        if labels_source == 'espn':
            competitions = ['England', 'France', 'Germany', 'Italy', 'Spain']
        else:
            competitions = ['England', 'European_Championship', 'France', 'Germany', 'Italy', 'Spain', 'World_Cup']

        # get events from all valid competitions
        full_events_df = get_full_events_df(competitions)
        if verbose:
            intermediate_time = time.perf_counter()
            print(f"All competition events imported\t\t ({intermediate_time - start_time:.3f} secs)")

        # get coords
        full_coords_df = CustomData.get_coords_df(events_df = full_events_df, separate_by_events = separate_by_events,
                                                  labels_source = labels_source)
        if verbose:
            intermediate_time = time.perf_counter()
            print(f"Coordinates aggregated\t\t\t ({intermediate_time - start_time:.3f} secs)")

        # turn into coords into heatmap tensors
        X_train_images, y_train, X_val_images, y_val, X_test_images, y_test = CustomData._get_train_val_test_tensors(full_coords_df, separate_by_events, train_test_frac, train_val_frac)
        if verbose:
            intermediate_time = time.perf_counter()
            print(f"Converted to heatmaps\t\t\t ({intermediate_time - start_time:.3f} secs)")

        # turn into HeatmapDataset objects
        train_image_dataset = HeatmapDataset(X_train_images, y_train)
        val_image_dataset = HeatmapDataset(X_val_images, y_val)
        test_image_dataset = HeatmapDataset(X_test_images, y_test)
        if verbose:
            intermediate_time = time.perf_counter()
            print(f"Converted to HeatmapDataset objects\t ({intermediate_time - start_time:.3f} secs)")

        end_time = time.perf_counter()
        print(f"\nExecution time: {end_time - start_time:.3f} secs")

        return train_image_dataset, val_image_dataset, test_image_dataset

