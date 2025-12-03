import torch
import torch.nn as nn
from torch.utils.data import Dataset

import time

from sklearn.model_selection import train_test_split

import os
os.environ['TORCH'] = torch.__version__

from internal_data_handlers.wyscout_data_handler import get_full_events_df
from internal_data_handlers.wyscout_metadata_handler import get_eventids_df

import nn_models.heatmap_data as heatmap_data

eventids_df = get_eventids_df(verbose = False)
valid_subeventids = eventids_df[~eventids_df['event'].isin([5,6])]['subevent'].values

class CustomData:
    @staticmethod
    def get_subevent_counts(events_df):
        temp_df = events_df.copy()
    
        # filter out interruptions and offsides
        temp_df = temp_df[~ temp_df['eventId'].isin([5,6])]
        
        # get subevent count features
        temp_df['subEventId'] = temp_df['subEventId'].astype('Int64')
        subevent_counts_df = temp_df.groupby(['matchId', 'teamId', 'playerId'])['subEventId'].value_counts().unstack(fill_value = 0)

        # fill absent subevents with fill value 0
        subevent_counts_df = subevent_counts_df.reindex(columns = valid_subeventids, fill_value = 0)

        return subevent_counts_df

    @staticmethod
    def df_to_tensors(features_df, subevent_cols):
        heatmaps_tensor = torch.stack(list(features_df['heatmap'].values))
        
        subevent_counts_tensor = torch.tensor(features_df[subevent_cols].values.astype(float), dtype = torch.float32)

        targets_tensor = torch.tensor(features_df['numericalLabel'].values)
        
        return heatmaps_tensor, subevent_counts_tensor, targets_tensor

    @staticmethod
    def get_datasets(labels_source: str = None, separate_by_events: bool = False,
                     train_test_frac: float = 0.2, train_val_frac: float = 0.2, 
                     verbose: bool = True):
        
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
        full_coords_df = heatmap_data.CustomData.get_coords_df(events_df = full_events_df, separate_by_events = separate_by_events,
                                                  labels_source = labels_source)
        if verbose:
            intermediate_time = time.perf_counter()
            print(f"Coordinates aggregated\t\t\t ({intermediate_time - start_time:.3f} secs)")

        # get subevent counts
        subevent_counts = CustomData.get_subevent_counts(full_events_df)
        subevent_cols = subevent_counts.columns
        if verbose:
            intermediate_time = time.perf_counter()
            print(f"Subevent counts obtained\t\t ({intermediate_time - start_time:.3f} secs)")

        # convert coords to heatmaps
        features_df = full_coords_df.merge(subevent_counts, how = 'inner', left_index = True, right_index = True)
        transform = None
        if separate_by_events:
            transform = heatmap_data.CustomTransforms.multi_coords_list_to_heatmap
        else:
            transform = heatmap_data.CustomTransforms.coords_list_to_heatmap
        features_df['heatmap'] = features_df['coordsList'].apply(transform)
        if verbose:
            intermediate_time = time.perf_counter()
            print(f"Converted coords to heatmaps\t\t ({intermediate_time - start_time:.3f} secs)")


        # split into train/val/test
        train_df, test_df = train_test_split(features_df, test_size = train_test_frac)
        train_df, val_df = train_test_split(train_df, test_size = train_val_frac)

        # convert to tensors
        train_heatmaps_tensor, train_subevent_counts_tensor, train_targets_tensor = CustomData.df_to_tensors(train_df, subevent_cols)
        val_heatmaps_tensor, val_subevent_counts_tensor, val_targets_tensor = CustomData.df_to_tensors(val_df, subevent_cols)
        test_heatmaps_tensor, test_subevent_counts_tensor, test_targets_tensor = CustomData.df_to_tensors(test_df, subevent_cols)

        # convert to datasets
        train_player_dataset = PlayerDataset(train_heatmaps_tensor, train_subevent_counts_tensor, train_targets_tensor)
        val_player_dataset = PlayerDataset(val_heatmaps_tensor, val_subevent_counts_tensor, val_targets_tensor)
        test_player_dataset = PlayerDataset(test_heatmaps_tensor, test_subevent_counts_tensor, test_targets_tensor)

        if verbose:
            intermediate_time = time.perf_counter()
            print(f"Converted to PlayerDatasets\t\t ({intermediate_time - start_time:.3f} secs)")

        end_time = time.perf_counter()
        print(f"\nExecution time: {end_time - start_time:.3f} secs")
        return train_player_dataset, val_player_dataset, test_player_dataset

class PlayerDataset(Dataset):
    def __init__(self, heatmaps_arr, event_counts_arr, targets_arr):
        self.heatmaps = heatmaps_arr
        self.event_counts = event_counts_arr
        self.targets = targets_arr

    def __len__(self):
        return len(self.targets)
    
    def __getitem__(self, i):
        heatmap = self.heatmaps[i]
        event_counts = self.event_counts[i]
        y = self.targets[i]
        return heatmap, event_counts, y
    
    def __str__(self):
        output = f"""PlayerDataset:
        - length: {len(self)}
        - sample shapes: {tuple(e.shape for e in self[0])}
        """
        return output

