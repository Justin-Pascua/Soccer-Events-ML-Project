import numpy as np
import pandas as pd
import ast

import torch
import torchvision
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt
import seaborn as sns
import time

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from scipy import sparse

from torcheval.metrics import MulticlassF1Score

import os
os.environ['TORCH'] = torch.__version__

from local_data_handlers.wyscout_data_handler import get_full_events_df
from local_data_handlers.wyscout_metadata_handler import get_eventids_df

import heatmap_classifier

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
        full_coords_df = heatmap_classifier.CustomData.get_coords_df(events_df = full_events_df, separate_by_events = separate_by_events,
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
            transform = heatmap_classifier.CustomTransforms.multi_coords_list_to_heatmap
        else:
            transform = heatmap_classifier.CustomTransforms.coords_list_to_heatmap
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

class PlayerClassifier(nn.Module):
    def __init__(self, num_classes = 11):
        """
        Initializes model.
        params:
            num_classes: number of possible class labels
        """
        super().__init__()
        self.heatmap_stack = nn.Sequential(
            nn.Flatten(start_dim = -3),
            nn.Linear(50*50, 256),
            nn.ReLU(),
            nn.Dropout(p = 0.3),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(p = 0.3),
        )
        self.subevent_count_stack = nn.Sequential(
            nn.Linear(33, 128),
            nn.ReLU(),
            nn.Dropout(p = 0.3),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(p = 0.3),
        )
        self.combined_stack = nn.Sequential(
            nn.Linear(256+128, 128),
            nn.ReLU(),
            nn.Dropout(p = 0.3),
        )
        self.final = nn.Linear(128, num_classes) 

    def forward(self, data: tuple, get_hidden: bool = False):
        """
        Performs a forward pass given data.
        params:
            data: a triple of the form (heatmap, subevent_counts, label), 
            get_hidden: a bool indicating whether or not to get the hidden layer's output
        """
        heatmap, subevent_counts, _ = data

        heatmap_embedding = self.heatmap_stack(heatmap)
        
        subevent_counts_embedding = self.subevent_count_stack(subevent_counts)
        
        combined_features = torch.cat([heatmap_embedding, subevent_counts_embedding], dim = -1)
        
        hidden = self.combined_stack(combined_features)
        
        logits = self.final(hidden)
        if get_hidden:
            return hidden
        else:
            return logits
    
    def predict(self, data):
        self.eval()
        logits = self(data)
        probabilities = logits.softmax(dim = 1)
        predictions = probabilities.argmax(dim = 1)
        return predictions
    
    def compile(self, optimizer, loss_fn):
        """
        Sets the model's optimizer and loss function.
        params:
            optimizer: an optimizer function such as those found in torch.optim. This optimizer should already have the model's parameters.
            loss_fn: a loss function such as those found in torch.nn.
        """
        self.optimizer = optimizer
        self.loss_fn = loss_fn

    def _check_if_compiled(self):
        """
        Checks if the model's optimizer and loss function have been set.
        """
        assert ((self.optimizer is not None) and (self.loss_fn is not None)), "Please set optimizer and loss function using model.compile(optimizer, loss_fn)"

    def _train_step(self, train_loader: DataLoader):
        """
        Performs a single training loop over a given dataloader.
        params:
            train_loader: a torch.utils.data.DataLoader object equipped with data appropriate for the given model.
        """
        self._check_if_compiled()   # raise error if optimizer and loss function haven't been configured
        self.train()                # set training mode

        for batch in train_loader:
            out = self(batch)                   # forward pass
            loss = self.loss_fn(out, batch[-1])  # compute error
            loss.backward()                     # backward pass
            self.optimizer.step()               # backprop
            self.optimizer.zero_grad()          # reset gradients

    def _eval_metrics(self, loader: DataLoader):
        """
        Evaluates model accuracy and loss on a given dataloader.
        params:
            loader: a torch.utils.data.DataLoader object equipped with data appropriate for the given model.
        """
        self._check_if_compiled()   # raise error if optimizer and loss function haven't been configured
        self.eval()                 # set eval mode

        correct, loss = 0, 0
        with torch.no_grad():
            for batch in loader:
                out = self(batch)
                loss += self.loss_fn(out, batch[-1]).item()
                pred = out.softmax(dim = 1).argmax(dim = 1)
                correct += (pred == batch[-1]).sum().item()
            
        return correct/(len(loader.dataset)), loss/len(loader.dataset)

    def _get_hidden(self, loader: DataLoader):
        """
        Gets hidden layer outputs on a given dataloader.
        params:
            loader: a torch.utils.data.DataLoader object equipped with data appropriate for the given model.
        """
        self.eval()

        with torch.no_grad():
            full_arr = torch.tensor([])
            for data in loader:
                hidden = self(data, get_hidden = True)
                y = data[-1]
                current_arr = torch.cat([hidden, y.unsqueeze(1)], dim = 1)
                full_arr = torch.cat([full_arr, current_arr], dim = 0)
        return full_arr

    def fit(self, train_loader: DataLoader, val_loader: DataLoader, epochs: int, verbose: bool = True):
        """
        Fits the model to a given training dataset while also tracking metrics on both the training set and a provided validation set.
        params:
            train_loader: a torch.utils.data.DataLoader object equipped with data appropriate for the given model.
            val_loader: a torch.utils.data.DataLoader object equipped with data appropriate for the given model.
            epochs: an int specifying how many times to cover the training set
            verbose: a bool indicating whether or not to print training progress
        """
        self._check_if_compiled()   # raise error if optimizer and loss function haven't been configured
        train_acc_history = []
        train_loss_history = []
        val_acc_history = []
        val_loss_history = []
        hidden_history = []

        for t in range(epochs):
            print(f"Epoch {t+1}", end = " | ")

            # perform training step
            self._train_step(train_loader)

            # get current metrics
            train_acc, train_loss = self._eval_metrics(train_loader)
            val_acc, val_loss = self._eval_metrics(val_loader)

            train_acc_history.append(train_acc)
            train_loss_history.append(train_loss)
            val_acc_history.append(val_acc)
            val_loss_history.append(val_loss)
            
            if(verbose):    
                print(f"training accuracy: {train_acc:.3f} - training loss: {train_loss:.3e}", end = " | ")
                print(f"val accuracy: {val_acc:.3f} - val loss: {val_loss:.3e}")
            print(' ')

            if((t+1)%5 == 0):
                hidden_history.append(self._get_hidden(train_loader))


        print("Done!")

        return {'train_acc': train_acc_history, 'train_loss': train_loss_history, 
                'val_acc': val_acc_history, 'val_loss': val_loss_history, 
                'hidden_history': hidden_history}



