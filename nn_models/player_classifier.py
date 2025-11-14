import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import os
os.environ['TORCH'] = torch.__version__

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



