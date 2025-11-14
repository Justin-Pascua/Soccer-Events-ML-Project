import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import os
os.environ['TORCH'] = torch.__version__

class HeatmapAutoencoder(nn.Module):
    def __init__(self, latent_dim = 128):
        super().__init__()
        self.encoding_stack = nn.Sequential(
            nn.Flatten(start_dim = -3),
            nn.Linear(50*50, latent_dim*4),
            nn.Tanh(),
            nn.Dropout(p = 0.3),
            nn.Linear(latent_dim*4, latent_dim*2),
            nn.Tanh(),
            nn.Dropout(p = 0.3),
            nn.Linear(latent_dim*2, latent_dim),
        )
        self.decoding_stack = nn.Sequential(
            nn.Linear(latent_dim, latent_dim*2),
            nn.Tanh(),
            nn.Dropout(p = 0.3),
            nn.Linear(latent_dim*2, latent_dim*4),
            nn.Tanh(),
            nn.Dropout(p = 0.3),
            nn.Linear(latent_dim*4, 50*50),
            nn.Unflatten(dim = -1, unflattened_size = (1, 50, 50))
        )

    def forward(self, data: tuple | list | torch.Tensor):
        """
        Performs a forward pass given data.
        params:
            data: a tuple or list where the first element is a torch.Tensor of shape (1, 50, 50) or (batch_size, 1, 50, 50) (to accommodate DataLoader inputs)
            or a torch.Tensor of shape (1, 50, 50) or (batch_size, 1, 50, 50)
        """
        x = None
        if type(data) == tuple or type(data) == list:
            x = data[0]
        else:
            x = data
        
        hidden = self.encoding_stack(x)
        out = self.decoding_stack(hidden)
        return out
    
    def encode(self, data):
        return self.encoding_stack(data)
    
    def decode(self, hidden):
        return self.decoding_stack(hidden)
        
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
            loss = self.loss_fn(out, batch[0])     # compute error
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

        total_loss = 0.
        with torch.no_grad():
            for batch in loader:
                out = self(batch)
                total_loss += self.loss_fn(out, batch[0]).item()
            
        return total_loss/len(loader.dataset)

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
        train_loss_history = []
        val_loss_history = []

        for t in range(epochs):
            print(f"Epoch {t+1}", end = " | ")

            # perform training step
            self._train_step(train_loader)

            # get current metrics
            train_loss = self._eval_metrics(train_loader)
            val_loss = self._eval_metrics(val_loader)

            train_loss_history.append(train_loss)
            val_loss_history.append(val_loss)
            
            if(verbose):    
                print(f"training loss: {train_loss:.3e}", end = " | ")
                print(f"val loss: {val_loss:.3e}")
            print(' ')

        print("Done!")

        return {'train_loss': train_loss_history, 'val_loss': val_loss_history}
    
