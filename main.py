from classifiers.player_classifier import CustomData, PlayerClassifier

import model_metrics_and_vis.embedding_visualizer as embedding_visualizer
from model_metrics_and_vis.metrics import Metrics

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

train_player_dataset, val_player_dataset, test_player_dataset = CustomData.get_datasets(labels_source = 'espn', separate_by_events = False)

LEARNING_RATE = 1e-3
BATCH_SIZE = 4096
EPOCHS = 30

train_player_dataloader = DataLoader(train_player_dataset, batch_size = BATCH_SIZE, shuffle = True)
val_player_dataloader = DataLoader(val_player_dataset, batch_size = BATCH_SIZE, shuffle = True)
test_player_dataloader = DataLoader(test_player_dataset, batch_size = BATCH_SIZE, shuffle = True)

player_model = PlayerClassifier(num_classes = 10)
player_model.compile(torch.optim.Adam(player_model.parameters(), lr = LEARNING_RATE),
                    nn.CrossEntropyLoss(weight = torch.tensor([1., 1.2, 1., 1.2, 2., 1., 2., 2., 1., 2.]))
                    # nn.CrossEntropyLoss()
                    )
history = player_model.fit(train_player_dataloader, val_player_dataloader, epochs = EPOCHS)

Metrics.model_training_metrics(player_model, history, 
                               train_player_dataloader, val_player_dataloader, 
                               label_source = 'espn')

embedding_visualizer.plot_embeddings_3d(history['hidden_history'])