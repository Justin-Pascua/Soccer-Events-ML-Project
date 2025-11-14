import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from torcheval.metrics import MulticlassF1Score

import os
os.environ['TORCH'] = torch.__version__

from local_data_handlers.wyscout_metadata_handler import wyscout_class_labels
from local_data_handlers.espn_data_handler import final_espn_class_labels


def _plot_training_progress(history: dict, fig, axes, row_num: int):
    """
    Plots the loss and accuracy of the model over the course of training.
    The plots are made as subplots on a given axes object.
    params:
        history: a dictionary with keys {'train_acc', 'train_loss', 'val_acc', 'val_loss'}, and whose values are lists of values tracking the model's accuracy/loss on the train/validation set.
        fig: a matplotlib.figure.Figure object 
        axes: a numpy.ndarray of shape (2, 3) whose elements are of type matplotlib.axes._axes.Axes
        row_num: an integer specifying which row of subplots to use.  
    """
    train_acc_history = history['train_acc'] 
    train_loss_history = history['train_loss'] 
    val_acc_history = history['val_acc']
    val_loss_history = history['val_loss']    
    
    epochs = len(train_acc_history)

    axes[row_num, 0].set_xlabel('Epoch')
    axes[row_num, 0].set_ylabel('Accuracy')
    axes[row_num, 0].set_title('Accuracy')
    axes[row_num, 0].plot([n for n in range(epochs)], train_acc_history,
                color = 'blue', label = 'Training Accuracy')
    axes[row_num, 0].plot([n for n in range(epochs)], val_acc_history,
                color = 'orange', label = 'Validation Accuracy')
    axes[row_num, 0].legend()
    axes[row_num, 0].set_ylim(ymin = 0)
    
    axes[row_num, 1].set_xlabel('Epoch')
    axes[row_num, 1].set_ylabel('Loss')
    axes[row_num, 1].set_title('Loss')
    axes[row_num, 1].plot([n for n in range(epochs)], train_loss_history,
                color = 'blue', label = 'Training Loss')
    axes[row_num, 1].plot([n for n in range(epochs)], val_loss_history,
                color = 'orange', label = 'Validation Loss')
    axes[row_num, 1].legend()
    #axes[row_num, 1].set_ylim(ymin = 0)

def _get_true_and_pred(model: nn.Module, dataloader: DataLoader):
    """
    Retrieves the model's predictions on a given dataset as well as the true labels.
    params:
        model: a nn.Module model whose forward method's signature is forward(data), where data is of the form (input, label)
        dataloader: a torch.utils.data.DataLoader object equipped with data appropriate for the given model.
    """
    y_true, y_pred = [], []
    model.eval()
    with torch.no_grad():
        for data in dataloader:
            logits = model(data)
            preds = logits.softmax(dim = 1).argmax(dim = 1).numpy()
            targets = data[-1].numpy()
            y_true.extend(targets)
            y_pred.extend(preds)
    return y_true, y_pred

def model_training_metrics(model: nn.Module, history: dict, 
                            train_dataloader: DataLoader, val_dataloader: DataLoader,
                            label_source: str = 'wyscout'):
    """
    Creates a subplot which visualizes the evolution of the model's performance over the course of its training history,
    and its final performance on specified training/validation data using confusion matrices. Also prints F1-scores over the training and validation sets
    params:
        model: a nn.Module model whose forward method's signature is forward(data), where data is of the form (input, label)
        history: a dictionary with keys {'train_acc', 'train_loss', 'val_acc', 'val_loss'}, and whose values are lists of values tracking the model's accuracy/loss on the train/validation set.
        train_dataloader: a torch.utils.data.DataLoader object equipped with data appropriate for the given model.
        val_dataloader: a torch.utils.data.DataLoader object equipped with data appropriate for the given model.
    """
    y_true_train, y_pred_train = _get_true_and_pred(model, train_dataloader)
    cm_train = np.round(confusion_matrix(y_true_train, y_pred_train, normalize = 'true'), 2)
    y_true_val, y_pred_val = _get_true_and_pred(model, val_dataloader)
    cm_val = np.round(confusion_matrix(y_true_val, y_pred_val, normalize = 'true'), 2)

    position_labels, num_classes, annot_train, annot_val = None, None, None, None
    if label_source == 'wyscout':
        position_labels = wyscout_class_labels
        annot_train = True
        annot_val = True
    elif label_source == 'espn':
        position_labels = final_espn_class_labels
        
        annot_train = np.diag(np.diag(cm_train)).astype('str')
        annot_train[annot_train == '0.0'] = ''
        
        annot_val = np.diag(np.diag(cm_val)).astype('str')
        annot_val[annot_val == '0.0'] = ''

    num_classes = len(position_labels)

    fig, axes = plt.subplots(nrows = 2, ncols = 2, figsize = (10, 8))
    
    _plot_training_progress(history, fig, axes, 0)
    
    sns.heatmap(cm_train, annot = annot_train, fmt = '', cmap = plt.cm.Oranges, 
                xticklabels = position_labels, yticklabels = position_labels, 
                ax = axes[1, 0])
    axes[1, 0].set_xlabel('Predicted')
    axes[1, 0].set_ylabel("True")
    axes[1, 0].set_title(f"Confusion Matrix On Training Set")

    
    sns.heatmap(cm_val, annot = annot_val, fmt = '', cmap = plt.cm.Oranges, 
                xticklabels = position_labels, yticklabels = position_labels, 
                ax = axes[1, 1])
    axes[1, 1].set_xlabel("Predicted")
    axes[1, 1].set_ylabel("True")
    axes[1, 1].set_title(f"Confusion Matrix On Validation Set")

    macro_score = MulticlassF1Score(num_classes = num_classes, average = 'macro')

    macro_score.update(torch.tensor(y_pred_train), torch.tensor(y_true_train))
    train_macro_f1_score = macro_score.compute()

    macro_score.update(torch.tensor(y_pred_val), torch.tensor(y_true_val))
    val_macro_f1_score = macro_score.compute()

    train_text = f"""Training Metrics:
    - Final Accuracy: {history['train_acc'][-1]:.3f}
    - Macro F1-score: {train_macro_f1_score.item():.3f}"""

    val_text = f"""Validation Metrics:
    - Final Accuracy: {history['val_acc'][-1]:.3f}
    - Macro F1-score: {val_macro_f1_score.item():.3f}"""

    print(train_text)
    print(val_text)

    fig.tight_layout()
    plt.show()

def model_test_metrics(model: nn.Module, test_dataloader: DataLoader, label_source: str = 'wyscout'):
    """
    Evaluates the model's performance by running predictions on a test dataset, plotting confusion matrices, and printing F1-scores. 
    params:
        model: a nn.Module model whose forward method's signature is forward(data), where data is of the form (input, label)
        test_dataloader: a torch.utils.data.DataLoader object equipped with data appropriate for the given model.
    """
    y_true, y_pred = _get_true_and_pred(model, test_dataloader)
    correct = (torch.tensor(y_true) == torch.tensor(y_pred)).sum().item()
    acc = correct/len(test_dataloader.dataset)

    cm = np.round(confusion_matrix(y_true, y_pred, normalize = 'true'), 2)

    position_labels, num_classes, annot = None, None, None
    if label_source == 'wyscout':
        position_labels = wyscout_class_labels
        annot = True
    elif label_source == 'espn':
        position_labels = final_espn_class_labels            
        annot = np.diag(np.diag(cm)).astype('str')
        annot[annot == '0.0'] = ''
    num_classes = len(position_labels)

    sns.heatmap(cm, annot = annot, fmt = '', cmap = plt.cm.Oranges, 
                xticklabels = position_labels, yticklabels = position_labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix On Validation Set")

    macro_score = MulticlassF1Score(num_classes = num_classes, average = 'macro')
    macro_score.update(torch.tensor(y_pred), torch.tensor(y_true))
    macro_f1_score = macro_score.compute()

    metric_text = f"""Test Metrics:
    - Final Accuracy: {acc:.3f}
    - Macro F1-score: {macro_f1_score.item():.3f}"""
    print(metric_text)
    plt.show()

