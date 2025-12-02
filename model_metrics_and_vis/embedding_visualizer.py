import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
import plotly.express as px

from internal_data_handlers.espn_data_handler import espn_label_decoder, formation_decoder
from internal_data_handlers.wyscout_metadata_handler import wyscout_label_decoder

def plot_embeddings_2d(embedding_history: list):
    """
    Performs PCA to project data embeddings onto 2D plane and visualizes the results as a matplotlib 2d scatterplot.
    params:
        embedding_history: a list of 6 PyTorch tensors of the form [X y] 
        (i.e. the rows represent individual samples, the first columns represent the learned features, 
        and the last column is a numerical class label).
    """
    fig, axes = plt.subplots(nrows = 2, ncols = 3, figsize = (8, 6))
    pca_decomps = []
    for i, embedding in enumerate(embedding_history):
        X_data = embedding[:, :-1]
        y_data = embedding[:, -1].unsqueeze(1).numpy()
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(X_data)
        pca = PCA(n_components = 2)
        pca_decomp = pca.fit_transform(scaled_data)
        pca_decomps.append(np.concat([pca_decomp, y_data], axis = 1))

    for i in range(6):
        axes[i//3, i%3].scatter(x = pca_decomps[i][:, 0], 
                                y = pca_decomps[i][:, 1], 
                                c = pca_decomps[i][:, 2], 
                                s = 0.25,
                                alpha = 0.25,
                                cmap = 'viridis')
        axes[i//3, i%3].set_title(f'Epoch {5*(i+1)}')
    
    plt.tight_layout()
    plt.show()

def plot_embeddings_3d(embedding_history: list):
    """
    Performs PCA to project data embeddings onto 3D and visualizes the results as a matplotlib 3d scatterplot.
    params:
        embedding_history: a list of 6 PyTorch tensors of the form [X y] 
        (i.e. the rows represent individual samples, the first columns represent the learned features, 
        and the last column is a numerical class label).
    """
    fig = plt.figure(figsize = (8, 8))
    pca_decomps = []
    for i, embedding in enumerate(embedding_history):
        X_data = embedding[:, :-1]
        y_data = embedding[:, -1].unsqueeze(1).numpy()
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(X_data)
        pca = PCA(n_components = 3)
        pca_decomp = pca.fit_transform(scaled_data)
        pca_decomps.append(np.concat([pca_decomp, y_data], axis = 1))

    for i in range(6):
        ax = fig.add_subplot(2, 3, i+1, projection = '3d')
        ax.scatter3D(xs = pca_decomps[i][:, 0], 
                    ys = pca_decomps[i][:, 1], 
                    zs = pca_decomps[i][:, 2],
                    c = y_data,
                    s = 0.25,
                    alpha = 0.25,
                    cmap = 'viridis')
        ax.set_title(f'Epoch {5*(i+1)}')

    plt.tight_layout()
    plt.show()

def plot_single_embedding_3d(embedding, label_source: str):
    """
    Performs PCA to project data embeddings onto 3D and visualizes the results as a plotly interactable 3d scatter plot.
    params:
        embedding_history: a PyTorch tensors of the form [X y] 
        (i.e. the rows represent individual samples, the first columns represent the learned features, 
        and the last column is a numerical class label).
        label_source: a string, either 'wyscout' or 'espn' indicating what labeling convention to use.
        If None, then labels will still be used to color points, but will not be annotated with their actual meaning.
    """
    X_data = embedding[:, :-1]
    y_data = embedding[:, -1].unsqueeze(1).numpy()
    
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(X_data)
    
    pca = PCA(n_components = 3)
    pca_decomp = pca.fit_transform(scaled_data)
    pca_decomp = np.concat([pca_decomp, y_data], axis = 1)
    
    embedding_df = pd.DataFrame(pca_decomp)
    embedding_df[3] = embedding_df[3].astype(int)
    if label_source == 'wyscout':
        embedding_df[3] = embedding_df[3].map(wyscout_label_decoder)
    elif label_source == 'espn':
        embedding_df[3] = embedding_df[3].map(espn_label_decoder)
    elif label_source =='espn_formations':
        embedding_df[3] = embedding_df[3].map(formation_decoder)
    else:
        embedding_df[3] = embedding_df[3].astype(str)    

    embedding_df.rename(columns = {3: 'Player Position'}, inplace = True)
    fig = px.scatter_3d(embedding_df, 
                        x = 0, 
                        y = 1, 
                        z = 2, 
                        color = 'Player Position', 
                        opacity = 1)
    fig.update_layout(margin = {'r': 0, 't': 0, 'l': 0, 'b': 0})
    fig.show()


