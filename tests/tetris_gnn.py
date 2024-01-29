"""
For reference: 
https://github.com/search?q=repo%3Apyg-team%2Fpytorch_geometric+graph+classification&type=code
For code:
https://colab.research.google.com/drive/1I8a0DfQ3fI7Njc62__mVXUlcAleUclnb?usp=sharing
"""

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import torch
from torch.nn import Linear
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv
from torch_cluster import radius_graph


def tetris_as_graph(tetris: torch.Tensor):
    """
    Represent tetris as graphs
    """
    edge_index = radius_graph(tetris, r=1.1, loop=False)

    return Data(pos=tetris, edge_index=edge_index)


def visualize_graph(tetris_graph):
    pos, edge_index = tetris_graph.pos, tetris_graph.edge_index

    # # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot nodes
    ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], c='r', marker='o')

    # Plot edges
    for edge in edge_index.t().tolist():
        ax.plot([pos[edge[0], 0], pos[edge[1], 0]],
                [pos[edge[0], 1], pos[edge[1], 1]],
                [pos[edge[0], 2], pos[edge[1], 2]], 'b-')

    # Set labels and show the plot
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.title('Tetris')
    plt.show()