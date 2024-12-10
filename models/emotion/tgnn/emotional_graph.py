"""
Emotional Graph Neural Network for ACM Project

Processes emotional states and relationships using Graph Convolutional Networks.
"""

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class EmotionalGraphNN(torch.nn.Module):
    def __init__(self, num_features, hidden_dim, num_classes):
        """
        Initialize the Emotional Graph Neural Network.
        Args:
            num_features (int): Number of input features.
            hidden_dim (int): Hidden layer dimensionality.
            num_classes (int): Number of output classes.
        """
        super(EmotionalGraphNN, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim // 2)
        self.fc = torch.nn.Linear(hidden_dim // 2, num_classes)

    def forward(self, x, edge_index):
        """
        Forward pass for the GCN.
        Args:
            x (Tensor): Node feature matrix.
            edge_index (Tensor): Edge index tensor.
        Returns:
            Tensor: Log-softmax of output classes.
        """
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)
