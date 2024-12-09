import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class EmotionalGraphNN(torch.nn.Module):
    def __init__(self, num_features, hidden_dim, num_classes):
        super(EmotionalGraphNN, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim // 2)
        self.fc = torch.nn.Linear(hidden_dim // 2, num_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)