class EmotionalGraphNN(torch.nn.Module):
    def __init__(self, num_features, hidden_dim, num_classes):
        super(EmotionalGraphNN, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim // 2)
        self.fc = torch.nn.Linear(hidden_dim // 2, num_classes)

    def forward(self, x, edge_index, edge_attr=None, multimodal_context=None):
        x = self.conv1(x, edge_index, edge_weight=edge_attr)
        x = F.relu(x)
        if multimodal_context is not None:
            x += multimodal_context
        x = self.conv2(x, edge_index, edge_weight=edge_attr)
        x = F.relu(x)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)