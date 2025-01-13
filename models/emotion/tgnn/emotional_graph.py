"""
Emotional Graph Neural Network for processing emotional relationships and social interactions.

Key components:
1. Graph-based emotional state representation
2. Social relationship modeling through node connections  
3. Emotional state propagation across the network
4. Integration with consciousness development

Dependencies:
- torch.nn.Module for neural network implementation
- models/evaluation/emotional_evaluation.py for metrics
- models/memory/emotional_memory_core.py for memory storage

The EmotionalGraphNN class uses a Graph Convolutional Network (GCN) to:
- Model emotional relationships between agents
- Process multimodal emotional inputs
- Enable social learning through graph message passing
- Support consciousness emergence through emotional interactions
"""

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