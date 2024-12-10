"""
Predictive Attention Mechanism for ACM Project

Implements advanced attention processing for multimodal data.
Supports visualization, debugging, and flexible configurations.
"""

import torch
from torch.nn import MultiheadAttention


class PredictiveAttention(torch.nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        """
        Initialize Predictive Attention Mechanism.
        Args:
            embed_dim (int): Dimension of input embeddings.
            num_heads (int): Number of attention heads.
            dropout (float): Dropout rate for regularization.
        """
        super(PredictiveAttention, self).__init__()
        self.attention = MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)

    def forward(self, query, key, value, mask=None):
        """
        Forward pass for the attention mechanism.
        Args:
            query (Tensor): Query tensor.
            key (Tensor): Key tensor.
            value (Tensor): Value tensor.
            mask (Tensor): Optional attention mask.
        Returns:
            Tuple: (attention output, attention weights)
        """
        attn_output, attn_weights = self.attention(query, key, value, attn_mask=mask)
        return attn_output, attn_weights

    @staticmethod
    def visualize_attention(attn_weights, labels=None):
        """
        Visualize attention weights using heatmaps.
        Args:
            attn_weights (Tensor): Attention weights matrix.
            labels (list): Optional labels for axes.
        """
        import matplotlib.pyplot as plt
        import seaborn as sns

        sns.heatmap(attn_weights.squeeze().cpu().detach().numpy(), xticklabels=labels, yticklabels=labels, cmap="coolwarm")
        plt.title("Attention Weights")
        plt.xlabel("Keys")
        plt.ylabel("Queries")
        plt.show()
