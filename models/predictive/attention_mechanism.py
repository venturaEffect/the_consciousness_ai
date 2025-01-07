"""
Predictive Attention Mechanism for ACM Project

Implements advanced attention processing for multimodal data.
Supports visualization, debugging, and flexible configurations.
"""

import torch
from torch.nn import MultiheadAttention
import torch.nn as nn
import numpy as np
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass

@dataclass
class AttentionMetrics:
    """Tracks attention-related metrics"""
    attention_level: float = 0.0
    focus_duration: float = 0.0
    context_relevance: float = 0.0
    emotional_salience: float = 0.0

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

class ConsciousnessAttention(nn.Module):
    """
    Implements attention mechanisms for consciousness development:
    1. Survival-based attention activation
    2. Emotional salience detection
    3. Context-aware focus
    """
    
    def __init__(self, config: Dict):
        super().__init__()
        
        # Configuration
        self.hidden_size = config.get('hidden_size', 256)
        self.num_heads = config.get('num_heads', 8)
        self.dropout = config.get('dropout', 0.1)
        
        # Core attention components
        self.survival_attention = nn.MultiheadAttention(
            embed_dim=self.hidden_size,
            num_heads=self.num_heads,
            dropout=self.dropout
        )
        
        self.emotional_attention = nn.MultiheadAttention(
            embed_dim=self.hidden_size,
            num_heads=self.num_heads,
            dropout=self.dropout
        )
        
        # Projections
        self.stress_projection = nn.Linear(self.hidden_size, 1)
        self.emotional_projection = nn.Linear(self.hidden_size, 1)
        self.context_projection = nn.Linear(self.hidden_size, self.hidden_size)
        
        # Metrics tracking
        self.metrics = AttentionMetrics()
        
    def forward(
        self,
        input_state: torch.Tensor,
        emotional_context: torch.Tensor,
        environment_context: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Process input through attention mechanisms
        
        Args:
            input_state: Current state tensor
            emotional_context: Emotional embeddings
            environment_context: Optional environmental context
            attention_mask: Optional attention mask
        """
        batch_size = input_state.size(0)
        
        # Survival-based attention
        survival_attention, survival_weights = self.survival_attention(
            query=input_state,
            key=input_state,
            value=input_state,
            attn_mask=attention_mask
        )
        
        # Calculate stress level
        stress_level = torch.sigmoid(self.stress_projection(survival_attention))
        
        # Emotional attention with context
        emotional_attention, emotional_weights = self.emotional_attention(
            query=emotional_context,
            key=input_state,
            value=input_state,
            attn_mask=attention_mask
        )
        
        # Calculate emotional salience
        emotional_salience = torch.sigmoid(
            self.emotional_projection(emotional_attention)
        )
        
        # Combine attention mechanisms
        if environment_context is not None:
            context_features = self.context_projection(environment_context)
            combined_attention = survival_attention + emotional_attention + context_features
        else:
            combined_attention = survival_attention + emotional_attention
            
        # Update metrics
        self.update_metrics(
            stress_level=stress_level,
            emotional_salience=emotional_salience,
            survival_weights=survival_weights,
            emotional_weights=emotional_weights
        )
        
        return combined_attention, self.get_metrics()
        
    def update_metrics(
        self,
        stress_level: torch.Tensor,
        emotional_salience: torch.Tensor,
        survival_weights: torch.Tensor,
        emotional_weights: torch.Tensor
    ):
        """Update attention metrics"""
        
        # Calculate attention level based on stress and emotion
        self.metrics.attention_level = float(
            torch.mean(stress_level * emotional_salience).item()
        )
        
        # Calculate focus duration
        self.metrics.focus_duration = float(
            torch.mean(torch.sum(survival_weights, dim=1)).item()
        )
        
        # Calculate emotional salience
        self.metrics.emotional_salience = float(
            torch.mean(emotional_salience).item()
        )
        
        # Calculate context relevance
        self.metrics.context_relevance = float(
            torch.mean(torch.sum(emotional_weights, dim=1)).item()
        )
        
    def get_metrics(self) -> Dict:
        """Get current attention metrics"""
        return {
            'attention_level': self.metrics.attention_level,
            'focus_duration': self.metrics.focus_duration,
            'context_relevance': self.metrics.context_relevance,
            'emotional_salience': self.metrics.emotional_salience
        }
