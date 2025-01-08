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
    """Tracks attention-related metrics for consciousness development"""
    attention_level: float = 0.0
    focus_duration: float = 0.0
    emotional_salience: float = 0.0
    context_relevance: float = 0.0
    stress_adaptation: float = 0.0

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
    Enhanced attention mechanism for consciousness development through:
    1. Stress-induced attention activation
    2. Emotional salience detection
    3. Memory-guided attention
    4. Temporal context integration
    """
    
    def __init__(self, config: Dict):
        super().__init__()
        
        # Configuration
        self.hidden_size = config.get('hidden_size', 768)
        self.num_heads = config.get('num_heads', 12)
        self.dropout = config.get('dropout', 0.1)
        self.stress_sensitivity = config.get('stress_sensitivity', 2.0)
        
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
        
        # Memory attention for temporal context
        self.memory_attention = nn.MultiheadAttention(
            embed_dim=self.hidden_size,
            num_heads=self.num_heads,
            dropout=self.dropout
        )
        
        # Projections
        self.stress_projection = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, 1)
        )
        
        self.emotional_projection = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, 1)
        )
        
        self.context_projection = nn.Linear(self.hidden_size, self.hidden_size)
        
        # Fusion layer
        self.attention_fusion = nn.Sequential(
            nn.Linear(self.hidden_size * 3, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size)
        )
        
        # Metrics tracking
        self.metrics = AttentionMetrics()
        self.attention_history = []
        
    def forward(
        self,
        input_state: torch.Tensor,
        emotional_context: torch.Tensor,
        memory_context: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Process input through enhanced attention mechanisms
        
        Args:
            input_state: Current state tensor
            emotional_context: Emotional embeddings
            memory_context: Optional memory context for temporal integration
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
        
        # Calculate stress level with enhanced sensitivity
        stress_level = torch.sigmoid(
            self.stress_projection(survival_attention) * self.stress_sensitivity
        )
        
        # Emotional attention with enhanced context
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
        
        # Memory-guided attention if context provided
        if memory_context is not None:
            memory_attention, memory_weights = self.memory_attention(
                query=input_state,
                key=memory_context,
                value=memory_context,
                attn_mask=attention_mask
            )
            temporal_context = self.context_projection(memory_attention)
        else:
            temporal_context = torch.zeros_like(input_state)
            memory_weights = None
        
        # Fuse attention streams
        combined_attention = self.attention_fusion(
            torch.cat([
                survival_attention,
                emotional_attention,
                temporal_context
            ], dim=-1)
        )
        
        # Update metrics
        self.update_metrics(
            stress_level=stress_level,
            emotional_salience=emotional_salience,
            survival_weights=survival_weights,
            emotional_weights=emotional_weights,
            memory_weights=memory_weights
        )
        
        # Store attention state
        self._store_attention_state(combined_attention, self.metrics)
        
        return combined_attention, self.get_metrics()
    
    def update_metrics(
        self,
        stress_level: torch.Tensor,
        emotional_salience: torch.Tensor,
        survival_weights: torch.Tensor,
        emotional_weights: torch.Tensor,
        memory_weights: Optional[torch.Tensor]
    ):
        """Update attention metrics with enhanced tracking"""
        
        # Calculate attention level with stress-emotion interaction
        self.metrics.attention_level = float(
            torch.mean(stress_level * emotional_salience).item()
        )
        
        # Calculate focus duration from survival weights
        self.metrics.focus_duration = float(
            torch.mean(torch.sum(survival_weights, dim=1)).item()
        )
        
        # Calculate emotional salience
        self.metrics.emotional_salience = float(
            torch.mean(emotional_salience).item()
        )
        
        # Calculate context relevance including memory
        if memory_weights is not None:
            context_weights = torch.mean(
                torch.stack([emotional_weights, memory_weights]), dim=0
            )
        else:
            context_weights = emotional_weights
            
        self.metrics.context_relevance = float(
            torch.mean(torch.sum(context_weights, dim=1)).item()
        )
        
        # Calculate stress adaptation
        self.metrics.stress_adaptation = self._calculate_stress_adaptation(
            stress_level
        )
        
    def _calculate_stress_adaptation(self, stress_level: torch.Tensor) -> float:
        """Calculate adaptation to stress over time"""
        if len(self.attention_history) < 2:
            return 0.0
            
        recent_stress = [
            state['metrics'].attention_level 
            for state in self.attention_history[-10:]
        ]
        stress_change = np.mean(np.diff(recent_stress))
        
        # Higher score for reducing stress
        return float(1.0 / (1.0 + np.exp(stress_change)))
        
    def _store_attention_state(
        self,
        attention: torch.Tensor,
        metrics: AttentionMetrics
    ):
        """Store attention state for temporal analysis"""
        self.attention_history.append({
            'attention': attention.detach(),
            'metrics': metrics
        })
        
        # Maintain history size
        if len(self.attention_history) > 1000:
            self.attention_history = self.attention_history[-1000:]
            
    def get_metrics(self) -> Dict:
        """Get current attention metrics"""
        return {
            'attention_level': self.metrics.attention_level,
            'focus_duration': self.metrics.focus_duration,
            'emotional_salience': self.metrics.emotional_salience,
            'context_relevance': self.metrics.context_relevance,
            'stress_adaptation': self.metrics.stress_adaptation
        }
