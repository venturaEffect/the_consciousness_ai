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
from dataclasses import dataclass, field

@dataclass
class AttentionMetrics:
    """Tracks attention-related metrics for consciousness development"""
    attention_level: float = 0.0
    focus_duration: float = 0.0 
    emotional_salience: float = 0.0
    context_relevance: float = 0.0
    stress_adaptation: float = 0.0

@dataclass
class AttentionState:
    """Tracks attention state and temporal context"""
    current_level: float = 0.0
    baseline: float = 0.0
    decay_rate: float = 0.1
    history: List[float] = field(default_factory=list)
    stress_adaptation: float = 0.0
    emotional_context: Optional[Dict[str, float]] = None
    temporal_coherence: float = 0.0

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
    Enhanced attention mechanism for consciousness development with:
    1. Stress-modulated attention
    2. Emotional context integration
    3. Temporal memory coherence
    4. Adaptive attention thresholds
    """
    
    def __init__(self, config: Dict):
        super().__init__()
        
        # Core attention parameters
        self.hidden_size = config.get('hidden_size', 768)
        self.num_heads = config.get('num_heads', 12)
        self.dropout = config.get('dropout', 0.1)
        
        # Stress-attention coupling
        self.stress_sensitivity = nn.Parameter(
            torch.ones(1) * config.get('stress_sensitivity', 2.0)
        )
        self.attention_baseline = config.get('attention_baseline', 0.5)
        self.min_attention = config.get('min_attention', 0.2)
        
        # Multi-head attention components
        self.query_net = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size)
        )
        
        self.key_net = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size)
        )
        
        self.value_net = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size)
        )
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=self.hidden_size,
            num_heads=self.num_heads,
            dropout=self.dropout
        )
        
        # Emotional context integration
        self.emotional_projection = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size)
        )
        
        # Memory context integration
        self.memory_projection = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size)
        )
        
        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size)
        )
        
        # State tracking
        self.state = AttentionState()
        
    def forward(
        self,
        input_state: torch.Tensor,
        emotional_context: torch.Tensor,
        memory_context: Optional[torch.Tensor] = None,
        stress_level: Optional[float] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Process input through enhanced attention mechanism"""
        
        batch_size = input_state.size(0)
        
        # Project inputs
        query = self.query_net(input_state)
        key = self.key_net(input_state)
        value = self.value_net(input_state)
        
        # Process emotional context
        if emotional_context is not None:
            emotional_features = self.emotional_projection(emotional_context)
            key = key + emotional_features
            value = value + emotional_features
            
        # Integrate memory context
        if memory_context is not None:
            memory_features = self.memory_projection(memory_context)
            key = torch.cat([key, memory_features], dim=1)
            value = torch.cat([value, memory_features], dim=1)
            
        # Calculate attention with temporal masking
        attention_output, attention_weights = self.attention(
            query=query,
            key=key,
            value=value
        )
        
        # Calculate stress-modulated attention level
        if stress_level is not None:
            attention_level = self._calculate_attention_level(stress_level)
        else:
            attention_level = torch.sigmoid(attention_weights.mean())
            
        # Update attention state
        self._update_state(attention_level, emotional_context)
        
        # Project output with residual connection
        output = self.output_projection(
            torch.cat([attention_output, input_state], dim=-1)
        )
        
        return output, self._get_metrics()
        
    def _calculate_attention_level(self, stress_level: float) -> float:
        """Calculate attention level based on stress and adaptation"""
        # Base attention from stress
        base_attention = torch.sigmoid(
            self.stress_sensitivity * torch.tensor(stress_level)
        ).item()
        
        # Add adaptation factor
        adapted_attention = base_attention * (1.0 + self.state.stress_adaptation)
        
        # Ensure minimum attention
        return max(self.min_attention, adapted_attention)
        
    def _update_state(
        self,
        attention_level: float,
        emotional_context: Optional[torch.Tensor]
    ):
        """Update attention state with temporal context"""
        # Update history
        self.state.history.append(attention_level)
        if len(self.state.history) > 1000:
            self.state.history = self.state.history[-1000:]
            
        # Update current level with decay
        self.state.current_level = (
            (1 - self.state.decay_rate) * self.state.current_level +
            self.state.decay_rate * attention_level
        )
        
        # Update baseline
        if len(self.state.history) > 100:
            self.state.baseline = np.mean(self.state.history[-100:])
            
        # Update stress adaptation
        self.state.stress_adaptation = self._calculate_stress_adaptation()
        
        # Update temporal coherence
        self.state.temporal_coherence = self._calculate_temporal_coherence()
        
    def _get_metrics(self) -> Dict[str, float]:
        """Get current attention metrics"""
        return {
            'attention_level': self.state.current_level,
            'attention_baseline': self.state.baseline,
            'stress_adaptation': self.state.stress_adaptation,
            'temporal_coherence': self.state.temporal_coherence,
            'stability': self._calculate_stability()
        }
        
    def _calculate_stability(self) -> float:
        """Calculate attention stability"""
        if len(self.state.history) < 50:
            return 0.0
            
        recent_attention = self.state.history[-50:]
        return float(1.0 / (1.0 + np.std(recent_attention)))
