"""
Attention Gating Module for the Artificial Consciousness Module (ACM)

This module implements:
1. Attention-based information gating for consciousness emergence
2. Stress-modulated attention control
3. Emotional salience weighting
4. Memory formation triggers

Dependencies:
- models/emotion/tgnn/emotional_graph.py for emotion processing
- models/memory/emotional_memory_core.py for memory storage
- configs/consciousness_development.yaml for parameters

Key Components:
- AttentionGating: Main gating mechanism for consciousness
- StressModulation: Modulates attention based on stress
- EmotionalWeighting: Weights information by emotional salience
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple

class AttentionGating(nn.Module):
    def __init__(self, config: Dict):
        """Initialize attention gating system"""
        super().__init__()
        self.config = config
        
        # Initialize attention layers
        self.attention_network = nn.Sequential(
            nn.Linear(config.hidden_size, config.attention_size),
            nn.Tanh(),
            nn.Linear(config.attention_size, 1)
        )
        
        # Initialize modulation components
        self.stress_modulation = StressModulation(config)
        self.emotional_weighting = EmotionalWeighting(config)
        
    def forward(
        self,
        input_features: torch.Tensor,
        emotional_context: Optional[torch.Tensor] = None,
        stress_level: Optional[float] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Process input through attention gating"""
        # Calculate base attention
        attention_weights = self.attention_network(input_features)
        
        # Apply stress modulation if provided
        if stress_level is not None:
            attention_weights = self.stress_modulation(
                attention_weights,
                stress_level
            )
            
        # Apply emotional weighting if context provided
        if emotional_context is not None:
            attention_weights = self.emotional_weighting(
                attention_weights,
                emotional_context
            )
            
        # Return gated output and metrics
        return attention_weights * input_features, {
            'attention_level': attention_weights.mean().item(),
            'stress_modulation': stress_level if stress_level else 0.0,
            'emotional_weight': self.emotional_weighting.current_weight
        }