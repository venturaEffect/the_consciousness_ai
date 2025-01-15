"""
Consciousness gating mechanism that controls information flow and adaptation
in the ACM system. Implements controlled learning rates and meta-memory stability.

Key components:
- Attention-based gating for information flow
- Meta-memory stability tracking
- Controlled adaptation mechanisms
- Integration with LLaMA 3.3 narrator
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple
from dataclasses import dataclass

@dataclass
class GatingState:
    """Track gating mechanism state"""
    attention_level: float = 0.0
    stability_score: float = 0.0
    adaptation_rate: float = 0.0
    meta_memory_coherence: float = 0.0
    narrator_confidence: float = 0.0

class ConsciousnessGate(nn.Module):
    def __init__(self, config):
        """Initialize gating mechanism"""
        super().__init__()
        
        # Core gating parameters
        self.attention_threshold = config.gating.attention_threshold
        self.stability_threshold = config.gating.stability_threshold
        self.adaptation_rate = config.gating.base_adaptation_rate
        
        # Initialize components
        self.attention_net = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.GELU(),
            nn.Linear(config.hidden_size, 1),
            nn.Sigmoid()
        )
        
        self.stability_net = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.GELU(),
            nn.Linear(config.hidden_size, 1),
            nn.Sigmoid()
        )
        
        # State tracking
        self.state = GatingState()
        
    def forward(
        self,
        input_state: torch.Tensor,
        meta_memory_context: Optional[Dict] = None,
        narrator_state: Optional[Dict] = None
    ) -> Tuple[torch.Tensor, GatingState]:
        """Process input through gating mechanism"""
        
        # Calculate attention level
        attention_level = self.attention_net(input_state)
        
        # Calculate stability score
        stability_score = self.stability_net(input_state)
        
        # Adjust adaptation rate based on stability
        adaptation_rate = self._calculate_adaptation_rate(
            stability_score,
            meta_memory_context
        )
        
        # Apply gating
        gated_output = self._apply_gating(
            input_state,
            attention_level,
            stability_score
        )
        
        # Update state
        self._update_state(
            attention_level,
            stability_score,
            adaptation_rate,
            narrator_state
        )
        
        return gated_output, self.state
        
    def _calculate_adaptation_rate(
        self,
        stability_score: torch.Tensor,
        meta_memory_context: Optional[Dict]
    ) -> float:
        """Calculate controlled adaptation rate"""
        base_rate = self.adaptation_rate
        
        if meta_memory_context:
            # Reduce adaptation for stable patterns
            if meta_memory_context.get('stable_patterns'):
                base_rate *= 0.5
                
            # Increase adaptation for novel experiences
            if meta_memory_context.get('novel_experiences'):
                base_rate *= 2.0
                
        return base_rate * stability_score.item()