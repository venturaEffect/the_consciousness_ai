"""
Consciousness Gating Module

Implements attention-based gating mechanisms for consciousness development through:
1. Stress-activated attention gating
2. Emotional salience detection 
3. Temporal coherence maintenance
4. Meta-memory formation gates

Based on the MANN (Modular Artificial Neural Networks) architecture and holonic principles.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple
from dataclasses import dataclass

@dataclass
class GatingMetrics:
    """Tracks gating mechanism performance"""
    attention_activation: float = 0.0
    emotional_salience: float = 0.0
    stress_response: float = 0.0
    temporal_coherence: float = 0.0

class ConsciousnessGating(nn.Module):
    """
    Implements gating mechanisms for consciousness development.
    Controls information flow based on attention, emotion and stress levels.
    """
    
    def __init__(self, config: Dict):
        super().__init__()
        
        # Core networks
        self.attention_gate = AttentionGate(config)
        self.emotional_gate = EmotionalGate(config)
        self.stress_gate = StressGate(config)
        self.temporal_gate = TemporalCoherenceGate(config)
        
        # Fusion layer
        self.gate_fusion = GateFusion(config)
        
        # Metrics tracking
        self.metrics = GatingMetrics()
        
        # Thresholds
        self.attention_threshold = config.get('attention_threshold', 0.7)
        self.stress_threshold = config.get('stress_threshold', 0.8)

    def forward(
        self,
        current_state: torch.Tensor,
        emotional_context: Dict[str, float],
        stress_level: float,
        attention_level: float,
        temporal_context: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Process input through gating mechanisms
        
        Args:
            current_state: Current input state tensor
            emotional_context: Current emotional state values
            stress_level: Current stress level (0-1)
            attention_level: Current attention level (0-1)
            temporal_context: Optional temporal context tensor
        """
        # Process through individual gates
        attention_gate = self.attention_gate(
            current_state, 
            attention_level
        )
        
        emotional_gate = self.emotional_gate(
            current_state,
            emotional_context
        )
        
        stress_gate = self.stress_gate(
            current_state,
            stress_level
        )
        
        temporal_gate = self.temporal_gate(
            current_state,
            temporal_context
        ) if temporal_context is not None else None

        # Fuse gate outputs
        gated_output = self.gate_fusion(
            attention=attention_gate,
            emotional=emotional_gate,
            stress=stress_gate,
            temporal=temporal_gate
        )
        
        # Update metrics
        self._update_metrics(
            attention_level=attention_level,
            emotional_context=emotional_context,
            stress_level=stress_level
        )
        
        return gated_output, self.get_metrics()

    def _update_metrics(
        self,
        attention_level: float,
        emotional_context: Dict[str, float],
        stress_level: float
    ):
        """Update gating mechanism metrics"""
        self.metrics.attention_activation = attention_level
        self.metrics.emotional_salience = self._calculate_emotional_salience(
            emotional_context
        )
        self.metrics.stress_response = stress_level
        self.metrics.temporal_coherence = self._calculate_temporal_coherence()

    def get_metrics(self) -> Dict[str, float]:
        """Get current gating metrics"""
        return {
            'attention_activation': self.metrics.attention_activation,
            'emotional_salience': self.metrics.emotional_salience,
            'stress_response': self.metrics.stress_response,
            'temporal_coherence': self.metrics.temporal_coherence
        }