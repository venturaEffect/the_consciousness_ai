"""
Attention and consciousness gating system for ACM

This module controls information flow through:
1. Stress-modulated attention gating
2. Emotional salience weighting 
3. Survival-driven attention mechanisms
4. Integration with consciousness development

Dependencies:
- models/emotion/emotional_processing.py for affect detection
- models/memory/emotional_memory_core.py for memory storage
- configs/consciousness_development.yaml for parameters
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple
from dataclasses import dataclass

@dataclass
class GatingMetrics:
    """Tracks comprehensive gating performance"""
    attention_activation: float = 0.0
    emotional_salience: float = 0.0
    stress_response: float = 0.0
    temporal_coherence: float = 0.0
    memory_relevance: float = 0.0
    gating_efficiency: float = 0.0

class AdaptiveGatingNetwork(nn.Module):
    """
    Implements adaptive gating based on multiple context factors
    """
    
    def __init__(self, config: Dict):
        super().__init__()
        
        # Context processing networks
        self.emotion_processor = nn.Sequential(
            nn.Linear(config['emotion_dim'], config['hidden_dim']),
            nn.LayerNorm(config['hidden_dim']),
            nn.GELU(),
            nn.Linear(config['hidden_dim'], config['gate_dim'])
        )
        
        self.memory_processor = nn.Sequential(
            nn.Linear(config['memory_dim'], config['hidden_dim']),
            nn.LayerNorm(config['hidden_dim']),
            nn.GELU(),
            nn.Linear(config['hidden_dim'], config['gate_dim'])
        )
        
        # Gating networks
        self.gate_generator = nn.Sequential(
            nn.Linear(config['gate_dim'] * 3, config['hidden_dim']),
            nn.LayerNorm(config['hidden_dim']),
            nn.GELU(),
            nn.Linear(config['hidden_dim'], 1),
            nn.Sigmoid()
        )

    def forward(
        self,
        emotional_context: torch.Tensor,
        memory_context: torch.Tensor,
        attention_level: float
    ) -> torch.Tensor:
        """Generate adaptive gating signal"""
        # Process contexts
        emotional_features = self.emotion_processor(emotional_context)
        memory_features = self.memory_processor(memory_context)
        
        # Combine features
        combined_features = torch.cat([
            emotional_features,
            memory_features,
            torch.tensor([attention_level])
        ])
        
        # Generate gate values
        return self.gate_generator(combined_features)

class ConsciousnessGating(nn.Module):
    """
    Main gating module for consciousness development
    """
    
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        
        # Initialize gating components
        self.attention_network = AttentionNetwork(
            input_dim=config.model.hidden_size,
            attention_dim=config.model.attention_dim
        )
        
        # Stress and emotion modulation
        self.stress_modulation = StressModulation(config)
        self.emotional_weighting = EmotionalWeighting(config)
        
        # Gating networks
        self.adaptive_gate = AdaptiveGatingNetwork(config)
        self.memory_gate = TemporalMemoryGate(config)
        self.attention_modulator = AttentionModulationNetwork(config)
        
        # Fusion network
        self.gate_fusion = GateFusion(config)
        
        # Metrics tracking
        self.metrics = GatingMetrics()
        
        # Adaptive thresholds
        self.min_attention = config.get('min_attention_threshold', 0.5)
        self.base_threshold = config.get('base_threshold', 0.7)
        self.adaptation_rate = config.get('threshold_adaptation_rate', 0.1)

    def forward(
        self,
        input_state: torch.Tensor,
        emotional_context: torch.Tensor,
        stress_level: Optional[float] = None,
        current_state: torch.Tensor = None,
        memory_context: Optional[torch.Tensor] = None,
        attention_level: float = 0.0
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Process input through enhanced gating mechanism
        """
        # Calculate base attention
        attention_weights = self.attention_network(input_state)
        
        # Modulate with stress if provided
        if stress_level is not None:
            attention_weights = self.stress_modulation(
                attention_weights,
                stress_level
            )
        
        # Generate base gating signal
        base_gate = self.adaptive_gate(
            emotional_context=torch.tensor([v for v in emotional_context.values()]),
            memory_context=memory_context if memory_context is not None else torch.zeros(self.config['memory_dim']),
            attention_level=attention_level
        )
        
        # Apply memory-based modulation
        if memory_context is not None:
            memory_gate = self.memory_gate(
                current_state=current_state,
                memory_context=memory_context
            )
            base_gate = base_gate * memory_gate
            
        # Modulate with attention
        attention_modulation = self.attention_modulator(
            attention_level=attention_level,
            stress_level=stress_level
        )
        
        gated_output = current_state * base_gate * attention_modulation
        
        # Update metrics
        self._update_metrics(
            base_gate=base_gate,
            memory_gate=memory_gate if memory_context is not None else None,
            attention_modulation=attention_modulation,
            emotional_context=emotional_context,
            stress_level=stress_level
        )
        
        return gated_output, self.get_metrics()