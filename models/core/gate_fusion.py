"""
Gate Fusion Module

Implements fusion of multiple gating mechanisms for consciousness development:
1. Attention gate integration
2. Emotional salience weighting
3. Stress response modulation 
4. Temporal coherence maintenance

Based on the holonic MANN architecture where each component functions both 
independently and as part of the whole system.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

@dataclass
class FusionMetrics:
    """Tracks gate fusion performance"""
    attention_weight: float = 0.0
    emotional_weight: float = 0.0 
    stress_weight: float = 0.0
    temporal_weight: float = 0.0
    fusion_quality: float = 0.0

class GateFusion(nn.Module):
    """
    Fuses multiple gating signals into coherent consciousness control
    
    Key Features:
    1. Adaptive weighting of different gates
    2. Temporal stability maintenance
    3. Dynamic fusion based on current context
    4. Meta-learning for weight optimization
    """

    def __init__(self, config: Dict):
        super().__init__()
        
        # Gate weighting networks
        self.attention_weighting = nn.Sequential(
            nn.Linear(config['state_dim'], config['hidden_dim']),
            nn.LayerNorm(config['hidden_dim']),
            nn.GELU(),
            nn.Linear(config['hidden_dim'], 1),
            nn.Sigmoid()
        )
        
        self.emotional_weighting = nn.Sequential(
            nn.Linear(config['emotion_dim'], config['hidden_dim']),
            nn.LayerNorm(config['hidden_dim']), 
            nn.GELU(),
            nn.Linear(config['hidden_dim'], 1),
            nn.Sigmoid()
        )
        
        # Fusion layers
        self.fusion_network = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=config['state_dim'],
                nhead=config['n_heads']
            ) for _ in range(config['n_fusion_layers'])
        ])
        
        self.metrics = FusionMetrics()

    def forward(
        self,
        attention: torch.Tensor,
        emotional: torch.Tensor,
        stress: Optional[torch.Tensor] = None,
        temporal: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Fuse multiple gating signals
        
        Args:
            attention: Attention gate output
            emotional: Emotional gate output
            stress: Optional stress gate output
            temporal: Optional temporal coherence gate output
        """
        # Get gate weights
        attention_weight = self.attention_weighting(attention)
        emotional_weight = self.emotional_weighting(emotional)
        
        # Combine weighted gates
        gates = [
            attention * attention_weight,
            emotional * emotional_weight
        ]
        
        if stress is not None:
            stress_weight = self.stress_weighting(stress)
            gates.append(stress * stress_weight)
            
        if temporal is not None:
            temporal_weight = self.temporal_weighting(temporal)
            gates.append(temporal * temporal_weight)
            
        # Fuse through transformer layers
        fused = torch.cat(gates, dim=-1)
        for layer in self.fusion_network:
            fused = layer(fused)
            
        # Update metrics
        self._update_metrics(
            attention_weight=attention_weight,
            emotional_weight=emotional_weight,
            stress_weight=stress_weight if stress is not None else None,
            temporal_weight=temporal_weight if temporal is not None else None,
            fused=fused
        )
        
        return fused, self.get_metrics()

    def _update_metrics(
        self,
        attention_weight: torch.Tensor,
        emotional_weight: torch.Tensor,
        stress_weight: Optional[torch.Tensor] = None,
        temporal_weight: Optional[torch.Tensor] = None,
        fused: Optional[torch.Tensor] = None
    ):
        """Update fusion metrics"""
        self.metrics.attention_weight = attention_weight.mean().item()
        self.metrics.emotional_weight = emotional_weight.mean().item()
        
        if stress_weight is not None:
            self.metrics.stress_weight = stress_weight.mean().item()
            
        if temporal_weight is not None:
            self.metrics.temporal_weight = temporal_weight.mean().item()
            
        if fused is not None:
            self.metrics.fusion_quality = self._calculate_fusion_quality(fused)

    def _calculate_fusion_quality(self, fused: torch.Tensor) -> float:
        """Calculate quality of gate fusion"""
        # Measure stability and coherence of fused output
        stability = torch.std(fused, dim=0).mean().item()
        coherence = torch.corrcoef(fused.T)[0,1].item()
        return (stability + coherence) / 2