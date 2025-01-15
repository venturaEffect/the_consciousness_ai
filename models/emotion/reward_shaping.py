# models/emotion/reward_shaping.py

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional
from dataclasses import dataclass
from models.emotion.tgnn.emotional_graph import EmotionalGraphNetwork

"""
Emotional reward shaping for ACM consciousness development.
Integrates with LLaMA 3.3 narrative states and meta-memory system.

Key features:
- Emotion-based reward modulation
- Meta-memory reinforcement
- Controlled adaptation rates
- Narrative coherence rewards

Dependencies:
- models/emotion/tgnn/emotional_graph.py for emotion processing
- models/evaluation/emotional_rl_metrics.py for progress tracking
"""

@dataclass
class RewardMetrics:
    """Track reward shaping metrics"""
    emotional_coherence: float = 0.0
    memory_influence: float = 0.0
    narrative_alignment: float = 0.0
    adaptation_rate: float = 0.0

class EmotionalRewardShaper(nn.Module):
    """Shapes rewards based on emotional responses and learning progress"""
    
    def __init__(self, config):
        """Initialize reward shaping system"""
        super().__init__()
        
        # Core components
        self.emotion_encoder = nn.Linear(
            config.emotional_dims,
            config.hidden_size
        )
        
        self.memory_gate = nn.Sequential(
            nn.Linear(config.hidden_size * 2, config.hidden_size),
            nn.GELU(),
            nn.Linear(config.hidden_size, 1),
            nn.Sigmoid()
        )
        
        # Configuration
        self.base_reward_scale = config.reward.base_scale
        self.memory_influence = config.reward.memory_influence
        self.coherence_weight = config.reward.coherence_weight
        
        # Metrics tracking
        self.metrics = RewardMetrics()
        
    def compute_reward(
        self,
        emotion_values: Dict[str, float],
        attention_level: float,
        context: Optional[Dict] = None,
        meta_memory: Optional[Dict] = None
    ) -> float:
        """Compute shaped reward based on emotional context"""
        
        # Get base emotional reward
        emotional_embedding = self._encode_emotions(emotion_values)
        base_reward = self._calculate_base_reward(emotional_embedding)
        
        # Apply memory influence if available
        if meta_memory:
            memory_gate = self._calculate_memory_influence(
                emotional_embedding,
                meta_memory
            )
            base_reward = base_reward * (1.0 + memory_gate)
            
        # Apply attention modulation
        attention_modulated = base_reward * (1.0 + attention_level)
        
        # Update metrics
        self._update_metrics(
            emotional_embedding,
            base_reward,
            attention_level
        )
        
        return attention_modulated