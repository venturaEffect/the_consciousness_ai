"""
Emotional reward shaping for ACM consciousness development.
Integrates with LLaMA 3.3 narrative states and meta-memory system.

Key features:
- Emotion-based reward modulation
- Meta-memory reinforcement
- Controlled adaptation rates
- Narrative coherence rewards
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional
from dataclasses import dataclass

from models.emotion.tgnn.emotional_graph import EmotionalGraphNetwork


@dataclass
class RewardMetrics:
    """Track reward shaping metrics."""
    emotional_coherence: float = 0.0
    memory_influence: float = 0.0
    narrative_alignment: float = 0.0
    adaptation_rate: float = 0.0


class EmotionalRewardShaper(nn.Module):
    """
    Shapes rewards based on emotional responses and learning progress.
    """

    def __init__(self, config: Dict):
        """
        Initializes the reward shaping system.

        Args:
            config: Dictionary containing reward shaping parameters:
                - 'emotional_dims': Size of the input emotion vector
                - 'hidden_size': Embedding dimension
                - 'reward': Sub-dict with 'base_scale', 'memory_influence', 'coherence_weight'
        """
        super().__init__()

        # Core components.
        self.emotion_encoder = nn.Linear(
            config['emotional_dims'],
            config['hidden_size']
        )

        self.memory_gate = nn.Sequential(
            nn.Linear(config['hidden_size'] * 2, config['hidden_size']),
            nn.GELU(),
            nn.Linear(config['hidden_size'], 1),
            nn.Sigmoid()
        )

        # Configuration.
        reward_cfg = config.get('reward', {})
        self.base_reward_scale = reward_cfg.get('base_scale', 1.0)
        self.memory_influence = reward_cfg.get('memory_influence', 0.5)
        self.coherence_weight = reward_cfg.get('coherence_weight', 0.5)

        # Metrics tracking.
        self.metrics = RewardMetrics()

        self.valence_weight = config.get("valence_weight", 0.1)
        self.dominance_weight = config.get("dominance_weight", 0.05)
        self.arousal_penalty = config.get("arousal_penalty", 0.1)
        self.arousal_threshold = config.get("arousal_threshold", 0.8)

    def compute_reward(
        self,
        emotion_values: Dict[str, float],
        attention_level: float,
        meta_memory: Optional[Dict] = None
    ) -> float:
        """
        Compute the shaped reward based on emotional context.

        Args:
            emotion_values: Dict of emotional signals (valence, arousal, etc.).
            attention_level: Current attention level or weighting factor.
            meta_memory: Additional memory-based data or patterns.

        Returns:
            Shaped reward value (float).
        """
        # Encode emotions into embeddings
        emotional_embedding = self._encode_emotions(emotion_values)
        base_reward = self._calculate_base_reward(emotional_embedding)

        # Apply memory influence
        if meta_memory:
            memory_gate_val = self._calculate_memory_influence(
                emotional_embedding, 
                meta_memory
            )
            base_reward *= (1.0 + memory_gate_val)

        # Modulate by attention
        return base_reward * (1.0 + attention_level)

    def _encode_emotions(self, emotion_values: Dict[str, float]) -> torch.Tensor:
        """
        Encode emotional values into a tensor for further processing.
        Placeholder logic; adjust as needed.
        """
        # Example: sorted keys for deterministic ordering.
        keys = sorted(emotion_values.keys())
        vec = torch.tensor([emotion_values[k] for k in keys], dtype=torch.float).unsqueeze(0)
        return self.emotion_encoder(vec).squeeze(0)

    def _calculate_base_reward(self, emotional_embedding: torch.Tensor) -> float:
        """
        Derive a base reward from the emotional embedding.
        Placeholder logic: sum the embedding and scale.
        """
        base_val = torch.sum(emotional_embedding).item()
        return float(base_val * self.base_reward_scale)

    def _calculate_memory_influence(
        self,
        emotional_embedding: torch.Tensor,
        meta_memory: Dict
    ) -> float:
        """
        Compute how meta-memory influences the reward.
        Placeholder logic: feed combined embedding to a gating net.
        """
        # Dummy memory embedding from meta_memory; or your real approach.
        memory_vec = torch.tensor(
            [meta_memory.get('stability_score', 0.5)],
            dtype=torch.float
        )
        combined = torch.cat([emotional_embedding, memory_vec], dim=0)
        gate_val = self.memory_gate(combined.unsqueeze(0)).squeeze(0).item()
        return float(gate_val * self.memory_influence)

    def _update_metrics(
        self,
        emotional_embedding: torch.Tensor,
        base_reward: float,
        attention_level: float
    ) -> None:
        """
        Update reward shaping metrics with placeholder logic.
        """
        self.metrics.emotional_coherence = float(torch.norm(emotional_embedding).item())
        self.metrics.memory_influence = float(base_reward)
        self.metrics.narrative_alignment = 0.0  # Adjust if you integrate narratives.
        self.metrics.adaptation_rate = attention_level

    def compute_emotional_reward(self, emotion_values, base_reward=1.0, context=None):
        """
        Calculate reward based on emotional values with historical context
        """
        # Start with base reward
        reward = base_reward
        
        # Apply core emotional modulation
        if 'valence' in emotion_values:
            reward += emotion_values['valence'] * self.config['valence_weight']
        if 'dominance' in emotion_values:
            reward += emotion_values['dominance'] * self.config['dominance_weight']
        
        # Apply arousal penalty for high stress (if configured)
        if 'arousal' in emotion_values and self.config.get('arousal_penalty', 0) > 0:
            if emotion_values['arousal'] > self.config.get('arousal_threshold', 0.7):
                penalty = (emotion_values['arousal'] - self.config['arousal_threshold']) * self.config['arousal_penalty']
                reward -= penalty
                
        # NEW: Incorporate historical trend analysis 
        if context and 'emotional_history' in context and len(context['emotional_history']) > 5:
            recent_emotions = context['emotional_history'][-5:]
            # Reward improvement in emotional state
            valence_trend = sum(e.get('valence', 0) for e in recent_emotions) / len(recent_emotions)
            valence_delta = emotion_values.get('valence', 0) - valence_trend
            reward += valence_delta * self.config.get('trend_weight', 0.1)
        
        # NEW: Apply adaptation bonus when appropriate
        if context and context.get('adaptation_detected', False):
            reward += self.config.get('adaptation_bonus', 0.2)
            
        return reward
