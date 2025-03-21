"""
DreamerV3 Integration Wrapper for Emotional Processing in ACM

Implements:
1. Emotional context integration with DreamerV3
2. Dream-based emotion prediction and simulation
3. Emotional reward shaping for world model learning
4. Integration with consciousness development
"""

import torch
import numpy as np
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass

from models.predictive.dreamerv3_wrapper import DreamerV3
from models.emotion.tgnn.emotional_graph import EmotionalGraphNetwork
from models.emotion.reward_shaping import EmotionalRewardShaper
from models.memory.memory_core import MemoryCore
from models.evaluation.consciousness_metrics import ConsciousnessMetrics


@dataclass
class EmotionalMetrics:
    """Tracks emotional learning metrics."""
    valence: float = 0.0
    arousal: float = 0.0
    dominance: float = 0.0
    reward_history: List[float] = None
    consciousness_score: float = 0.0


@dataclass
class EmotionalDreamState:
    """Tracks emotional state during dream generation."""
    valence: float = 0.0
    arousal: float = 0.0
    dominance: float = 0.0
    attention: float = 0.0


class DreamerEmotionalWrapper:
    """
    Integrates DreamerV3 with emotional learning for ACM.
    """

    def __init__(self, config: Dict):
        """Initialize emotional dreamer wrapper."""
        self.config = config

        # Load DreamerV3 with matching config key.
        self.dreamer = DreamerV3(config['dreamerV3'])

        self.emotion_network = EmotionalGraphNetwork()
        self.reward_shaper = EmotionalRewardShaper(config)
        self.memory = MemoryCore(config['memory_config'])
        self.consciousness_metrics = ConsciousnessMetrics(config)

        self.dream_state = EmotionalDreamState()
        self.metrics = EmotionalMetrics(reward_history=[])

        # Training parameters.
        self.world_model_lr = config.get('world_model_lr', 1e-4)
        self.actor_lr = config.get('actor_lr', 8e-5)
        self.critic_lr = config.get('critic_lr', 8e-5)
        self.gamma = config.get('gamma', 0.99)

        # Default base_reward if missing in config.
        self.base_reward = float(config.get('base_reward', 1.0))

    def process_interaction(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        reward: float,
        next_state: torch.Tensor,
        emotion_values: Dict[str, float],
        done: bool
    ) -> Dict:
        """Process interaction with emotional context."""
        self.update_emotional_state(emotion_values)
        emotional_embedding = self.emotion_network.get_embedding(emotion_values)

        shaped_reward = self.reward_shaper.compute_reward(
            emotion_values=emotion_values,
            learning_progress=self.calculate_learning_progress(),
            context={
                'state': state,
                'action': action,
                'emotional_embedding': emotional_embedding
            }
        )

        self.store_experience(
            state=state,
            action=action,
            reward=shaped_reward,
            next_state=next_state,
            emotion_values=emotion_values,
            done=done
        )

        world_model_loss = self.dreamer.update_world_model(
            state=state,
            action=action,
            reward=shaped_reward,
            next_state=next_state,
            done=done,
            additional_context=emotional_embedding
        )

        actor_loss, critic_loss = self.dreamer.update_actor_critic(
            state=state,
            action=action,
            reward=shaped_reward,
            next_state=next_state,
            done=done,
            importance_weight=emotion_values.get('valence', 1.0)
        )

        consciousness_score = self.consciousness_metrics.evaluate_emotional_awareness(
            interactions=[{
                'state': state,
                'action': action,
                'emotion_values': emotion_values,
                'reward': shaped_reward
            }]
        )

        self.metrics.consciousness_score = consciousness_score['mean_emotional_awareness']

        return {
            'world_model_loss': world_model_loss,
            'actor_loss': actor_loss,
            'critic_loss': critic_loss,
            'shaped_reward': shaped_reward,
            'consciousness_score': consciousness_score,
            'emotional_state': self.get_emotional_state()
        }

    def update_emotional_state(self, emotion_values: Dict[str, float]):
        """Update internal emotional state tracking."""
        self.metrics.valence = emotion_values.get('valence', self.metrics.valence)
        self.metrics.arousal = emotion_values.get('arousal', self.metrics.arousal)
        self.metrics.dominance = emotion_values.get('dominance', self.metrics.dominance)

    def calculate_learning_progress(self) -> float:
        """Calculate recent learning progress from reward history."""
        if not self.metrics.reward_history:
            return 0.0
        recent_rewards = self.metrics.reward_history[-100:]
        return float(np.mean(np.diff(recent_rewards)))

    def store_experience(self, **kwargs):
        """Store experience with emotional context."""
        self.memory.store_experience(kwargs)
        if 'reward' in kwargs:
            self.metrics.reward_history.append(kwargs['reward'])

    def get_emotional_state(self) -> Dict:
        """Return current emotional state."""
        return {
            'valence': self.metrics.valence,
            'arousal': self.metrics.arousal,
            'dominance': self.metrics.dominance,
            'consciousness_score': self.metrics.consciousness_score
        }

    def get_action(
        self,
        state: torch.Tensor,
        emotion_context: Optional[Dict] = None
    ) -> torch.Tensor:
        """Get action with optional emotional context."""
        if emotion_context is not None:
            emotional_embedding = self.emotion_network.get_embedding(emotion_context)
            action = self.dreamer.get_action(state, additional_context=emotional_embedding)
        else:
            action = self.dreamer.get_action(state)
        return action

    def save_checkpoint(self, path: str):
        """Save model checkpoint."""
        checkpoint = {
            'dreamer_state': self.dreamer.state_dict(),
            'emotion_network_state': self.emotion_network.state_dict(),
            'metrics': self.metrics,
            'config': self.config
        }
        torch.save(checkpoint, path)

    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path)
        self.dreamer.load_state_dict(checkpoint['dreamer_state'])
        self.emotion_network.load_state_dict(checkpoint['emotion_network_state'])
        self.metrics = checkpoint['metrics']
        self.config = checkpoint['config']

    def imagine_trajectory(
        self,
        current_state: torch.Tensor,
        emotional_context: Dict[str, float],
        horizon: int = 10
    ) -> Tuple[torch.Tensor, Dict]:
        """Generate imagined trajectory with emotional context."""
        imagined_trajectory = []
        for _ in range(horizon):
            action = self.get_action(current_state, emotional_context)
            next_state = self.dreamer.predict_next_state(current_state, action)
            imagined_trajectory.append((current_state, action, next_state))
            current_state = next_state
        return imagined_trajectory, self.get_emotional_state()

    def process_dream(
        self,
        dream_state: torch.Tensor,
        emotional_context: Optional[Dict] = None
    ) -> Tuple[torch.Tensor, Dict]:
        """Process dream state with optional emotional context."""
        emotional_features = self.emotion_network.extract_features(dream_state)
        self.dream_state = self._update_dream_state(emotional_features, emotional_context)

        shaped_reward = self._shape_emotional_reward(dream_state, self.dream_state)
        reward_scaling = shaped_reward / self.base_reward

        return shaped_reward, {
            'emotional_state': self.dream_state,
            'reward_scaling': reward_scaling
        }

    def _update_dream_state(
        self,
        emotional_features: torch.Tensor,
        emotional_context: Optional[Dict]
    ) -> EmotionalDreamState:
        """Update the dream state using extracted emotional features."""
        updated_state = EmotionalDreamState(
            valence=float(emotional_features[0].item()),
            arousal=float(emotional_features[1].item()) if emotional_features.size(0) > 1 else 0.0,
            dominance=float(emotional_features[2].item()) if emotional_features.size(0) > 2 else 0.0,
            attention=emotional_context.get('attention', 0.0) if emotional_context else 0.0
        )
        return updated_state

    def _shape_emotional_reward(
        self,
        dream_state: torch.Tensor,
        dream_emotional_state: EmotionalDreamState
    ) -> float:
        """Derive an emotional reward from dream state and dream emotional state."""
        # Very basic example: sum of valence and arousal.
        return dream_emotional_state.valence + dream_emotional_state.arousal
