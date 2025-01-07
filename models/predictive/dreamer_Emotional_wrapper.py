# models/predictive/dreamer_emotional_wrapper.py

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
    """Tracks emotional learning metrics"""
    valence: float = 0.0
    arousal: float = 0.0
    dominance: float = 0.0
    reward_history: List[float] = None
    consciousness_score: float = 0.0

class DreamerEmotionalWrapper:
    """
    Integrates DreamerV3 with emotional learning capabilities for ACM
    """
    
    def __init__(self, config: Dict):
        # Core components
        self.config = config
        self.dreamer = DreamerV3(config['dreamer_config'])
        self.emotion_network = EmotionalGraphNetwork()
        self.reward_shaper = EmotionalRewardShaper(config)
        self.memory = MemoryCore(config['memory_config'])
        self.consciousness_metrics = ConsciousnessMetrics(config)
        
        # Initialize metrics
        self.metrics = EmotionalMetrics(
            reward_history=[]
        )
        
        # Training parameters
        self.world_model_lr = config.get('world_model_lr', 1e-4)
        self.actor_lr = config.get('actor_lr', 8e-5)
        self.critic_lr = config.get('critic_lr', 8e-5)
        self.gamma = config.get('gamma', 0.99)
        
    def process_interaction(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        reward: float,
        next_state: torch.Tensor,
        emotion_values: Dict[str, float],
        done: bool
    ) -> Dict:
        """Process interaction with emotional context"""
        
        # Update emotional state
        self.update_emotional_state(emotion_values)
        
        # Get emotional embedding
        emotional_embedding = self.emotion_network.get_embedding(emotion_values)
        
        # Shape reward using emotional context
        shaped_reward = self.reward_shaper.compute_reward(
            emotion_values=emotion_values,
            learning_progress=self.calculate_learning_progress(),
            context={
                'state': state,
                'action': action,
                'emotional_embedding': emotional_embedding
            }
        )
        
        # Store experience
        self.store_experience(
            state=state,
            action=action,
            reward=shaped_reward,
            next_state=next_state,
            emotion_values=emotion_values,
            done=done
        )
        
        # Update world model with emotional context
        world_model_loss = self.dreamer.update_world_model(
            state=state,
            action=action,
            reward=shaped_reward,
            next_state=next_state,
            done=done,
            additional_context=emotional_embedding
        )
        
        # Update actor-critic with emotional weighting
        actor_loss, critic_loss = self.dreamer.update_actor_critic(
            state=state,
            action=action,
            reward=shaped_reward,
            next_state=next_state,
            done=done,
            importance_weight=emotion_values.get('valence', 1.0)
        )
        
        # Update consciousness metrics
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
        """Update internal emotional state tracking"""
        self.metrics.valence = emotion_values.get('valence', self.metrics.valence)
        self.metrics.arousal = emotion_values.get('arousal', self.metrics.arousal)
        self.metrics.dominance = emotion_values.get('dominance', self.metrics.dominance)
        
    def calculate_learning_progress(self) -> float:
        """Calculate recent learning progress"""
        if not self.metrics.reward_history:
            return 0.0
        recent_rewards = self.metrics.reward_history[-100:]
        return np.mean(np.diff(recent_rewards))
        
    def store_experience(self, **kwargs):
        """Store experience with emotional context"""
        self.memory.store_experience(kwargs)
        if 'reward' in kwargs:
            self.metrics.reward_history.append(kwargs['reward'])
            
    def get_emotional_state(self) -> Dict:
        """Get current emotional state"""
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
        """Get action with emotional context consideration"""
        if emotion_context is not None:
            emotional_embedding = self.emotion_network.get_embedding(emotion_context)
            action = self.dreamer.get_action(
                state, 
                additional_context=emotional_embedding
            )
        else:
            action = self.dreamer.get_action(state)
        return action

    def save_checkpoint(self, path: str):
        """Save model checkpoint"""
        checkpoint = {
            'dreamer_state': self.dreamer.state_dict(),
            'emotion_network_state': self.emotion_network.state_dict(),
            'metrics': self.metrics,
            'config': self.config
        }
        torch.save(checkpoint, path)
        
    def load_checkpoint(self, path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(path)
        self.dreamer.load_state_dict(checkpoint['dreamer_state'])
        self.emotion_network.load_state_dict(checkpoint['emotion_network_state'])
        self.metrics = checkpoint['metrics']
        self.config = checkpoint['config']

    def _layer(self, x):
        try:
            shape = (x.shape[-1], int(np.prod(self.units)))
            if not all(dim > 0 for dim in shape):
                raise ValueError("Invalid shape dimensions")
            x = x @ self.get('kernel', self._winit, shape).astype(x.dtype)
            return x
        except Exception as e:
            raise RuntimeError(f"Layer computation failed: {str(e)}")