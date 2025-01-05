# models/predictive/dreamer_emotional_wrapper.py

import torch
import numpy as np
from typing import Dict, Optional, Tuple
from models.predictive.dreamerv3_wrapper import DreamerV3
from models.emotion.tgnn.emotional_graph import EmotionalGraphNetwork
from models.emotion.reward_shaping import EmotionalRewardShaper

class DreamerEmotionalWrapper:
    """Integrates DreamerV3 with emotional learning capabilities"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.dreamer = DreamerV3(config['dreamer_config'])
        self.emotion_network = EmotionalGraphNetwork()
        self.reward_shaper = EmotionalRewardShaper(config)
        
        # Initialize learning parameters
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
        """Process a single interaction with emotional context"""
        
        # Get emotional embedding
        emotional_embedding = self.emotion_network.get_embedding(emotion_values)
        
        # Shape reward based on emotional response
        shaped_reward = self.reward_shaper.compute_reward(
            emotion_values=emotion_values,
            context={'state': state, 'action': action}
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
        
        return {
            'world_model_loss': world_model_loss,
            'actor_loss': actor_loss,
            'critic_loss': critic_loss,
            'shaped_reward': shaped_reward
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
            'config': self.config
        }
        torch.save(checkpoint, path)
        
    def load_checkpoint(self, path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(path)
        self.dreamer.load_state_dict(checkpoint['dreamer_state'])
        self.emotion_network.load_state_dict(checkpoint['emotion_network_state'])
        self.config = checkpoint['config']