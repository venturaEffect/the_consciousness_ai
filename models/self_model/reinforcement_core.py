# models/self_model/reinforcement_core.py

import torch
import numpy as np
from models.predictive.dreamerv3_wrapper import DreamerV3
from models.memory.memory_core import MemoryCore
from models.narrative.narrative_engine import NarrativeEngine
from models.self_model.emotion_context_tracker import EmotionContextTracker

class ReinforcementCore:
    def __init__(self, config):
        self.memory = MemoryCore()
        self.dreamer = DreamerV3()
        self.narrative = NarrativeEngine()
        self.emotion_tracker = EmotionContextTracker()
        
        # Emotional reward scaling factor
        self.emotional_scale = config.get('emotional_scale', 1.0)
        
    def compute_reward(self, state, emotion_values):
        """
        Compute reward based on emotional response and state
        """
        # Get emotional valence from tracker
        emotional_reward = self.emotion_tracker.get_emotional_value(emotion_values)
        
        # Scale emotional reward
        scaled_reward = emotional_reward * self.emotional_scale
        
        # Store experience in memory
        self.memory.store_experience({
            'state': state,
            'emotion': emotion_values,
            'reward': scaled_reward
        })
        
        return scaled_reward
        
    def update(self, state, action, reward, next_state, done):
        """
        Update the model using DreamerV3
        """
        # Create world model training batch
        world_model_batch = self.dreamer.create_batch(
            state, action, reward, next_state, done
        )
        
        # Update world model
        world_model_loss = self.dreamer.update_world_model(world_model_batch)
        
        # Update actor-critic 
        actor_loss, critic_loss = self.dreamer.update_actor_critic(world_model_batch)
        
        # Generate narrative description
        narrative = self.narrative.generate_experience_narrative(
            state, action, reward, emotion=self.emotion_tracker.current_emotion
        )
        
        return {
            'world_model_loss': world_model_loss,
            'actor_loss': actor_loss, 
            'critic_loss': critic_loss,
            'narrative': narrative
        }