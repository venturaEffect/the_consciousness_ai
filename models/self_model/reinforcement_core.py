# models/self_model/reinforcement_core.py

import torch
import numpy as np
from models.predictive.dreamerv3_wrapper import DreamerV3
from models.memory.memory_core import MemoryCore
from models.narrative.narrative_engine import NarrativeEngine
from models.self_model.emotion_context_tracker import EmotionContextTracker
from models.self_model.belief_system import BeliefSystem

class ReinforcementCore:
    def __init__(self, config):
        # Core components
        self.memory = MemoryCore()
        self.dreamer = DreamerV3(config.dreamer_config)
        self.narrative = NarrativeEngine()
        self.emotion_tracker = EmotionContextTracker()
        self.belief_system = BeliefSystem()
        
        # Configuration
        self.config = config
        self.emotional_scale = config.emotional_scale
        self.positive_emotion_bonus = config.positive_emotion_bonus
        
        # Meta-learning setup
        self.meta_learning = config.meta_config.enabled
        if self.meta_learning:
            self.adaptation_steps = config.meta_config.adaptation_steps
            self.inner_lr = config.meta_config.inner_learning_rate
            
    def compute_reward(self, state, emotion_values, action_info):
        """
        Compute reward based on emotional response and state
        Args:
            state: Current environment state
            emotion_values: Dict of emotion measurements
            action_info: Information about the taken action
        """
        # Get emotional valence from tracker
        emotional_reward = self.emotion_tracker.get_emotional_value(emotion_values)
        
        # Apply emotion-based scaling
        scaled_reward = emotional_reward * self.emotional_scale
        
        # Add bonus for positive emotions to reinforce good interactions
        if emotional_reward > 0:
            scaled_reward += self.positive_emotion_bonus
            
        # Store experience with emotional context
        experience = {
            'state': state,
            'emotion': emotion_values,
            'action': action_info,
            'reward': scaled_reward,
            'narrative': self.narrative.generate_experience_narrative(
                state, emotion_values, scaled_reward
            )
        }
        self.memory.store_experience(experience)
        
        return scaled_reward

    def update(self, state, action, reward, next_state, done, emotion_context):
        """
        Update the model using DreamerV3 with emotional context
        """
        # Create world model training batch
        world_model_batch = self.dreamer.create_batch(
            state, action, reward, next_state, done,
            additional_context=emotion_context
        )
        
        # Update world model with emotional context
        world_model_loss = self.dreamer.update_world_model(
            world_model_batch, 
            emotion_context=emotion_context
        )
        
        # Update actor-critic with emotional weighting
        actor_loss, critic_loss = self.dreamer.update_actor_critic(
            world_model_batch,
            emotion_scale=self.emotional_scale
        )
        
        # Update belief system based on experience
        belief_update = self.belief_system.update(
            state, action, reward, emotion_context
        )
        
        # Generate narrative description of update
        narrative = self.narrative.generate_experience_narrative(
            state=state,
            action=action, 
            reward=reward,
            emotion=self.emotion_tracker.current_emotion,
            belief_update=belief_update
        )
        
        return {
            'world_model_loss': world_model_loss,
            'actor_loss': actor_loss,
            'critic_loss': critic_loss, 
            'narrative': narrative,
            'belief_update': belief_update
        }

    def meta_adapt(self, task):
        """
        Adapt to new task using meta-learning if enabled
        """
        if not self.meta_learning:
            return

        # Get relevant experiences for the task
        task_experiences = self.memory.get_relevant_experiences(task)
        
        # Perform quick adaptation using MAML-style update
        for _ in range(self.adaptation_steps):
            batch = self.memory.sample_batch(task_experiences)
            self.dreamer.inner_update(batch, self.inner_lr)