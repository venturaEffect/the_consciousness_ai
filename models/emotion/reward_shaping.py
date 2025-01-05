# models/emotion/reward_shaping.py

import torch
import numpy as np
from typing import Dict, Optional
from models.emotion.tgnn.emotional_graph import EmotionalGraphNetwork

class EmotionalRewardShaper:
    """Shapes rewards based on emotional responses and learning progress"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.emotion_network = EmotionalGraphNetwork()
        
        # Reward scaling parameters
        self.base_scale = config.get('emotional_scale', 2.0)
        self.positive_bonus = config.get('positive_emotion_bonus', 0.5)
        self.learning_scale = config.get('learning_progress_scale', 0.3)
        
    def compute_reward(
        self,
        emotion_values: Dict[str, float],
        learning_progress: Optional[float] = None,
        context: Optional[Dict] = None
    ) -> float:
        """
        Compute shaped reward based on emotional response
        
        Args:
            emotion_values: Dict of emotion measurements
            learning_progress: Optional measure of learning improvement
            context: Optional additional context for reward shaping
        """
        # Get base emotional reward
        base_reward = self._compute_base_reward(emotion_values)
        
        # Scale based on learning progress if available
        if learning_progress is not None:
            base_reward *= (1.0 + self.learning_scale * learning_progress)
            
        # Apply positive emotion bonus
        if self._is_positive_emotion(emotion_values):
            base_reward += self.positive_bonus
            
        # Apply context-specific scaling
        if context is not None:
            base_reward = self._apply_context_scaling(base_reward, context)
            
        return base_reward
        
    def _compute_base_reward(self, emotion_values: Dict[str, float]) -> float:
        """Compute base reward from emotion values"""
        # Weight different emotion components
        valence = emotion_values.get('valence', 0.0) 
        arousal = emotion_values.get('arousal', 0.0)
        dominance = emotion_values.get('dominance', 0.0)
        
        # Combine emotional components with learned weights
        base_reward = (
            0.5 * valence +  # Higher weight on valence
            0.3 * arousal +  # Medium weight on arousal
            0.2 * dominance  # Lower weight on dominance
        )
        
        return base_reward * self.base_scale
        
    def _is_positive_emotion(self, emotion_values: Dict[str, float]) -> bool:
        """Check if emotion state is positive"""
        valence = emotion_values.get('valence', 0.0)
        return valence > 0.6  # Threshold for positive emotion
        
    def _apply_context_scaling(self, reward: float, context: Dict) -> float:
        """Apply context-specific reward scaling"""
        # Scale based on interaction type
        if 'interaction_type' in context:
            if context['interaction_type'] == 'teaching':
                reward *= 1.2  # Boost learning interactions
            elif context['interaction_type'] == 'social':
                reward *= 1.1  # Slightly boost social interactions
                
        # Scale based on task difficulty
        if 'difficulty' in context:
            reward *= (1.0 + 0.1 * context['difficulty'])
            
        return reward