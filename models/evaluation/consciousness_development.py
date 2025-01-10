# models/evaluation/consciousness_development.py

import torch
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass
from models.emotion.reward_shaping import EmotionalRewardShaper
from models.memory.memory_core import MemoryCore
from models.evaluation.consciousness_metrics import ConsciousnessMetrics
from models.predictive.dreamer_emotional_wrapper import DreamerEmotionalWrapper
from models.self.self_representation_core import SelfRepresentationCore
from models.social.social_learning_pipeline import SocialLearningPipeline

@dataclass
class DevelopmentMetrics:
    """Tracks consciousness development metrics"""
    emotional_awareness: float = 0.0
    memory_coherence: float = 0.0
    attention_level: float = 0.0
    behavioral_adaptation: float = 0.0
    survival_success: float = 0.0

class ConsciousnessDevelopment:
    """
    Manages and evaluates consciousness development through:
    1. Survival-driven attention mechanisms
    2. Emotional reinforcement learning
    3. Memory formation and coherence
    4. Behavioral adaptation
    """
    
    def __init__(self, config: Dict):
        self.config = config
        
        # Core components
        self.dreamer = DreamerEmotionalWrapper(config)
        self.reward_shaper = EmotionalRewardShaper(config)
        self.memory = MemoryCore(config['memory_config'])
        self.consciousness_metrics = ConsciousnessMetrics(config)
        self.self_model = SelfRepresentationCore(config)
        self.social_learning = SocialLearningPipeline(config)
        
        # Development tracking
        self.metrics = DevelopmentMetrics()
        self.experience_history = []
        
    def process_experience(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        reward: float,
        next_state: torch.Tensor,
        emotion_values: Dict[str, float],
        attention_level: float,
        done: bool
    ) -> Dict:
        """Process a single experience for consciousness development"""
        
        # Shape reward based on emotional response and attention
        shaped_reward = self.reward_shaper.compute_reward(
            emotion_values=emotion_values,
            attention_level=attention_level,
            context={
                'state': state,
                'action': action
            }
        )
        
        # Update DreamerV3 with emotional context
        learning_info = self.dreamer.process_interaction(
            state=state,
            action=action,
            reward=shaped_reward,
            next_state=next_state,
            emotion_values=emotion_values,
            done=done
        )
        
        # Store experience in memory
        self.store_experience(
            state=state,
            action=action,
            reward=shaped_reward,
            emotion=emotion_values,
            attention=attention_level
        )
        
        # Update development metrics
        self.update_metrics(
            emotion_values=emotion_values,
            attention_level=attention_level,
            learning_info=learning_info
        )
        
        return {
            'shaped_reward': shaped_reward,
            'metrics': self.get_metrics(),
            'learning_info': learning_info
        }
        
    def store_experience(self, **kwargs):
        """Store experience with emotional and attention context"""
        self.memory.store_experience(kwargs)
        self.experience_history.append(kwargs)
        
    def update_metrics(
        self,
        emotion_values: Dict[str, float],
        attention_level: float,
        learning_info: Dict
    ):
        """Update consciousness development metrics"""
        # Update emotional awareness
        self.metrics.emotional_awareness = self.consciousness_metrics.evaluate_emotional_awareness(
            self.experience_history[-100:]
        )['mean_emotional_awareness']
        
        # Update memory coherence
        self.metrics.memory_coherence = self.consciousness_metrics.evaluate_memory_coherence()['temporal_coherence']
        
        # Update attention level
        self.metrics.attention_level = attention_level
        
        # Update behavioral adaptation
        self.metrics.behavioral_adaptation = learning_info.get('adaptation_score', 0.0)
        
        # Update survival success
        self.metrics.survival_success = self.calculate_survival_success()
        
    def calculate_survival_success(self) -> float:
        """Calculate success rate in survival scenarios"""
        if not self.experience_history:
            return 0.0
            
        recent_experiences = self.experience_history[-100:]
        success_count = sum(1 for exp in recent_experiences if exp.get('survival_success', False))
        return success_count / len(recent_experiences)
        
    def get_metrics(self) -> Dict:
        """Get current development metrics"""
        return {
            'emotional_awareness': self.metrics.emotional_awareness,
            'memory_coherence': self.metrics.memory_coherence,
            'attention_level': self.metrics.attention_level,
            'behavioral_adaptation': self.metrics.behavioral_adaptation,
            'survival_success': self.metrics.survival_success
        }

    def evaluate_development(
        self,
        current_state: Dict,
        social_interactions: List[Dict],
        attention_metrics: Dict[str, float]
    ):
        # Process current experiences
        for interaction in social_interactions:
            self.social_learning.process_interaction(
                interaction_data=interaction,
                emotion_values=current_state['emotion'],
                attention_level=attention_metrics['attention']
            )
            
        # Update development metrics
        self.metrics.update(
            self_model_coherence=self.self_model.get_coherence(),
            social_learning_progress=self.social_learning.get_progress(),
            attention_stability=attention_metrics['stability']
        )