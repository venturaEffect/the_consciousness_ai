"""
Self-Awareness Evaluation Module

Implements comprehensive metrics for evaluating self-awareness through:
1. Emotional state recognition
2. Behavioral pattern analysis
3. Social interaction assessment
4. Temporal consistency evaluation

Based on holonic principles where metrics contribute both independently 
and to overall self-awareness evaluation.
"""

import torch
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass

@dataclass
class SelfAwarenessMetrics:
    """Tracks self-awareness development metrics"""
    emotional_recognition: float = 0.0
    behavioral_consistency: float = 0.0
    social_understanding: float = 0.0
    temporal_coherence: float = 0.0

class SelfAwarenessEvaluator:
    """
    Evaluates development of self-awareness across multiple dimensions
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.metrics = SelfAwarenessMetrics()

    def evaluate_self_awareness(
        self,
        self_model_state: Dict,
        interaction_history: List[Dict],
        emotional_context: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Comprehensive self-awareness evaluation
        
        Args:
            self_model_state: Current self-representation state
            interaction_history: Recent interaction records
            emotional_context: Current emotional state
        """
        # Evaluate emotional recognition
        emotional_recognition = self._evaluate_emotional_recognition(
            self_model_state,
            emotional_context
        )
        
        # Evaluate behavioral consistency
        behavioral_consistency = self._evaluate_behavioral_consistency(
            interaction_history
        )
        
        # Evaluate social understanding
        social_understanding = self._evaluate_social_understanding(
            interaction_history
        )
        
        # Evaluate temporal coherence
        temporal_coherence = self._evaluate_temporal_coherence(
            self_model_state,
            interaction_history
        )
        
        # Update metrics
        self.metrics.emotional_recognition = emotional_recognition
        self.metrics.behavioral_consistency = behavioral_consistency
        self.metrics.social_understanding = social_understanding
        self.metrics.temporal_coherence = temporal_coherence
        
        return self.get_metrics()

    def _evaluate_emotional_recognition(
        self,
        self_model_state: Dict,
        emotional_context: Dict[str, float]
    ) -> float:
        """Evaluate accuracy of emotional state recognition"""
        if not self_model_state or not emotional_context:
            return 0.0
            
        predicted_emotions = self_model_state.get('emotional_state', {})
        
        # Calculate alignment between predicted and actual emotions
        alignment_scores = []
        for emotion, value in emotional_context.items():
            if emotion in predicted_emotions:
                alignment = 1 - abs(value - predicted_emotions[emotion])
                alignment_scores.append(alignment)
                
        return np.mean(alignment_scores) if alignment_scores else 0.0