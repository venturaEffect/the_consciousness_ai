# models/evaluation/emotional_evaluation.py

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from models.emotion.tgnn.emotional_graph import EmotionalGraphNetwork
from models.memory.memory_core import MemoryCore
from models.predictive.attention_mechanism import ConsciousnessAttention

@dataclass
class ConsciousnessMetrics:
    """Tracks development of consciousness-like behaviors"""
    emotional_awareness: float = 0.0
    attention_stability: float = 0.0
    memory_coherence: float = 0.0
    survival_adaptation: float = 0.0
    interaction_quality: float = 0.0
    narrative_consistency: float = 0.0

class EmotionalEvaluator:
    """
    Evaluates consciousness development through emotional learning metrics
    """
    def __init__(self, config: Dict):
        self.config = config
        self.emotion_network = EmotionalGraphNetwork()
        self.memory = MemoryCore(config['memory_config'])
        self.attention = ConsciousnessAttention(config)
        
        # Initialize metrics
        self.metrics = ConsciousnessMetrics()
        self.experience_history = []
        
    def evaluate_interaction(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        emotion_values: Dict[str, float],
        attention_level: float,
        narrative: str,
        stress_level: float
    ) -> Dict:
        """Evaluate a single interaction for consciousness development"""
        
        # Process emotional response
        emotional_embedding = self.emotion_network.get_embedding(emotion_values)
        
        # Get attention metrics
        attention_metrics = self.attention.forward(
            input_state=state,
            emotional_context=emotional_embedding,
            environment_context=None
        )[1]  # Get metrics from tuple
        
        # Store experience
        self.store_experience({
            'state': state,
            'action': action,
            'emotion': emotion_values,
            'attention': attention_level,
            'narrative': narrative,
            'stress_level': stress_level
        })
        
        # Update metrics
        self.update_metrics(
            emotion_values=emotion_values,
            attention_metrics=attention_metrics,
            stress_level=stress_level
        )
        
        return self.get_evaluation_results()
        
    def update_metrics(
        self,
        emotion_values: Dict[str, float],
        attention_metrics: Dict[str, float],
        stress_level: float
    ):
        """Update consciousness development metrics"""
        
        # Update emotional awareness
        self.metrics.emotional_awareness = self._calculate_emotional_awareness(
            emotion_values
        )
        
        # Update attention stability
        self.metrics.attention_stability = self._calculate_attention_stability(
            attention_metrics
        )
        
        # Update memory coherence
        self.metrics.memory_coherence = self._calculate_memory_coherence()
        
        # Update survival adaptation
        self.metrics.survival_adaptation = self._calculate_survival_adaptation(
            stress_level
        )
        
        # Update interaction quality
        self.metrics.interaction_quality = self._calculate_interaction_quality()
        
        # Update narrative consistency
        self.metrics.narrative_consistency = self._calculate_narrative_consistency()
        
    def _calculate_emotional_awareness(self, emotion_values: Dict[str, float]) -> float:
        """Calculate emotional awareness score"""
        if not self.experience_history:
            return 0.0
            
        recent_emotions = [exp['emotion'] for exp in self.experience_history[-100:]]
        
        # Calculate emotional stability
        stability = np.mean([
            1 - abs(e1['valence'] - e2['valence'])
            for e1, e2 in zip(recent_emotions[:-1], recent_emotions[1:])
        ])
        
        # Calculate emotional range
        emotional_range = np.std([e['valence'] for e in recent_emotions])
        
        return (stability + emotional_range) / 2
        
    def _calculate_attention_stability(self, attention_metrics: Dict[str, float]) -> float:
        """Calculate attention stability score"""
        return attention_metrics.get('attention_level', 0.0)
        
    def _calculate_memory_coherence(self) -> float:
        """Calculate memory coherence score"""
        if len(self.experience_history) < 2:
            return 0.0
            
        # Calculate temporal coherence
        coherence_scores = []
        for i in range(len(self.experience_history) - 1):
            curr = self.experience_history[i]
            next_exp = self.experience_history[i + 1]
            
            # Compare emotional states
            emotional_coherence = 1 - abs(
                curr['emotion']['valence'] - next_exp['emotion']['valence']
            )
            
            # Compare narratives
            narrative_coherence = self._calculate_narrative_similarity(
                curr['narrative'],
                next_exp['narrative']
            )
            
            coherence_scores.append((emotional_coherence + narrative_coherence) / 2)
            
        return np.mean(coherence_scores)
        
    def _calculate_survival_adaptation(self, stress_level: float) -> float:
        """Calculate survival adaptation score"""
        if not self.experience_history:
            return 0.0
            
        recent_stress = [exp['stress_level'] for exp in self.experience_history[-100:]]
        
        # Calculate stress reduction over time
        stress_change = np.mean(np.diff(recent_stress))
        
        # Higher score for reducing stress levels
        return 1.0 / (1.0 + np.exp(stress_change))
        
    def _calculate_interaction_quality(self) -> float:
        """Calculate interaction quality score"""
        if not self.experience_history:
            return 0.0
            
        recent_interactions = self.experience_history[-100:]
        
        # Calculate average emotional engagement
        emotional_engagement = np.mean([
            exp['emotion']['arousal'] for exp in recent_interactions
        ])
        
        # Calculate attention during interactions
        attention_quality = np.mean([
            exp['attention'] for exp in recent_interactions
        ])
        
        return (emotional_engagement + attention_quality) / 2
        
    def store_experience(self, experience: Dict):
        """Store experience in memory"""
        self.memory.store_experience(experience)
        self.experience_history.append(experience)
        
    def get_evaluation_results(self) -> Dict:
        """Get current evaluation results"""
        return {
            'emotional_awareness': self.metrics.emotional_awareness,
            'attention_stability': self.metrics.attention_stability,
            'memory_coherence': self.metrics.memory_coherence,
            'survival_adaptation': self.metrics.survival_adaptation,
            'interaction_quality': self.metrics.interaction_quality,
            'narrative_consistency': self.metrics.narrative_consistency,
            'consciousness_score': self._calculate_consciousness_score()
        }
        
    def _calculate_consciousness_score(self) -> float:
        """Calculate overall consciousness development score"""
        weights = {
            'emotional_awareness': 0.25,
            'attention_stability': 0.20,
            'memory_coherence': 0.20,
            'survival_adaptation': 0.15,
            'interaction_quality': 0.10,
            'narrative_consistency': 0.10
        }
        
        return sum(
            getattr(self.metrics, metric) * weight
            for metric, weight in weights.items()
        )