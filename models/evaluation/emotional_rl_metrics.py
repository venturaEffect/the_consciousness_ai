# models/evaluation/emotional_rl_metrics.py

import torch
import numpy as np
from typing import Dict, List, Optional
from collections import deque
from dataclasses import dataclass

@dataclass
class EmotionalMetrics:
    """Stores emotional learning metrics"""
    emotional_awareness: float = 0.0
    reward_stability: float = 0.0
    learning_progress: float = 0.0
    memory_coherence: float = 0.0
    narrative_consistency: float = 0.0

class EmotionalRLTracker:
    """
    Tracks and analyzes emotional reinforcement learning metrics
    """
    def __init__(self, config: Dict):
        self.config = config
        
        # Initialize metric histories
        self.reward_history = deque(maxlen=1000)
        self.emotion_history = deque(maxlen=1000)
        self.narrative_history = deque(maxlen=100)
        
        # Thresholds from config
        self.reward_stability_threshold = config.get('reward_stability_threshold', 0.1)
        self.emotional_awareness_threshold = config.get('emotional_awareness_threshold', 0.7)
        
    def update(self, metrics: Dict) -> EmotionalMetrics:
        """Update metrics with new data"""
        # Store new metrics
        if 'reward' in metrics:
            self.reward_history.append(metrics['reward'])
        if 'emotion_values' in metrics:
            self.emotion_history.append(metrics['emotion_values'])
        if 'narrative' in metrics:
            self.narrative_history.append(metrics['narrative'])
            
        # Calculate current metrics
        current_metrics = EmotionalMetrics(
            emotional_awareness=self._calculate_emotional_awareness(),
            reward_stability=self._calculate_reward_stability(),
            learning_progress=self._calculate_learning_progress(),
            memory_coherence=self._calculate_memory_coherence(),
            narrative_consistency=self._calculate_narrative_consistency()
        )
        
        return current_metrics
        
    def _calculate_emotional_awareness(self) -> float:
        """Calculate emotional awareness score"""
        if len(self.emotion_history) < 2:
            return 0.0
            
        # Compare consecutive emotional predictions
        awareness_scores = []
        for i in range(len(self.emotion_history) - 1):
            curr_emotion = self.emotion_history[i]
            next_emotion = self.emotion_history[i + 1]
            
            # Calculate emotional continuity
            continuity = 1.0 - np.mean([
                abs(curr_emotion[k] - next_emotion[k])
                for k in curr_emotion.keys()
            ])
            awareness_scores.append(continuity)
            
        return np.mean(awareness_scores)
        
    def _calculate_reward_stability(self) -> float:
        """Calculate reward stability"""
        if len(self.reward_history) < 10:
            return 0.0
            
        # Calculate reward variance over recent history
        recent_rewards = list(self.reward_history)[-10:]
        return 1.0 / (1.0 + np.std(recent_rewards))
        
    def _calculate_learning_progress(self) -> float:
        """Calculate learning progress trend"""
        if len(self.reward_history) < 100:
            return 0.0
            
        # Calculate slope of reward trend
        x = np.arange(len(self.reward_history))
        y = np.array(self.reward_history)
        slope = np.polyfit(x, y, 1)[0]
        
        # Normalize slope to [0, 1]
        return 1.0 / (1.0 + np.exp(-10 * slope))
        
    def _calculate_memory_coherence(self) -> float:
        """Calculate memory coherence score"""
        if len(self.emotion_history) < 10:
            return 0.0
            
        # Calculate temporal coherence of emotional memories
        coherence_scores = []
        for i in range(len(self.emotion_history) - 1):
            curr_emotion = self.emotion_history[i]
            next_emotion = self.emotion_history[i + 1]
            
            # Check emotional continuity
            coherence = 1.0 - np.mean([
                abs(curr_emotion[k] - next_emotion[k])
                for k in curr_emotion.keys()
            ])
            coherence_scores.append(coherence)
            
        return np.mean(coherence_scores)
        
    def _calculate_narrative_consistency(self) -> float:
        """Calculate narrative consistency score"""
        if len(self.narrative_history) < 2:
            return 0.0
            
        # Compare consecutive narratives for consistency
        consistency_scores = []
        for i in range(len(self.narrative_history) - 1):
            curr_narrative = self.narrative_history[i]
            next_narrative = self.narrative_history[i + 1]
            
            # Simple string similarity for now
            # Could be enhanced with semantic similarity
            similarity = len(set(curr_narrative.split()) & 
                          set(next_narrative.split())) / \
                      len(set(curr_narrative.split()) | 
                          set(next_narrative.split()))
            consistency_scores.append(similarity)
            
        return np.mean(consistency_scores)
        
    def get_summary(self) -> Dict:
        """Get summary of current learning state"""
        current_metrics = self.update({})
        
        return {
            'emotional_awareness': current_metrics.emotional_awareness,
            'reward_stability': current_metrics.reward_stability,
            'learning_progress': current_metrics.learning_progress,
            'memory_coherence': current_metrics.memory_coherence,
            'narrative_consistency': current_metrics.narrative_consistency,
            'meets_thresholds': self._check_thresholds(current_metrics)
        }
        
    def _check_thresholds(self, metrics: EmotionalMetrics) -> bool:
        """Check if current metrics meet minimum thresholds"""
        return (
            metrics.emotional_awareness >= self.emotional_awareness_threshold and
            metrics.reward_stability >= self.reward_stability_threshold and
            metrics.learning_progress > 0
        )