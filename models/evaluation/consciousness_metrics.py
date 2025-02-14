# models/evaluation/consciousness_metrics.py

import numpy as np
import torch
from typing import Dict, List, Optional
from models.self_model.reinforcement_core import ReinforcementCore
from models.emotion.tgnn.emotional_graph import EmotionalGraphNetwork
from models.memory.memory_core import MemoryCore

class ConsciousnessMetrics:
    """Evaluates consciousness development through various metrics"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.rl_core = ReinforcementCore(config)
        self.emotion_network = EmotionalGraphNetwork()
        self.memory = MemoryCore()
        
        # Metric thresholds
        self.coherence_threshold = config.get('coherence_threshold', 0.7)
        self.emotional_stability_threshold = config.get('emotional_stability', 0.6)
        
    def evaluate_emotional_awareness(self, interactions: List[Dict]) -> Dict[str, float]:
        """
        Evaluate emotional awareness level based on interaction history
        """
        emotional_scores = []
        prediction_accuracy = []
        
        for interaction in interactions:
            # Get emotional predictions
            predicted_emotion = self.emotion_network.predict_emotion(
                state=interaction['state'],
                action=interaction['action']
            )
            
            # Compare with actual emotions
            accuracy = self.calculate_emotion_accuracy(
                predicted_emotion,
                interaction['emotion_values']
            )
            
            emotional_scores.append(interaction['emotional_reward'])
            prediction_accuracy.append(accuracy)
            
        return {
            'mean_emotional_awareness': np.mean(emotional_scores),
            'emotion_prediction_accuracy': np.mean(prediction_accuracy),
            'emotional_stability': np.std(emotional_scores)
        }
        
    def evaluate_memory_coherence(self) -> Dict[str, float]:
        """
        Evaluate memory system coherence and retrieval capabilities
        """
        # Get recent experiences
        recent_experiences = self.memory.get_recent_experiences(limit=100)
        
        # Calculate temporal coherence
        temporal_coherence = self.calculate_temporal_coherence(recent_experiences)
        
        # Calculate emotional consistency
        emotional_consistency = self.calculate_emotional_consistency(recent_experiences)
        
        # Calculate narrative alignment
        narrative_alignment = self.calculate_narrative_alignment(recent_experiences)
        
        return {
            'temporal_coherence': temporal_coherence,
            'emotional_consistency': emotional_consistency,
            'narrative_alignment': narrative_alignment,
            'memory_utilization': self.memory.get_utilization_metrics()
        }
        
    def evaluate_learning_progress(self, training_history: List[Dict]) -> Dict[str, float]:
        """
        Evaluate reinforcement learning progress
        """
        reward_history = [episode['total_reward'] for episode in training_history]
        emotional_history = [episode['mean_emotion'] for episode in training_history]
        
        # Calculate learning curves
        reward_slope = np.polyfit(range(len(reward_history)), reward_history, 1)[0]
        emotional_slope = np.polyfit(range(len(emotional_history)), emotional_history, 1)[0]
        
        return {
            'reward_improvement': reward_slope,
            'emotional_learning': emotional_slope,
            'final_performance': np.mean(reward_history[-10:]),
            'stability': np.std(reward_history[-20:])
        }
        
    def calculate_temporal_coherence(self, experiences: List[Dict]) -> float:
        """
        Calculate temporal coherence of memories
        """
        coherence_scores = []
        for i in range(len(experiences) - 1):
            current = experiences[i]
            next_exp = experiences[i + 1]
            
            # Check state transitions
            state_coherence = torch.nn.functional.cosine_similarity(
                current['state'].unsqueeze(0),
                next_exp['state'].unsqueeze(0)
            ).item()
            
            # Check emotional continuity
            emotion_coherence = self.calculate_emotion_consistency(
                current['emotion'],
                next_exp['emotion']
            )
            
            coherence_scores.append((state_coherence + emotion_coherence) / 2)
            
        return np.mean(coherence_scores)
        
    def calculate_emotional_consistency(self, experiences: List[Dict]) -> float:
        """
        Calculate emotional consistency across experiences
        """
        emotion_values = [exp['emotion_values'] for exp in experiences]
        consistency_scores = []
        
        for i in range(len(emotion_values) - 1):
            consistency = self.calculate_emotion_similarity(
                emotion_values[i],
                emotion_values[i + 1]
            )
            consistency_scores.append(consistency)
            
        return np.mean(consistency_scores)
        
    def calculate_narrative_alignment(self, experiences: List[Dict]) -> float:
        """
        Calculate alignment between experiences and their narrative descriptions
        """
        alignment_scores = []
        
        for exp in experiences:
            if 'narrative' in exp and 'emotion_values' in exp:
                # Compare narrative sentiment with emotional values
                narrative_sentiment = self.emotion_network.extract_sentiment(exp['narrative'])
                alignment = self.calculate_emotion_similarity(
                    narrative_sentiment,
                    exp['emotion_values']
                )
                alignment_scores.append(alignment)
                
        return np.mean(alignment_scores)
        
    @staticmethod
    def calculate_emotion_similarity(emotion1: Dict[str, float], 
                                  emotion2: Dict[str, float]) -> float:
        """
        Calculate similarity between two emotion states
        """
        if not emotion1 or not emotion2:
            return 0.0
            
        common_keys = set(emotion1.keys()) & set(emotion2.keys())
        if not common_keys:
            return 0.0
            
        similarities = []
        for key in common_keys:
            similarities.append(1 - abs(emotion1[key] - emotion2[key]))
            
        return np.mean(similarities)
        
    def get_consciousness_score(self, metrics: Dict[str, float]) -> float:
        """
        Calculate overall consciousness score from individual metrics
        """
        weights = {
            'emotional_awareness': 0.3,
            'memory_coherence': 0.3,
            'learning_progress': 0.2,
            'narrative_consistency': 0.2
        }
        
        score = 0.0
        for key, weight in weights.items():
            if key in metrics:
                score += metrics[key] * weight
                
        return score

class IntegratedInformationCalculator:
    def __init__(self, acm_system):
        self.acm = acm_system

    def compute_phi(self) -> float:
        # Placeholder for PyPhi or custom integrated info computation
        # e.g. extracting a module connectivity graph from self.acm
        return 3.14

class GlobalWorkspaceTracker:
    def __init__(self, acm_system):
        self.acm = acm_system
        self.ignition_threshold = 0.8

    def check_global_workspace_events(self) -> float:
        # Count how often modules share data above ignition_threshold
        return float(self.acm.global_workspace.get_ignition_count())

class PerturbationTester:
    def __init__(self, acm_system):
        self.acm = acm_system

    def simulate_and_measure(self) -> float:
        # Example: memory wipe or random noise injection
        self.acm.memory_core.force_memory_wipe()
        score = self.acm.evaluate_recovery()  # Evaluate post-wipe coherence
        return score

class SelfAwarenessMonitor:
    def __init__(self, acm_system):
        self.acm = acm_system

    def evaluate_self_awareness(self) -> float:
        # E.g. count how often the system corrects its own mistakes
        return self.acm.self_model.get_self_reflection_score()