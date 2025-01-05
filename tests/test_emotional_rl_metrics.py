# tests/test_emotional_rl_metrics.py

import unittest
import torch
import numpy as np
from models.evaluation.emotional_rl_metrics import EmotionalRLTracker, EmotionalMetrics

class TestEmotionalRLMetrics(unittest.TestCase):
    def setUp(self):
        self.config = {
            'reward_stability_threshold': 0.1,
            'emotional_awareness_threshold': 0.7
        }
        self.tracker = EmotionalRLTracker(self.config)
        
    def test_metric_initialization(self):
        """Test proper initialization of metrics"""
        metrics = self.tracker.get_summary()
        
        self.assertIn('emotional_awareness', metrics)
        self.assertIn('reward_stability', metrics)
        self.assertIn('learning_progress', metrics)
        self.assertIn('memory_coherence', metrics)
        self.assertIn('narrative_consistency', metrics)
        
    def test_emotional_awareness_calculation(self):
        """Test emotional awareness computation"""
        # Add sample emotional data
        for _ in range(10):
            metrics = {
                'emotion_values': {
                    'valence': np.random.random(),
                    'arousal': np.random.random()
                }
            }
            self.tracker.update(metrics)
            
        awareness = self.tracker._calculate_emotional_awareness()
        self.assertTrue(0 <= awareness <= 1)
        
    def test_reward_stability_calculation(self):
        """Test reward stability computation"""
        # Add sample rewards
        rewards = [0.5 + np.random.normal(0, 0.1) for _ in range(20)]
        for reward in rewards:
            self.tracker.update({'reward': reward})
            
        stability = self.tracker._calculate_reward_stability()
        self.assertTrue(0 <= stability <= 1)
        
    def test_narrative_consistency(self):
        """Test narrative consistency computation"""
        narratives = [
            "Agent showed empathy",
            "Agent demonstrated empathy in interaction",
            "Agent displayed emotional understanding"
        ]
        
        for narrative in narratives:
            self.tracker.update({'narrative': narrative})
            
        consistency = self.tracker._calculate_narrative_consistency()
        self.assertTrue(0 <= consistency <= 1)
        
    def test_threshold_checking(self):
        """Test threshold validation"""
        metrics = EmotionalMetrics(
            emotional_awareness=0.8,
            reward_stability=0.2,
            learning_progress=0.1,
            memory_coherence=0.7,
            narrative_consistency=0.6
        )
        
        meets_thresholds = self.tracker._check_thresholds(metrics)
        self.assertTrue(meets_thresholds)

if __name__ == '__main__':
    unittest.main()