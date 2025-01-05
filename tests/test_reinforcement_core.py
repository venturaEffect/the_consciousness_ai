# tests/test_reinforcement_core.py

import unittest
import torch
import numpy as np
from models.self_model.reinforcement_core import ReinforcementCore
from models.memory.memory_core import MemoryCore
from models.emotion.tgnn.emotional_graph import EmotionalGraphNetwork

class TestReinforcementCore(unittest.TestCase):
    def setUp(self):
        self.config = {
            'emotional_scale': 2.0,
            'dreamer_config': {
                'hidden_size': 256,
                'learning_rate': 0.0001
            },
            'memory_config': {
                'capacity': 1000,
                'batch_size': 32
            },
            'meta_config': {
                'enabled': True,
                'adaptation_steps': 5
            }
        }
        self.rl_core = ReinforcementCore(self.config)
        
    def test_compute_reward(self):
        """Test emotional reward computation"""
        state = torch.randn(1, 32)  # Mock state tensor
        emotion_values = {
            'valence': 0.8,
            'arousal': 0.6,
            'dominance': 0.7
        }
        
        reward = self.rl_core.compute_reward(state, emotion_values)
        
        self.assertIsInstance(reward, float)
        self.assertTrue(0 <= reward <= self.config['emotional_scale'] * 2)

    def test_adaptation(self):
        """Test meta-learning adaptation"""
        scenario_data = {
            'task_id': 'emotional_interaction_1',
            'states': torch.randn(10, 32),
            'actions': torch.randn(10, 8),
            'rewards': torch.randn(10),
            'emotions': torch.randn(10, 3)
        }
        
        adaptation_result = self.rl_core.adapt_to_scenario(scenario_data)
        
        self.assertIn('task_loss', adaptation_result)
        self.assertIn('adapted_params', adaptation_result)
        
    def test_memory_integration(self):
        """Test memory storage and retrieval"""
        experience = {
            'state': torch.randn(32),
            'action': torch.randn(8),
            'reward': 0.5,
            'emotion': {'valence': 0.8}
        }
        
        self.rl_core.memory.store_experience(experience)
        retrieved = self.rl_core.memory.get_last_experience()
        
        self.assertTrue(torch.allclose(experience['state'], retrieved['state']))
        self.assertEqual(experience['reward'], retrieved['reward'])

if __name__ == '__main__':
    unittest.main()