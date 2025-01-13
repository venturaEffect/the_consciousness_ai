# tests/test_emotional_reinforcement.py

"""
Test suite for emotional reinforcement learning integration in ACM.

This module validates:
1. Emotional reward shaping based on attention and arousal
2. Integration between emotional memory and RL
3. Learning progress based on emotional states
4. Temporal sequence validation

Dependencies:
- models/emotion/reward_shaping.py for reward calculations 
- models/emotion/tgnn/emotional_graph.py for emotion processing
- models/memory/emotional_memory_core.py for storage
- configs/reinforcement.yaml for parameters
"""

import unittest
import torch
import numpy as np
from typing import Dict, Optional
from models.self_model.reinforcement_core import ReinforcementCore
from models.emotion.tgnn.emotional_graph import EmotionalGraphNetwork
from models.memory.memory_core import MemoryCore
from simulations.api.simulation_manager import SimulationManager

class TestEmotionalReinforcement(unittest.TestCase):
    def setUp(self):
        """Initialize test components"""
        self.config = self._load_test_config()
        self.emotion_network = EmotionalGraphNN(self.config)
        self.memory = EmotionalMemoryCore(self.config)
        self.monitor = ConsciousnessMonitor(self.config)

    def test_reward_shaping(self):
        """Test emotional reward shaping"""
        # Test state with high emotional engagement
        test_state = {
            'attention': 0.8,
            'arousal': 0.7,
            'valence': 0.6
        }
        
        shaped_reward = self.emotion_network.shape_reward(
            base_reward=1.0,
            emotional_state=test_state
        )
        
        # Verify reward scaling based on emotional state
        self.assertGreater(
            shaped_reward,
            1.0,
            "Positive emotional state should increase reward"
        )

    def test_emotional_reward_computation(self):
        """Test if emotional rewards are computed correctly"""
        # Create mock emotional state
        emotion_values = {
            'valence': 0.8,  # Positive emotion
            'arousal': 0.6,
            'dominance': 0.7
        }
        
        state = torch.randn(32)  # Mock state vector
        action_info = {'action_type': 'greet', 'intensity': 0.5}
        
        # Compute reward
        reward = self.rl_core.compute_reward(state, emotion_values, action_info)
        
        # Verify reward properties
        self.assertIsInstance(reward, float)
        self.assertTrue(0 <= reward <= self.config['reinforcement']['emotional_scale'] * 2)
        self.assertTrue(reward > 0)  # Should be positive for positive valence

    def test_meta_learning_adaptation(self):
        """Test meta-learning adaptation to new scenarios"""
        # Create mock scenario data
        scenario_data = {
            'task_id': 'emotional_interaction_1',
            'states': torch.randn(10, 32),
            'actions': torch.randn(10, 8),
            'rewards': torch.randn(10),
            'emotions': torch.randn(10, 3)
        }
        
        # Perform adaptation
        adaptation_result = self.rl_core.adapt_to_scenario(scenario_data)
        
        # Verify adaptation results
        self.assertIn('task_loss', adaptation_result)
        self.assertIn('adapted_params', adaptation_result)
        self.assertTrue(adaptation_result['task_loss'] >= 0)

    def test_memory_integration(self):
        """Test emotional experience storage and retrieval"""
        # Create mock experience
        experience = {
            'state': torch.randn(32),
            'action': torch.randn(8),
            'reward': 0.5,
            'emotion': {'valence': 0.8},
            'narrative': "Agent responded positively to greeting"
        }
        
        # Store experience
        self.memory.store_experience(experience)
        
        # Retrieve and verify
        retrieved = self.memory.get_last_experience()
        self.assertTrue(torch.allclose(experience['state'], retrieved['state']))
        self.assertEqual(experience['reward'], retrieved['reward'])
        self.assertEqual(experience['emotion']['valence'], 
                       retrieved['emotion']['valence'])

    def test_full_interaction_loop(self):
        """Test complete interaction loop with emotional reinforcement"""
        # Create mock environment and agent
        env = MockEnvironment()
        agent = MockAgent()
        
        # Run interaction episode
        result = self.sim_manager.run_interaction_episode(agent, env)
        
        # Verify interaction results
        self.assertIn('total_reward', result)
        self.assertIn('steps', result)
        self.assertIn('episode_data', result)
        self.assertIn('mean_emotion', result)
        self.assertTrue(len(result['episode_data']) > 0)

class MockEnvironment:
    """Mock environment for testing"""
    def reset(self):
        return torch.randn(32)
        
    def step(self, action):
        next_state = torch.randn(32)
        reward = torch.rand(1).item()
        done = torch.rand(1).item() > 0.95
        info = {
            'emotion_values': {
                'valence': torch.rand(1).item(),
                'arousal': torch.rand(1).item()
            }
        }
        return next_state, reward, done, info

class MockAgent:
    """Mock agent for testing"""
    def get_action(self, state):
        return torch.randn(8)

if __name__ == '__main__':
    unittest.main()