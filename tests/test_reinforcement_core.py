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
            'positive_emotion_bonus': 0.5,
            'dreamerV3': {
                'hidden_size': 256,
                'learning_rate': 0.0001
            },
            'memory_capacity': 1000,
            'meta_config': {
                'enabled': True,
                'adaptation_steps': 5,
                'inner_learning_rate': 0.01
            }
        }
        self.rl_core = ReinforcementCore(self.config)

    def test_compute_reward(self):
        """Test emotional reward computation."""
        state = torch.randn(1, 32)  # Mock state tensor
        emotion_values = {
            'valence': 0.8,
            'arousal': 0.6,
            'dominance': 0.7
        }
        action_info = {'type': 'mock_action'}

        reward = self.rl_core.compute_reward(state, emotion_values, action_info)

        self.assertIsInstance(reward, float)
        # Basic sanity check: reward should be >= 0 if valence is positive.
        self.assertGreaterEqual(reward, 0.0)

    def test_adaptation(self):
        """Test meta-learning adaptation."""
        # Scenario data should reflect something the meta-learner can use.
        scenario_data = {
            'task_id': 'emotional_interaction_1',
            'samples': 20,
            'description': 'Test scenario for meta-learning adaptation.'
        }

        adaptation_result = self.rl_core.adapt_to_scenario(scenario_data)

        # Depending on how your MetaLearner is implemented, adapt_to_scenario
        # might return these fields or different ones.
        self.assertIn('adapted_params', adaptation_result,
                      msg="Meta-learner should return 'adapted_params'.")
        # If your meta-learner returns additional keys (e.g., task_loss), check them as well:
        # self.assertIn('task_loss', adaptation_result)

    def test_memory_integration(self):
        """Test memory storage and retrieval."""
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
