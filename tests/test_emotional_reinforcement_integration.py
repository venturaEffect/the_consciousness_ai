import unittest
import torch
import numpy as np
from typing import Dict, Any

from models.self_model.reinforcement_core import ReinforcementCore
from models.emotion.reward_shaping import EmotionalRewardShaper
from models.memory.memory_core import MemoryCore, MemoryConfig

class TestEmotionalReinforcementIntegration(unittest.TestCase):
    """Integration tests for the MODERNIZED emotional reinforcement learning system."""

    def setUp(self):
        # 1. Config
        self.config = {
            "state_dim": 32,
            "action_dim": 4,
            "gamma": 0.99,
            "learning_rate": 0.001,
            "device": "cpu",
            "emotional_dims": 3, # Valence, Arousal, Dominance
            "hidden_size": 16,
            "reward": {
                "base_scale": 1.0
            },
            # Memory Config
            "max_memories": 100,
            "cleanup_threshold": 0.4,
            "vector_dim": 32,
            "index_batch_size": 10,
            "attention_threshold": 0.5
        }

        # 2. Components
        self.emotion_shaper = EmotionalRewardShaper(self.config)
        
        mem_config = MemoryConfig(
            max_memories=100,
            vector_dim=32,
            attention_threshold=0.5
        )
        self.memory = MemoryCore(mem_config)
        
        # 3. Core
        self.rl_core = ReinforcementCore(self.config, self.emotion_shaper, self.memory)

    def test_step_and_storage(self):
        """Test that taking a step computes rewards and stores data correctly."""
        
        # Inputs
        state = torch.randn(32)
        next_state = torch.randn(32)
        action = np.array([0.1, -0.2, 0.5, 0.0])
        raw_reward = 1.0
        emotion_state = {"valence": 0.8, "arousal": 0.5, "dominance": 0.2}
        
        # Step
        result = self.rl_core.step(
            state=state,
            action=action,
            raw_reward=raw_reward,
            next_state=next_state,
            done=False,
            emotion_state=emotion_state,
            attention_level=0.8,
            narrative="Test step"
        )
        
        # Checks
        self.assertIn("shaped_reward", result)
        self.assertGreater(result["shaped_reward"], raw_reward, "High valence should boost reward")
        
        # Memory Check
        self.assertEqual(len(self.memory.recent_experiences), 1)
        self.assertEqual(len(self.rl_core.rollout_buffer), 1)

    def test_policy_update(self):
        """Test that the policy can update after collecting enough steps."""
        
        # Collect rollouts
        for i in range(15): # Min batch size is 10
            self.rl_core.step(
                state=torch.randn(32),
                action=np.random.randn(4),
                raw_reward=1.0,
                next_state=torch.randn(32),
                done=(i % 5 == 0),
                emotion_state={"valence": 0.5, "arousal": 0.5},
                attention_level=0.5
            )
            
        # Update
        metrics = self.rl_core.update_policy()
        
        # Checks
        self.assertIn("total_loss", metrics)
        self.assertIn("policy_loss", metrics)
        self.assertEqual(len(self.rl_core.rollout_buffer), 0, "Buffer should be cleared after update")

    def test_action_selection(self):
        """Test that the policy produces valid actions."""
        state = torch.randn(32)
        action, value = self.rl_core.select_action(state)
        
        self.assertEqual(action.shape, (4,))
        self.assertTrue(np.all(action >= -1.0))
        self.assertTrue(np.all(action <= 1.0))
        self.assertIsInstance(value, float)

if __name__ == '__main__':
    unittest.main()
