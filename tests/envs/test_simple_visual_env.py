import unittest
import numpy as np
import gymnasium as gym
from simulations.environments.simple_visual_env import SimpleVisualEnv

class TestSimpleVisualEnv(unittest.TestCase):
    def setUp(self):
        self.env = SimpleVisualEnv(render_mode="rgb_array")
        
    def test_reset(self):
        obs, info = self.env.reset()
        self.assertEqual(obs.shape, (512, 512, 3))
        self.assertIn("distance_to_light", info)
        
    def test_step(self):
        self.env.reset()
        action = np.array([1.0, 0.0]) # Move Right
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        self.assertEqual(obs.shape, (512, 512, 3))
        self.assertIsInstance(reward, float)
        self.assertIsInstance(terminated, bool)
        
    def test_renderer(self):
        # Ensure PyGame headless rendering works
        self.env.reset()
        obs = self.env._get_obs()
        self.assertTrue(np.any(obs > 0), "Image should not be completely black")

if __name__ == '__main__':
    unittest.main()
