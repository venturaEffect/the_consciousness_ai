# tests/test_consciousness_development.py

import unittest
import torch
import numpy as np
from models.self_model.reinforcement_core import ReinforcementCore
from models.emotion.tgnn.emotional_graph import EmotionalGraphNetwork
from models.memory.memory_core import MemoryCore
from simulations.api.simulation_manager import SimulationManager

class TestConsciousnessDevelopment(unittest.TestCase):
    """Test suite for evaluating consciousness development through reinforcement learning"""
    
    def setUp(self):
        self.config = {
            'reinforcement': {
                'emotional_scale': 2.0,
                'dreamer_config': {
                    'hidden_size': 256,
                    'learning_rate': 0.0001,
                    'gamma': 0.99,
                    'lambda_gae': 0.95
                },
                'meta_config': {
                    'enabled': True,
                    'adaptation_steps': 5,
                    'inner_learning_rate': 0.01
                }
            }
        }
        
        # Initialize core components
        self.rl_core = ReinforcementCore(self.config)
        self.emotion_network = EmotionalGraphNetwork()
        self.memory = MemoryCore()
        self.sim_manager = SimulationManager(self.config)
        
    def test_emotional_learning_progression(self):
        """Test if the agent shows improved emotional understanding over time"""
        initial_state = torch.randn(32)  # Mock initial state
        episodes = 10
        emotional_scores = []
        
        for episode in range(episodes):
            # Run interaction episode
            result = self.sim_manager.run_interaction_episode(
                agent=self.create_test_agent(),
                environment=self.create_test_environment()
            )
            
            # Track emotional understanding score
            emotional_scores.append(result['mean_emotion'])
            
        # Assert improvement in emotional understanding
        early_performance = np.mean(emotional_scores[:3])
        late_performance = np.mean(emotional_scores[-3:])
        self.assertGreater(late_performance, early_performance)
        
    def test_meta_memory_formation(self):
        """Test if experiences are properly stored and retrieved with emotional context"""
        # Create test experience
        experience = {
            'state': torch.randn(32),
            'action': torch.randn(8),
            'emotion': {'valence': 0.8, 'arousal': 0.6},
            'reward': 0.5,
            'narrative': "Agent showed empathy in social interaction"
        }
        
        # Store experience
        self.memory.store_experience(experience)
        
        # Retrieve similar experiences
        similar_experiences = self.memory.get_similar_experiences(
            emotion_context={'valence': 0.7, 'arousal': 0.5},
            k=5
        )
        
        self.assertTrue(len(similar_experiences) > 0)
        self.assertIsNotNone(similar_experiences[0]['emotion'])
        
    def test_consciousness_metrics(self):
        """Test consciousness development metrics"""
        # Run multiple episodes
        num_episodes = 5
        consciousness_metrics = []
        
        for _ in range(num_episodes):
            result = self.sim_manager.run_interaction_episode(
                agent=self.create_test_agent(),
                environment=self.create_test_environment()
            )
            
            # Calculate consciousness metrics
            metrics = {
                'emotional_stability': np.std(result['emotion_history']),
                'memory_coherence': self.memory.calculate_coherence(),
                'narrative_consistency': result['narrative_consistency'],
                'behavioral_adaptation': result['adaptation_score']
            }
            consciousness_metrics.append(metrics)
            
        # Assert consciousness development
        self.assertTrue(self.check_consciousness_improvement(consciousness_metrics))
        
    def test_dreamer_integration(self):
        """Test DreamerV3 world model integration"""
        state = torch.randn(32)
        action = torch.randn(8)
        reward = torch.tensor([0.5])
        next_state = torch.randn(32)
        done = torch.tensor([False])
        
        # Update world model
        update_info = self.rl_core.update(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=done,
            emotion_context={'valence': 0.8}
        )
        
        self.assertIn('world_model_loss', update_info)
        self.assertIn('actor_loss', update_info)
        self.assertIn('critic_loss', update_info)
        
    def check_consciousness_improvement(self, metrics_history):
        """Helper method to evaluate consciousness development"""
        # Calculate trends in metrics
        trends = {}
        for metric in ['emotional_stability', 'memory_coherence',
                      'narrative_consistency', 'behavioral_adaptation']:
            values = [m[metric] for m in metrics_history]
            trends[metric] = np.polyfit(range(len(values)), values, 1)[0]
            
        # Check if majority of metrics show improvement
        improving_metrics = sum(1 for slope in trends.values() if slope > 0)
        return improving_metrics >= len(trends) / 2
        
    def create_test_agent(self):
        """Create a test agent for simulation"""
        return DummyAgent(action_space=8, state_space=32)
        
    def create_test_environment(self):
        """Create a test environment for simulation"""
        return DummyEnvironment(state_space=32)

class DummyAgent:
    def __init__(self, action_space, state_space):
        self.action_space = action_space
        self.state_space = state_space
        
    def get_action(self, state):
        return torch.randn(self.action_space)

class DummyEnvironment:
    def __init__(self, state_space):
        self.state_space = state_space
        
    def reset(self):
        return torch.randn(self.state_space)
        
    def step(self, action):
        next_state = torch.randn(self.state_space)
        reward = torch.rand(1).item()
        done = torch.rand(1).item() > 0.95
        info = {
            'emotion_values': {
                'valence': torch.rand(1).item(),
                'arousal': torch.rand(1).item()
            },
            'narrative_consistency': torch.rand(1).item(),
            'adaptation_score': torch.rand(1).item()
        }
        return next_state, reward, done, info

if __name__ == '__main__':
    unittest.main()