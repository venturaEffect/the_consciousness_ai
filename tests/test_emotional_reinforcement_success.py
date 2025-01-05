# tests/test_emotional_reinforcement_success.py

import unittest
import torch
import numpy as np
from models.self_model.reinforcement_core import ReinforcementCore
from models.emotion.tgnn.emotional_graph import EmotionalGraphNetwork
from models.memory.memory_core import MemoryCore
from simulations.api.simulation_manager import SimulationManager
from models.narrative.narrative_engine import NarrativeEngine

class TestEmotionalReinforcementSuccess(unittest.TestCase):
    """Test suite for evaluating emotional reinforcement learning success metrics"""
    
    def setUp(self):
        self.config = {
            'reinforcement': {
                'emotional_scale': 2.0,
                'dreamer_config': {
                    'hidden_size': 256,
                    'learning_rate': 0.0001
                },
                'meta_config': {
                    'enabled': True,
                    'adaptation_steps': 5,
                    'inner_learning_rate': 0.01
                },
                'memory_config': {
                    'capacity': 1000,
                    'emotion_embedding_size': 128
                }
            }
        }
        
        # Initialize core components
        self.rl_core = ReinforcementCore(self.config)
        self.emotion_network = EmotionalGraphNetwork()
        self.memory = MemoryCore()
        self.narrative = NarrativeEngine()
        
    def test_emotional_memory_formation(self):
        """Test if emotional experiences are properly stored and retrieved"""
        # Create test emotional experience
        experience = {
            'state': torch.randn(32),
            'action': torch.randn(8),
            'emotion': {
                'valence': 0.8,  # Positive emotion
                'arousal': 0.6,
                'dominance': 0.7
            },
            'reward': 0.5,
            'narrative': "Agent showed empathy in interaction"
        }
        
        # Store experience
        self.memory.store_experience(experience)
        
        # Retrieve similar emotional experiences
        similar_experiences = self.memory.get_similar_emotional_experiences(
            emotion_query={'valence': 0.7, 'arousal': 0.5},
            k=5
        )
        
        self.assertTrue(len(similar_experiences) > 0)
        self.assertIsNotNone(similar_experiences[0]['emotion'])
        
    def test_reward_shaping(self):
        """Test emotional reward shaping mechanism"""
        state = torch.randn(32)
        emotion_values = {
            'valence': 0.9,  # Very positive emotion
            'arousal': 0.7,
            'dominance': 0.8
        }
        action_info = {'action_type': 'help_human', 'intensity': 0.8}
        
        reward = self.rl_core.compute_reward(state, emotion_values, action_info)
        
        # Verify reward properties
        self.assertGreater(reward, 0)  # Positive reward for positive emotion
        self.assertLessEqual(reward, self.config['reinforcement']['emotional_scale'] * 2)
        
    def test_learning_progression(self):
        """Test if agent shows improved emotional understanding over time"""
        # Run multiple learning episodes
        num_episodes = 5
        emotional_scores = []
        
        for episode in range(num_episodes):
            result = self.run_test_episode()
            emotional_scores.append(result['emotional_understanding'])
            
        # Verify learning progression
        early_performance = np.mean(emotional_scores[:2])
        late_performance = np.mean(emotional_scores[-2:])
        self.assertGreater(late_performance, early_performance)
        
    def test_meta_adaptation(self):
        """Test meta-learning adaptation to new emotional scenarios"""
        # Create test scenario
        scenario = {
            'task_id': 'new_emotional_interaction',
            'context': torch.randn(64),
            'target_emotion': {'valence': 0.8, 'arousal': 0.6}
        }
        
        # Perform adaptation
        pre_adaptation_performance = self.evaluate_emotional_understanding(scenario)
        self.rl_core.adapt_to_scenario(scenario)
        post_adaptation_performance = self.evaluate_emotional_understanding(scenario)
        
        # Verify adaptation improvement
        self.assertGreater(post_adaptation_performance, pre_adaptation_performance)
        
    def test_narrative_integration(self):
        """Test if emotional experiences generate coherent narratives"""
        experience = {
            'state': torch.randn(32),
            'emotion': {'valence': 0.8, 'arousal': 0.6},
            'action': {'type': 'comfort', 'target': 'human'},
            'outcome': 'positive_interaction'
        }
        
        narrative = self.narrative.generate_experience_narrative(experience)
        
        self.assertIsNotNone(narrative)
        self.assertGreater(len(narrative), 0)
        
    def run_test_episode(self):
        """Helper method to run a test episode"""
        state = torch.randn(32)
        total_reward = 0
        emotional_understanding = 0
        
        for step in range(10):
            action = self.rl_core.get_action(state)
            emotion_values = {
                'valence': np.random.random(),
                'arousal': np.random.random()
            }
            
            reward = self.rl_core.compute_reward(state, emotion_values, action)
            next_state = torch.randn(32)
            
            # Update emotional understanding score
            predicted_emotion = self.emotion_network.predict_emotion(state, action)
            emotional_understanding += self.calculate_emotion_accuracy(
                predicted_emotion,
                emotion_values
            )
            
            total_reward += reward
            state = next_state
            
        return {
            'total_reward': total_reward,
            'emotional_understanding': emotional_understanding / 10
        }
        
    def evaluate_emotional_understanding(self, scenario):
        """Helper method to evaluate emotional understanding in a scenario"""
        predictions = []
        targets = []
        
        for _ in range(5):
            state = torch.randn(32)
            action = self.rl_core.get_action(state)
            predicted_emotion = self.emotion_network.predict_emotion(state, action)
            predictions.append(predicted_emotion)
            targets.append(scenario['target_emotion'])
            
        return self.calculate_emotional_accuracy(predictions, targets)
        
    def calculate_emotion_accuracy(self, predicted, target):
        """Helper method to calculate emotional prediction accuracy"""
        if isinstance(predicted, dict) and isinstance(target, dict):
            accuracy = 0
            for key in ['valence', 'arousal']:
                if key in predicted and key in target:
                    accuracy += 1 - abs(predicted[key] - target[key])
            return accuracy / 2
        return np.mean([1 - abs(p - t) for p, t in zip(predicted, target)])

if __name__ == '__main__':
    unittest.main()