# tests/test_emotional_reinforcement_integration.py

import unittest
import torch
import numpy as np
from models.self_model.reinforcement_core import ReinforcementCore
from models.emotion.tgnn.emotional_graph import EmotionalGraphNetwork
from models.memory.memory_core import MemoryCore
from models.predictive.dreamerv3_wrapper import DreamerV3
from models.narrative.narrative_engine import NarrativeEngine

class TestEmotionalReinforcementIntegration(unittest.TestCase):
    """Integration tests for emotional reinforcement learning system"""
    
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
                },
                'memory_config': {
                    'capacity': 1000,
                    'emotion_embedding_size': 128,
                    'context_length': 32
                }
            }
        }
        
        # Initialize components
        self.rl_core = ReinforcementCore(self.config)
        self.emotion_network = EmotionalGraphNetwork()
        self.memory = MemoryCore(self.config['reinforcement']['memory_config'])
        self.dreamer = DreamerV3(self.config['reinforcement']['dreamer_config'])
        self.narrative = NarrativeEngine()
        
    def test_end_to_end_learning(self):
        """Test complete learning cycle with emotional integration"""
        # Create mock environment and initial state
        state = torch.randn(32)
        emotion_values = {
            'valence': 0.8,
            'arousal': 0.6,
            'dominance': 0.7
        }
        
        # Run learning episode
        for step in range(10):
            # Get action from policy
            action = self.rl_core.get_action(state)
            
            # Simulate environment step
            next_state = torch.randn(32)
            reward = torch.rand(1).item()
            
            # Process emotional response
            emotion_output = self.emotion_network.process_interaction(
                state=state,
                action=action,
                next_state=next_state
            )
            
            # Update emotional reward
            emotional_reward = self.rl_core.compute_reward(
                state=state,
                emotion_values=emotion_output,
                action_info={'step': step}
            )
            
            # Update agent
            update_info = self.rl_core.update(
                state=state,
                action=action,
                reward=emotional_reward,
                next_state=next_state,
                done=(step == 9),
                emotion_context=emotion_output
            )
            
            # Verify update results
            self.assertIn('world_model_loss', update_info)
            self.assertIn('actor_loss', update_info)
            self.assertIn('critic_loss', update_info)
            
            state = next_state
            
    def test_emotional_memory_integration(self):
        """Test emotional experience storage and retrieval"""
        # Create test experiences
        experiences = []
        for i in range(5):
            experience = {
                'state': torch.randn(32),
                'action': torch.randn(8),
                'emotion': {
                    'valence': 0.7 + 0.1 * i,
                    'arousal': 0.5 + 0.1 * i
                },
                'reward': 0.5 + 0.1 * i,
                'narrative': f"Experience {i} with emotional response"
            }
            experiences.append(experience)
            self.memory.store_experience(experience)
        
        # Test retrieval by emotional similarity
        query_emotion = {'valence': 0.8, 'arousal': 0.6}
        similar_experiences = self.memory.get_similar_emotional_experiences(
            emotion_query=query_emotion,
            k=3
        )
        
        self.assertEqual(len(similar_experiences), 3)
        self.assertTrue(all('emotion' in exp for exp in similar_experiences))
        
    def test_meta_learning_adaptation(self):
        """Test meta-learning adaptation to new emotional scenarios"""
        # Create base scenario
        base_scenario = {
            'states': torch.randn(10, 32),
            'actions': torch.randn(10, 8),
            'emotions': torch.randn(10, 3),
            'rewards': torch.randn(10)
        }
        
        # Perform base training
        pre_adaptation_performance = self.evaluate_scenario(base_scenario)
        
        # Adapt to scenario
        adaptation_result = self.rl_core.adapt_to_scenario(base_scenario)
        
        # Evaluate post-adaptation
        post_adaptation_performance = self.evaluate_scenario(base_scenario)
        
        # Verify improvement
        self.assertGreater(
            post_adaptation_performance['emotional_accuracy'],
            pre_adaptation_performance['emotional_accuracy']
        )
        
    def evaluate_scenario(self, scenario):
        """Helper method to evaluate performance on a scenario"""
        total_reward = 0
        emotional_correct = 0
        
        for i in range(len(scenario['states'])):
            state = scenario['states'][i]
            action = self.rl_core.get_action(state)
            
            predicted_emotion = self.emotion_network.predict_emotion(
                state=state,
                action=action
            )
            
            actual_emotion = scenario['emotions'][i]
            emotional_correct += self.calculate_emotion_accuracy(
                predicted_emotion,
                actual_emotion
            )
            
            total_reward += scenario['rewards'][i].item()
            
        return {
            'total_reward': total_reward,
            'emotional_accuracy': emotional_correct / len(scenario['states'])
        }
        
    def calculate_emotion_accuracy(self, predicted, target):
        """Helper method to calculate emotional prediction accuracy"""
        if isinstance(predicted, dict) and isinstance(target, dict):
            accuracy = 0
            for key in ['valence', 'arousal']:
                if key in predicted and key in target:
                    accuracy += 1 - abs(predicted[key] - target[key].item())
            return accuracy / 2
            
        return np.mean([1 - abs(p - t.item()) for p, t in zip(predicted, target)])

if __name__ == '__main__':
    unittest.main()