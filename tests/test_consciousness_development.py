# tests/test_consciousness_development.py

import unittest
import torch
import numpy as np
from typing import Dict, List
from models.evaluation.consciousness_metrics import ConsciousnessMetrics
from models.predictive.attention_mechanism import ConsciousnessAttention
from models.emotion.reward_shaping import EmotionalRewardShaper
from simulations.scenarios.consciousness_scenarios import ConsciousnessScenarioManager

class TestConsciousnessDevelopment(unittest.TestCase):
    """Test suite for validating consciousness development through stress-induced learning"""
    
    def setUp(self):
        self.config = {
            'attention': {
                'base_threshold': 0.7,
                'stress_activation_level': 0.8,
                'focus_duration_min': 10
            },
            'emotional_learning': {
                'initial_scale': 2.0,
                'positive_emotion_bonus': 0.5,
                'learning_rate': 0.0001,
                'adaptation_steps': 5
            },
            'survival_metrics': {
                'stress_threshold': 0.7,
                'recovery_rate': 0.1,
                'adaptation_window': 100
            }
        }
        
        # Initialize components
        self.metrics = ConsciousnessMetrics(self.config)
        self.attention = ConsciousnessAttention(self.config)
        self.reward_shaper = EmotionalRewardShaper(self.config)
        self.scenario_manager = ConsciousnessScenarioManager(self.config)
        
    def test_attention_activation(self):
        """Test attention activation through stressful scenarios"""
        # Create stressful scenario
        scenario = self.scenario_manager.generate_scenario(
            scenario_type="survival"
        )
        
        # Process scenario with attention mechanism
        state = torch.randn(32)  # Initial state
        emotional_context = torch.randn(128)  # Emotional embedding
        
        attention_output, metrics = self.attention.forward(
            input_state=state,
            emotional_context=emotional_context,
            environment_context=None
        )
        
        # Verify attention activation
        self.assertGreater(
            metrics['attention_level'],
            self.config['attention']['base_threshold']
        )
        
    def test_emotional_memory_formation(self):
        """Test emotional memory formation during high-attention states"""
        # Create high-attention experience
        experience = {
            'state': torch.randn(32),
            'action': torch.randn(8),
            'emotion': {
                'valence': 0.3,  # Stress indication
                'arousal': 0.8,  # High arousal
                'dominance': 0.4  # Low dominance
            },
            'attention_level': 0.9,
            'narrative': "Agent successfully navigated dangerous situation"
        }
        
        # Store experience
        self.metrics.store_experience(experience)
        
        # Retrieve similar experiences
        similar_exp = self.metrics.get_similar_emotional_experiences(
            emotion_query={'valence': 0.4, 'arousal': 0.7},
            k=5
        )
        
        # Verify memory formation
        self.assertTrue(len(similar_exp) > 0)
        self.assertIsNotNone(similar_exp[0].get('emotion'))
        
    def test_survival_adaptation(self):
        """Test adaptation to survival scenarios"""
        num_episodes = 5
        stress_levels = []
        success_rates = []
        
        for _ in range(num_episodes):
            # Generate survival scenario
            scenario = self.scenario_manager.generate_scenario(
                scenario_type="survival"
            )
            
            # Run scenario
            result = self.run_survival_scenario(scenario)
            
            stress_levels.append(result['stress_level'])
            success_rates.append(result['success_rate'])
            
        # Verify adaptation
        avg_initial_stress = np.mean(stress_levels[:2])
        avg_final_stress = np.mean(stress_levels[-2:])
        
        self.assertLess(avg_final_stress, avg_initial_stress)
        self.assertGreater(success_rates[-1], success_rates[0])
        
    def run_survival_scenario(self, scenario: Dict) -> Dict:
        """Run a single survival scenario"""
        state = torch.randn(32)
        total_stress = 0
        success_count = 0
        steps = 0
        
        while steps < 100:  # Max steps per scenario
            # Get attention and stress levels
            attention_output, attention_metrics = self.attention.forward(
                input_state=state,
                emotional_context=torch.randn(128)
            )
            
            # Calculate stress
            stress_level = attention_metrics['attention_level']
            total_stress += stress_level
            
            # Check for successful adaptation
            if stress_level < self.config['survival_metrics']['stress_threshold']:
                success_count += 1
                
            steps += 1
            state = torch.randn(32)  # Next state
            
        return {
            'stress_level': total_stress / steps,
            'success_rate': success_count / steps,
            'total_steps': steps
        }

if __name__ == '__main__':
    unittest.main()