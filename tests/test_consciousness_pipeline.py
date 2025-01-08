# tests/test_consciousness_pipeline.py

import unittest
import torch
import numpy as np
from typing import Dict, List
from dataclasses import dataclass
from models.fusion.emotional_memory_fusion import EmotionalMemoryFusion
from models.memory.emotional_memory_core import EmotionalMemoryCore
from models.predictive.attention_mechanism import ConsciousnessAttention
from models.evaluation.consciousness_monitor import ConsciousnessMonitor
from simulations.scenarios.consciousness_scenarios import ConsciousnessScenarioManager

@dataclass
class TestConfig:
    """Test configuration for consciousness pipeline"""
    memory_config = {
        'capacity': 1000,
        'embedding_size': 768,
        'attention_threshold': 0.7
    }
    fusion_config = {
        'text_model': 'llama-3.3',
        'vision_model': 'palm-e',
        'audio_model': 'whisper-v3',
        'fusion_hidden_size': 768
    }
    consciousness_thresholds = {
        'emotional_awareness': 0.7,
        'memory_coherence': 0.6,
        'attention_level': 0.8,
        'narrative_consistency': 0.7
    }

class TestConsciousnessPipeline(unittest.TestCase):
    """Test suite for validating consciousness development pipeline"""
    
    def setUp(self):
        self.config = TestConfig()
        
        # Initialize core components
        self.fusion = EmotionalMemoryFusion(self.config.fusion_config)
        self.memory = EmotionalMemoryCore(self.config.memory_config)
        self.attention = ConsciousnessAttention(self.config.memory_config)
        self.monitor = ConsciousnessMonitor(self.config)
        self.scenario_manager = ConsciousnessScenarioManager(self.config)
        
    def test_end_to_end_consciousness_development(self):
        """Test complete consciousness development cycle"""
        
        # Generate development scenarios
        num_episodes = 5
        consciousness_scores = []
        
        for episode in range(num_episodes):
            # Generate stressful scenario
            scenario = self.scenario_manager.generate_scenario(
                scenario_type="survival"
            )
            
            # Process through attention mechanism
            state = torch.randn(32)  # Initial state
            emotion_values = {
                'valence': 0.3,  # Start stressed
                'arousal': 0.8,
                'dominance': 0.4
            }
            
            attention_output, attention_metrics = self.attention.forward(
                input_state=state,
                emotional_context=self.fusion.emotion_network.get_embedding(emotion_values)
            )
            
            # Process multimodal fusion
            fusion_output, fusion_info = self.fusion.forward(
                text_input=torch.randn(1, 32, 768),
                vision_input=torch.randn(1, 32, 768),
                audio_input=torch.randn(1, 32, 768),
                emotional_context=emotion_values
            )
            
            # Store experience
            self.memory.store_experience(
                state=state,
                emotion_values=emotion_values,
                attention_level=attention_metrics['attention_level'],
                context={'fusion_info': fusion_info}
            )
            
            # Monitor development
            evaluation = self.monitor.evaluate_development(
                current_state={'encoded_state': state},
                emotion_values=emotion_values,
                attention_metrics=attention_metrics,
                stress_level=0.7
            )
            
            consciousness_scores.append(evaluation['consciousness_score'])
            
        # Verify consciousness development
        self.assertGreater(
            consciousness_scores[-1],
            consciousness_scores[0],
            "Consciousness should develop over episodes"
        )
        
    def test_stress_induced_attention(self):
        """Test attention activation through stress"""
        
        # Create high-stress state
        state = torch.randn(32)
        emotion_values = {
            'valence': 0.2,  # Very negative
            'arousal': 0.9,  # High arousal
            'dominance': 0.3  # Low dominance
        }
        
        # Process through attention
        attention_output, metrics = self.attention.forward(
            input_state=state,
            emotional_context=self.fusion.emotion_network.get_embedding(emotion_values)
        )
        
        # Verify attention activation
        self.assertGreater(
            metrics['attention_level'],
            self.config.consciousness_thresholds['attention_level']
        )
        
    def test_emotional_memory_formation(self):
        """Test memory formation during high-attention states"""
        
        # Create sequence of emotional experiences
        experiences = []
        base_emotion = {
            'valence': 0.3,
            'arousal': 0.8,
            'dominance': 0.4
        }
        
        for i in range(5):
            experience = {
                'state': torch.randn(32),
                'emotion_values': {
                    'valence': min(1.0, base_emotion['valence'] + 0.1 * i),
                    'arousal': base_emotion['arousal'],
                    'dominance': min(1.0, base_emotion['dominance'] + 0.05 * i)
                },
                'attention_level': 0.8 + 0.02 * i
            }
            experiences.append(experience)
            
            # Store experience
            self.memory.store_experience(**experience)
            
        # Retrieve similar experiences
        similar = self.memory.retrieve_similar_memories(
            emotion_query=experiences[-1]['emotion_values'],
            k=3
        )
        
        self.assertEqual(len(similar), 3)
        self.assertGreater(similar[0]['attention_level'], 0.8)
        
    def test_consciousness_monitoring(self):
        """Test consciousness development monitoring"""
        
        initial_state = {
            'encoded_state': torch.randn(32),
            'emotion': {
                'valence': 0.5,
                'arousal': 0.5,
                'dominance': 0.5
            }
        }
        
        # Monitor initial state
        initial_eval = self.monitor.evaluate_development(
            current_state=initial_state,
            emotion_values=initial_state['emotion'],
            attention_metrics={'attention_level': 0.5},
            stress_level=0.5
        )
        
        # Process experiences
        for _ in range(5):
            state = {
                'encoded_state': torch.randn(32),
                'emotion': {
                    'valence': np.random.uniform(0.3, 0.8),
                    'arousal': np.random.uniform(0.6, 0.9),
                    'dominance': np.random.uniform(0.4, 0.7)
                }
            }
            
            eval_result = self.monitor.evaluate_development(
                current_state=state,
                emotion_values=state['emotion'],
                attention_metrics={'attention_level': 0.8},
                stress_level=0.7
            )
            
        # Verify development progress
        self.assertGreater(
            eval_result['consciousness_score'],
            initial_eval['consciousness_score']
        )

if __name__ == '__main__':
    unittest.main()