"""
Integration Tests for Complete Consciousness System

Tests the full consciousness development pipeline including:
1. Memory formation and emotional learning
2. Self-representation development 
3. Temporal coherence maintenance
4. System-wide metrics tracking

Based on the MANN architecture and holonic principles from the research paper.
"""

import unittest
import torch
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass

from models.memory.memory_integration import MemoryIntegrationCore
from models.evaluation.consciousness_metrics import ConsciousnessMetrics
from models.self_model.modular_self_representation import ModularSelfRepresentation
from models.evaluation.consciousness_monitor import ConsciousnessMonitor

@dataclass
class IntegrationTestConfig:
    """Test configuration for full system integration"""
    memory_config = {
        'capacity': 1000,
        'embedding_dim': 768,
        'emotional_dim': 256,
        'attention_threshold': 0.7
    }
    consciousness_config = {
        'coherence_threshold': 0.7,
        'emotional_stability': 0.6,
        'temporal_window': 100
    }
    development_stages = [
        'attention_activation',
        'emotional_learning',
        'self_awareness',
        'narrative_coherence'
    ]

class TestConsciousnessSystem(unittest.TestCase):
    """System-wide integration tests for consciousness development"""

    def setUp(self):
        self.config = IntegrationTestConfig()
        
        # Initialize core components
        self.memory = MemoryIntegrationCore(self.config.memory_config)
        self.consciousness = ConsciousnessMetrics(self.config.consciousness_config)
        self.self_model = ModularSelfRepresentation(self.config.consciousness_config)
        self.monitor = ConsciousnessMonitor(self.config.consciousness_config)

    def test_complete_development_cycle(self):
        """Test full consciousness development cycle"""
        development_metrics = []
        
        # Run development episodes
        for episode in range(10):
            # Generate experience with increasing complexity
            experience = self._generate_experience(episode)
            
            # Process through consciousness pipeline
            consciousness_state = self._process_consciousness_cycle(experience)
            
            # Update self-model
            self_model_update = self.self_model.update(
                current_state=consciousness_state['state'],
                emotional_context=experience['emotion'],
                attention_level=consciousness_state['attention_level']
            )
            
            # Store experience with consciousness context
            self.memory.store_experience(
                experience_data=consciousness_state['state'],
                emotional_context=experience['emotion'],
                consciousness_level=self_model_update['consciousness_level']
            )
            
            # Evaluate development
            metrics = self.monitor.evaluate_development(
                current_state=consciousness_state,
                self_model_state=self_model_update,
                memory_state=self.memory.get_state()
            )
            
            development_metrics.append(metrics)
            
        # Verify development progression
        self._verify_development_progression(development_metrics)

    def _generate_experience(self, episode: int) -> Dict:
        """Generate increasingly complex experiences"""
        return {
            'state': torch.randn(32),
            'emotion': {
                'valence': min(1.0, 0.5 + 0.05 * episode),
                'arousal': 0.7,
                'dominance': min(1.0, 0.4 + 0.05 * episode)
            },
            'attention': min(1.0, 0.6 + 0.04 * episode),
            'narrative': f"Experience {episode} with growing consciousness",
            'complexity_level': episode / 10.0
        }

    def _verify_development_progression(self, metrics_history: List[Dict]):
        """Verify consciousness development progression"""
        initial_metrics = metrics_history[0]
        final_metrics = metrics_history[-1]
        
        # Verify consciousness development
        self.assertGreater(
            final_metrics['consciousness_level'],
            initial_metrics['consciousness_level'],
            "Consciousness level should increase"
        )
        
        # Verify emotional development
        self.assertGreater(
            final_metrics['emotional_awareness'],
            initial_metrics['emotional_awareness'],
            "Emotional awareness should improve"
        )
        
        # Verify memory coherence
        self.assertGreater(
            final_metrics['memory_coherence'],
            initial_metrics['memory_coherence'],
            "Memory coherence should increase"
        )