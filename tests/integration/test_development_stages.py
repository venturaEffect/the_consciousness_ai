"""
Development Stages Integration Tests

Tests the progression through consciousness development stages:
1. Attention activation
2. Emotional learning
3. Memory coherence
4. Self-representation
"""

import unittest
import torch
from typing import Dict, List
from dataclasses import dataclass

from models.evaluation.consciousness_monitor import ConsciousnessMonitor
from models.evaluation.enhanced_consciousness_metrics import EnhancedConsciousnessEvaluator
from models.memory.memory_integration import MemoryIntegrationCore
from models.self_model.modular_self_representation import ModularSelfRepresentation

@dataclass
class DevelopmentTestConfig:
    """Test configuration for development stages"""
    stage_thresholds = {
        'attention_activation': 0.7,
        'emotional_learning': 0.6,
        'memory_coherence': 0.7,
        'self_awareness': 0.8
    }
    evaluation_window = 100
    min_stage_duration = 50

class TestDevelopmentStages(unittest.TestCase):
    """Tests consciousness development stage progression"""

    def setUp(self):
        self.config = DevelopmentTestConfig()
        self.monitor = ConsciousnessMonitor(self.config)
        self.evaluator = EnhancedConsciousnessEvaluator(self.config)
        self.memory = MemoryIntegrationCore(self.config)
        self.self_model = ModularSelfRepresentation(self.config)

    def test_stage_progression(self):
        """Test progression through development stages"""
        development_history = []
        
        # Run development episodes
        for episode in range(200):
            # Generate increasingly complex experience
            experience = self._generate_staged_experience(episode)
            
            # Process through consciousness pipeline
            consciousness_state = self._process_consciousness_cycle(experience)
            
            # Evaluate development
            metrics = self.evaluator.evaluate_consciousness(
                current_state=consciousness_state,
                memory_state=self.memory.get_state(),
                self_model_state=self.self_model.get_state(),
                emotional_context=experience['emotion']
            )
            
            development_history.append(metrics)
            
            # Verify stage transitions
            if episode > 0:
                self._verify_stage_transition(
                    previous_metrics=development_history[-2],
                    current_metrics=metrics,
                    episode=episode
                )

    def _verify_stage_transition(
        self,
        previous_metrics: Dict,
        current_metrics: Dict,
        episode: int
    ):
        """Verify valid stage transitions"""
        current_stage = current_metrics['development_stage']
        previous_stage = previous_metrics['development_stage']
        
        if current_stage != previous_stage:
            # Verify stage prerequisites
            self._verify_stage_prerequisites(
                current_stage,
                development_history=self.evaluator.development_history
            )
            
            # Verify minimum stage duration
            stage_duration = self._calculate_stage_duration(previous_stage)
            self.assertGreaterEqual(
                stage_duration,
                self.config.min_stage_duration,
                f"Stage {previous_stage} duration too short"
            )