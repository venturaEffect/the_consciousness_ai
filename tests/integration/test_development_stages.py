"""
Integration tests for consciousness development stages in ACM.

This test suite validates:
1. Stage progression through development cycle
2. Integration between components during development
3. Metric tracking across stages
4. Learning progress validation

Dependencies:
- models/core/consciousness_core.py for main system
- models/evaluation/consciousness_monitor.py for metrics
- models/memory/emotional_memory_core.py for memory storage
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
        """Initialize development stage test components"""
        self.config = DevelopmentTestConfig()
        self.consciousness = ConsciousnessCore(self.config)
        self.monitor = ConsciousnessMonitor(self.config)
        self.memory = EmotionalMemoryCore(self.config)

    def test_stage_progression(self):
        """Test progression through development stages"""
        initial_metrics = self.monitor.evaluate_current_state()
        
        # Run development episodes
        for episode in range(self.config.test_episodes):
            # Generate test scenario 
            scenario = self._generate_test_scenario()
            
            # Process through consciousness system
            result = self.consciousness.process_experience(scenario)
            
            # Evaluate development
            metrics = self.monitor.evaluate_state(
                consciousness_state=result.state,
                emotional_context=result.emotion,
                attention_metrics=result.attention
            )
            
            self._validate_stage_progress(metrics)

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