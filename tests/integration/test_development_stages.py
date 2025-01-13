"""
Integration tests for consciousness development stages in ACM.

This module validates:
1. Stage progression through consciousness development
2. Integration between core components during development
3. Memory formation and emotional context tracking
4. Long-term development stability metrics

Dependencies:
- models/core/consciousness_core.py for main system
- models/emotion/tgnn/emotional_graph.py for emotion processing
- models/memory/emotional_memory_core.py for storage
- configs/consciousness_development.yaml for parameters
"""

from typing import Dict, Optional
import unittest
import torch

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
        """Initialize test components"""
        self.config = DevelopmentTestConfig()
        self.consciousness = ConsciousnessCore(self.config)
        self.monitor = ConsciousnessMonitor(self.config)
        self.memory = EmotionalMemoryCore(self.config)

    def test_stage_progression(self):
        """Test progression through development stages"""
        # Initial state metrics
        initial_metrics = self.monitor.evaluate_current_state()
        self.assertLess(
            initial_metrics['consciousness_score'],
            self.config.consciousness.emergence_threshold,
            "Initial consciousness should be below emergence threshold"
        )
        
        # Process development episodes
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
            
            # Store metrics
            self._log_development_metrics(metrics)

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