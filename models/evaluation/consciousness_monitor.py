"""
Consciousness Development Monitoring System for ACM

This module implements:
1. Tracking of consciousness development metrics
2. Stage transition monitoring
3. Development milestone validation
4. Integration with emotional and memory systems

Dependencies:
- models/core/consciousness_core.py for main system
- models/emotion/tgnn/emotional_graph.py for emotion processing
- models/memory/emotional_memory_core.py for memory validation
"""

from typing import Dict, Any
import torch
import numpy as np
from dataclasses import dataclass
from models.evaluation.levin_consciousness_metrics import LevinConsciousnessEvaluator, LevinConsciousnessMetrics

@dataclass
class ConsciousnessMetrics:
    """Tracks consciousness development metrics."""
    emotional_awareness: float = 0.0
    attention_stability: float = 0.0
    memory_coherence: float = 0.0
    behavioral_adaptation: float = 0.0
    consciousness_score: float = 0.0

class ConsciousnessMonitor:
    def __init__(self, acm_system, config):
        """
        Initialize consciousness monitoring.
        
        Args:
            acm_system: The ACM system instance.
            config: Dictionary of monitoring-related settings and thresholds.
        """
        self.acm = acm_system
        self.config = config
        self.integrated_info_calculator = IntegratedInformationCalculator(self.acm)
        self.global_workspace_tracker = GlobalWorkspaceTracker(self.acm)
        self.perturbation_tester = PerturbationTester(self.acm)
        self.self_awareness_monitor = SelfAwarenessMonitor(self.acm)
        self.metrics = ConsciousnessMetrics()
        self.history = []
        self.levin_evaluator = LevinConsciousnessEvaluator(config)
        self.levin_metrics_history = []
        self.state_history = []  # For morphological adaptation tracking

    def evaluate_state(
        self,
        current_state: Dict[str, torch.Tensor],
        emotional_context: Dict[str, float] = None,
        attention_metrics: Dict[str, float] = None,
        bioelectric_state: Dict[str, Any] = None,
        holonic_output: Dict[str, Any] = None,
        actions: list = None,
        goals: list = None,
        outcomes: list = None,
        component_states: Dict[str, Any] = None
    ) -> Dict[str, float]:
        """
        Evaluate the current consciousness state, updating internal metrics.
        
        Args:
            current_state: Dictionary holding memory and system state tensors.
            emotional_context: Dictionary of emotional readings (valence, arousal, etc.).
            attention_metrics: Dictionary describing attention levels and stability.
            bioelectric_state: Dictionary of bioelectric state readings.
            holonic_output: Dictionary of holonic output readings.
            actions: List of actions taken.
            goals: List of goals.
            outcomes: List of outcomes.
            component_states: Dictionary of component states.
        
        Returns:
            A dictionary of computed consciousness metrics.
        """
        emotional_awareness = self._evaluate_emotional_awareness(emotional_context)
        attention_stability = self._evaluate_attention_stability(attention_metrics)
        memory_coherence = self._evaluate_memory_coherence(current_state)

        self.metrics.emotional_awareness = emotional_awareness
        self.metrics.attention_stability = attention_stability
        self.metrics.memory_coherence = memory_coherence

        consciousness_score = self._calculate_consciousness_score()
        self.metrics.consciousness_score = consciousness_score

        # Record metrics history for trend analysis.
        self.history.append({
            'emotional_awareness': emotional_awareness,
            'attention_stability': attention_stability,
            'memory_coherence': memory_coherence,
            'consciousness_score': consciousness_score
        })

        # Store current state for history
        self.state_history.append(current_state)
        if len(self.state_history) > 50:  # Keep last 50 states
            self.state_history.pop(0)
        
        # Evaluate Levin consciousness metrics
        levin_metrics = self.levin_evaluator.evaluate_levin_consciousness(
            bioelectric_state or {},
            holonic_output or {},
            self.state_history[:-1],  # Past states
            current_state,
            actions or [],
            goals or [],
            outcomes or [],
            component_states or {}
        )
        
        # Store metrics history
        self.levin_metrics_history.append(levin_metrics)
        if len(self.levin_metrics_history) > 100:  # Keep last 100 records
            self.levin_metrics_history.pop(0)
            
        # Return combined metrics
        return {
            **self.get_current_metrics(),  # Existing metrics
            **levin_metrics    # Levin-inspired metrics
        }

    def _evaluate_emotional_awareness(self, emotional_context: Dict[str, float]) -> float:
        """
        Evaluate how well the system understands and integrates emotional inputs.
        Placeholder logic; refine per your architecture.
        """
        valence = emotional_context.get('valence', 0.5)
        arousal = emotional_context.get('arousal', 0.5)
        # Simple average as a placeholder.
        return (valence + arousal) / 2.0

    def _evaluate_attention_stability(self, attention_metrics: Dict[str, float]) -> float:
        """
        Evaluate attention stability from attention_metrics.
        Placeholder logic; refine per your architecture.
        """
        focus = attention_metrics.get('focus', 0.5)
        fluctuation = attention_metrics.get('fluctuation', 0.5)
        # Higher focus + lower fluctuation → higher stability.
        return max(0.0, focus - 0.5 * fluctuation)

    def _evaluate_memory_coherence(self, current_state: Dict[str, torch.Tensor]) -> float:
        """
        Evaluate how coherent current memories are.
        Placeholder logic; refine per your architecture.
        """
        memory_tensor = current_state.get('memory', torch.zeros(1))
        # Simple approach: measure standard deviation or L2-norm as a stand-in for 'coherence'.
        return float(1.0 / (1.0 + torch.std(memory_tensor).item()))

    def _calculate_consciousness_score(self) -> float:
        """
        Compute an overall consciousness score from the partial metrics.
        Placeholder weighting; adjust as per config thresholds.
        """
        ea = self.metrics.emotional_awareness
        as_ = self.metrics.attention_stability
        mc = self.metrics.memory_coherence
        # Basic average.
        return (ea + as_ + mc) / 3.0

    def get_current_metrics(self) -> Dict[str, float]:
        """Return the current consciousness metrics as a dictionary."""
        return {
            'emotional_awareness': self.metrics.emotional_awareness,
            'attention_stability': self.metrics.attention_stability,
            'memory_coherence': self.metrics.memory_coherence,
            'behavioral_adaptation': self.metrics.behavioral_adaptation,
            'consciousness_score': self.metrics.consciousness_score
        }

    def update_metrics(self) -> Dict[str, float]:
        phi_value = self.integrated_info_calculator.compute_phi()
        gwt_score = self.global_workspace_tracker.check_global_workspace_events()
        pci_score = self.perturbation_tester.simulate_and_measure()
        meta_score = self.self_awareness_monitor.evaluate_self_awareness()

        return {
            "IntegratedInformation": phi_value,
            "GlobalWorkspace": gwt_score,
            "PCI": pci_score,
            "SelfAwareness": meta_score
        }
